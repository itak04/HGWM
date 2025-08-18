from typing import Dict, List, Tuple, Any, Optional, Set
import ast
import logging
import json
import numpy as np
import re
import time
import traceback
import habitat_sim

from WMNav_agent import WMNavAgent
from simWrapper import PolarAction
from utils import *
from api import *

print("🚀 CoTGraphAgent loading...")


class CoTGraphAgent(WMNavAgent):
    """
    Chain-of-Thought Graph-inspired Navigation Agent
    
    Implements CL-CoTNav methodology for panoramic navigation:
    - Uses VLM for directional scoring of panoramic images
    - Implements multi-stage reasoning: Perception -> Semantic -> Planning
    - Maintains scene memory and goal-oriented reasoning
    - Replaces explicit graph matching with VLM-LLM collaboration
    """
    
    def __init__(self, cfg: dict):
        print("🚀 CoTGraphAgent initializing...")
        logging.info("CoTGraphAgent initializing...")
        
        # Initialize memory systems first to ensure they exist
        self.cvalue_map = np.ones((self.map_size, self.map_size, 3), dtype=np.float16) * 10
        self.scene_memory = {}  # VLM-based scene graph memory
        self.goal_subgraph = {}  # LLM-constructed goal subgraph
        self.subtask_history = []  # Chain of subtasks
        self.confidence_map = {}  # Confidence-weighted curiosity map
        self.direction_analysis = {}  # Store directional analysis results
        
        # Goal subgraph caching to avoid redundant LLM calls
        self.goal_subgraph_cache = {}  # Cache for goal -> subgraph mapping
        self.last_cached_goal = None  # Track last goal for quick access
        
        # CoT state tracking
        self.current_subtask = None
        self.qa_round = 0
        self.perception_memory = []
        
        # CoT-specific configurations
        self.cot_cfg = cfg.get('cot_cfg', {})
        self.max_qa_rounds = self.cot_cfg.get('max_qa_rounds', 3)
        self.confidence_threshold = self.cot_cfg.get('confidence_threshold', 0.7)
        self.memory_decay_factor = self.cot_cfg.get('memory_decay_factor', 0.9)
        self.panoramic_analysis_mode = self.cot_cfg.get('panoramic_analysis_mode', 'directional_scoring')
        
        # Initialize parent class after setting up our attributes
        super().__init__(cfg)
        
        # Initialize LLM for goal subgraph construction and reasoning
        self._initialize_llm(cfg)
        
        # Initialize unified memory structure for optimized data flow
        self.unified_memory = {
            'goal_subgraphs': {},
            'object_correlations': {},
            'spatial_reasoning_cache': {},
            'vlm_analysis_history': [],
            'performance_metrics': {
                'json_parse_success_rate': 0.0,
                'correlation_cache_hits': 0,
                'avg_processing_time': 0.0
            }
        }
        
        print("🎯 CoTGraphAgent initialized successfully!")
        logging.info("CoTGraphAgent initialized successfully!")
    
    def _update_unified_memory(self, key: str, data: Dict, goal: str = None):
        """
        Update unified memory structure for efficient data management
        """
        if key == 'goal_subgraph':
            self.unified_memory['goal_subgraphs'][goal] = {
                'data': data,
                'timestamp': time.time(),
                'access_count': self.unified_memory['goal_subgraphs'].get(goal, {}).get('access_count', 0) + 1
            }
        elif key == 'object_correlations':
            # Cache correlation scores to avoid recomputation
            cache_key = f"{goal}_{hash(str(sorted(data.keys())))}"
            self.unified_memory['object_correlations'][cache_key] = data
        elif key == 'spatial_reasoning':
            self.unified_memory['spatial_reasoning_cache'][goal] = data
        elif key == 'vlm_analysis':
            self.unified_memory['vlm_analysis_history'].append({
                'goal': goal,
                'data': data,
                'timestamp': time.time()
            })
            # Keep only recent entries to manage memory
            if len(self.unified_memory['vlm_analysis_history']) > 10:
                self.unified_memory['vlm_analysis_history'] = self.unified_memory['vlm_analysis_history'][-10:]
    
    def _initialize_llm(self, cfg: dict):
        """Initialize LLM for goal subgraph construction and reasoning"""
        try:
            llm_cfg = cfg.get('llm_cfg', self.cot_cfg)
            
            # LLM system instruction for goal subgraph and semantic reasoning
            llm_system_instruction = (
                "You are an AI assistant specialized in spatial reasoning and goal decomposition for embodied navigation. "
                "Your role is to: (1) Decompose navigation goals into semantic subgraphs and subtasks, "
                "(2) Provide spatial reasoning based on visual observations, "
                "(3) Generate confidence-weighted guidance for exploration direction. "
                "Always provide structured, actionable responses with clear reasoning chains."
            )
            
            llm_model = llm_cfg.get('llm_model', 'Pro/Qwen/Qwen2.5-7B-Instruct')
            
            # Initialize LLM based on model type
            if 'SiliconFlow' in llm_model:
                from api import SiliconFlowLLM
                self.ReasonLLM = SiliconFlowLLM(
                    model=llm_model.replace('SiliconFlow/', ''),
                    system_instruction=llm_system_instruction
                )
            else:
                # Default to SiliconFlow LLM
                from api import SiliconFlowLLM
                self.ReasonLLM = SiliconFlowLLM(
                    model=llm_model,
                    system_instruction=llm_system_instruction
                )
                
            print("🧠 LLM initialized for goal subgraph construction")
            logging.info(f"LLM initialized: {llm_model}")
            
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            logging.error(f"LLM initialization failed: {e}")
            # Fallback: create a dummy LLM that returns structured responses
            self.ReasonLLM = self._create_fallback_llm()
    
    def _create_fallback_llm(self):
        """Create a fallback LLM that returns structured responses"""
        class FallbackLLM:
            def call(self, prompt):
                return "Fallback LLM response: Goal subgraph construction not available."
            def call_chat(self, prompt):
                return self.call(prompt)
            def get_spend(self):
                return 0
        return FallbackLLM()
    
    def reset(self):
        """Reset agent state for new episode"""
        super().reset()
        
        # Reset CoT-specific state
        self.scene_memory.clear()
        self.goal_subgraph = {}  # Ensure this is initialized as empty dict
        self.subtask_history.clear()
        self.confidence_map.clear()
        self.direction_analysis.clear()
        
        # Clear goal subgraph cache for new episode
        self.goal_subgraph_cache.clear()
        self.last_cached_goal = None
        
        self.current_subtask = None
        self.qa_round = 0
        self.perception_memory.clear()
        
        print("🔄 CoTGraphAgent reset complete!")
    
    def _predicting_module(self, evaluator_image, goal):
        """
        Enhanced predicting module using VLM-LLM collaboration to replace UniGoal graph matching.
        
        三阶段替代图匹配：
        阶段1（零匹配）：LLM构建目标子图，VLM进行地图探索
        阶段2（部分匹配）：LLM基于VLM反馈进行空间推理
        阶段3（完全匹配）：VLM定位目标，LLM验证完成
        """
        print(f"\n{'='*60}")
        print(f"🧠 VLM-LLM协作预测 - 目标: {goal}")
        print(f"{'='*60}")
        
        try:
            # === 阶段1: LLM构建目标子图和初始子任务 ===
            print("阶段1: LLM构建目标子图...")
            goal_subgraph = self._construct_goal_subgraph_via_llm(goal)
            self.goal_subgraph.update(goal_subgraph)
            
            # === 阶段2: VLM进行语义增强的全景分析 ===  
            print("阶段2: VLM语义增强全景分析...")
            llm_semantic_hints = self._generate_semantic_hints_for_vlm(goal, goal_subgraph)
            vlm_predictions = self._vlm_enhanced_panoramic_analysis(evaluator_image, goal, llm_semantic_hints)
            
            # === 阶段3: LLM基于VLM反馈进行空间推理 ===
            print("阶段3: LLM空间推理和置信度评估...")
            spatial_reasoning = self._llm_spatial_reasoning(goal, goal_subgraph, vlm_predictions)
            
            # === 协作融合和最终预测 ===
            print("VLM-LLM协作融合...")
            final_predictions = self._collaborative_prediction_fusion(
                vlm_predictions, spatial_reasoning, goal_subgraph
            )
            
            # 更新记忆系统
            self._update_collaborative_memory(goal, goal_subgraph, vlm_predictions, spatial_reasoning)
            
            print(f"📊 协作预测完成:")
            for direction, data in final_predictions.items():
                if direction != 'confidence' and isinstance(data, dict):
                    confidence = data.get('confidence', 0)
                    explanation = data.get('Explanation', data.get('reasoning', ''))[:50] + '...'
                    print(f"   {direction}: 置信度={confidence:.2f}, 推理={explanation}")
            
            return final_predictions
            
        except Exception as e:
            error_msg = f"VLM-LLM协作预测失败: {str(e)}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            return self._get_fallback_predictions()
    
    def _get_fallback_predictions(self) -> Dict:
        """
        Generate fallback predictions when main analysis fails.
        """
        directions = ['30', '90', '150', '210', '270', '330']
        fallback = {}
        
        for i, direction in enumerate(directions):
            # Prefer forward directions, avoid backwards
            if direction in ['150', '210']:
                score = 2
            elif direction in ['30', '330']:
                score = 6
            else:
                score = 4
                
            fallback[direction] = {
                'Score': score,
                'Explanation': f"Fallback analysis for direction {direction}° - Default navigation score"
            }
        
        return fallback
    
    def make_curiosity_value(self, pano_images, goal):
        """
        Enhanced curiosity value generation using CoT panoramic analysis.
        Integrates with the existing navigation flow.
        """
        print(f"\n🎯 CoT Curiosity Value Generation - Goal: {goal}")
        
        # Initialize inference_image outside try block to avoid UnboundLocalError
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)  # Use instance method
        
        try:
            # Get panoramic analysis using CoT methodology
            response = self._predicting_module(inference_image, goal)
            
            # Extract values in the expected format
            explorable_value = {}
            reason = {}
            
            if response and isinstance(response, dict):
                for angle_str, values in response.items():
                    if isinstance(values, dict):
                        score = values.get('Score', values.get('score', 5))  # Try both cases
                        explorable_value[angle_str] = score
                        reason[angle_str] = values.get('Explanation', values.get('explanation', f'Direction {angle_str}° analysis'))
                        print(f"🔍 Debug: Direction {angle_str}° - Score: {score}, Type: {type(score)}")
                    else:
                        print(f"⚠️ Non-dict values for {angle_str}: {values}")
            
            # Fallback if extraction failed
            if not explorable_value:
                print("⚠️ Using fallback curiosity values")
                for i in range(1, 12, 2):  # 30, 90, 150, 210, 270, 330
                    angle_str = str(i * 30)
                    explorable_value[angle_str] = 5
                    reason[angle_str] = f'Fallback analysis for direction {angle_str}°'
            
            print(f"📊 Curiosity Values Generated:")
            for angle, score in explorable_value.items():
                print(f"  🧭 {angle}°: {score}/10")
            
        except Exception as e:
            print(f"❌ Error in CoT curiosity generation: {e}")
            explorable_value = None
            reason = None
            
        
        return inference_image, explorable_value, reason
    
    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        """
        Enhanced planning using CoT methodology and navigation sequence from goal subgraph.
        """
        try:
            # Extract scene graph context from goal subgraph
            scene_context = ""
            if hasattr(self, 'goal_subgraph') and self.goal_subgraph:
                target_rooms = self.goal_subgraph.get('target_rooms', [])
                if target_rooms:
                    scene_context += f"Target likely in: {', '.join(target_rooms)}\n"
                
                visual_cues = self.goal_subgraph.get('visual_cues', [])
                if visual_cues:
                    scene_context += f"Look for: {', '.join(visual_cues)}\n"
            
            # Make sure pano_images is in the right format
            planning_images = pano_images
            if isinstance(pano_images, list) and len(pano_images) > 0:
                planning_images = [pano_images[0]]
            
            # Create planning prompt with enhanced context
            planning_prompt = self._construct_prompt(
                goal, 
                'planning', 
                previous_subtask, 
                goal_reason,
                scene_graph_context=scene_context
            )
            
            # Call planning VLM
            planning_response = self.PlanVLM.call(planning_images, planning_prompt)
            if not planning_response:
                print("⚠️ Empty planning response from VLM")
                return False, {'description': f"Continue searching for {goal}", 'priority': 'medium'}
            
            # Process response
            planning_response = planning_response.replace('false', 'False').replace('true', 'True')
            dct = self._eval_response(planning_response)
            
            # Handle response parsing
            if not dct:
                print("⚠️ Failed to parse planning response")
                return False, {'description': f"Explore the environment to find {goal}", 'priority': 'medium'}
                
            # Extract values with fallbacks
            try:
                goal_flag = dct.get('Flag', False)
                if not isinstance(goal_flag, bool):
                    goal_flag = str(goal_flag).lower() in ['true', '1', 'yes']
                    
                subtask = dct.get('Subtask', {})
                if not isinstance(subtask, dict):
                    subtask = {'description': str(subtask), 'priority': 'medium'}
            except Exception as e:
                print(f"⚠️ Error extracting planning values: {e}")
                goal_flag = False
                subtask = {'description': f"Continue searching for {goal}", 'priority': 'medium'}
            
            # Update memory
            if subtask:
                if isinstance(subtask, dict) and 'description' in subtask:
                    self.current_subtask = subtask['description']
                else:
                    self.current_subtask = str(subtask)
                    
                self.subtask_history.append({
                    'subtask': subtask,
                    'goal': goal,
                    'reasoning': goal_reason,
                    'step': len(self.subtask_history)
                })
            
            return goal_flag, subtask
                
        except Exception as e:
            print(f"❌ Planning error: {e}")
            traceback.print_exc()
            logging.error(f"Error in CoT planning: {e}")
            # Return a reasonable fallback
            return False, {'description': f"Explore and find {goal}", 'priority': 'high'}
            
    def _identify_current_nav_step(self, subtask_desc: str, nav_sequence: List[str]) -> Optional[int]:
        """
        Identify which step in the navigation sequence the current subtask corresponds to.
        Returns the index in nav_sequence or None if no match found.
        """
        subtask_lower = subtask_desc.lower()
        
        # Try to find exact or partial matches
        for i, step in enumerate(nav_sequence):
            step_lower = step.lower()
            
            # Check for key phrases from this step in the subtask
            key_phrases = step_lower.split()
            for phrase in key_phrases:
                if len(phrase) > 3 and phrase in subtask_lower:
                    return i
        
        # No match found
        return None
    
    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal, scene_graph_context=None):
        """
        Enhanced planning module with scene graph context.
        """
        planning_prompt = self._construct_prompt(
            goal, 
            'planning', 
            previous_subtask, 
            goal_reason,
            scene_graph_context=scene_graph_context
        )
        
        planning_response = self.PlanVLM.call(planning_image, planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)
        
        # Make sure to return the dictionary
        return dct
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization and performance statistics.
        """
        base_stats = {
            'scene_memory_size': len(self.scene_memory),
            'subtask_history_length': len(self.subtask_history),
            'goal_subgraph_size': len(self.goal_subgraph),
            'confidence_map_size': len(self.confidence_map),
            'current_subtask': self.current_subtask,
            'qa_round': self.qa_round,
            'direction_analysis_size': len(self.direction_analysis)
        }
        
        # Add cache performance stats
        cache_stats = self.get_cache_performance()
        base_stats.update(cache_stats)
        
        return base_stats
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """
        Get caching performance statistics to monitor optimization effectiveness.
        """
        total_goals = len(self.goal_subgraph_cache)
        cache_hits = total_goals - 1 if total_goals > 0 else 0  # Subtract 1 for the initial construction
        
        return {
            'cached_goals': total_goals,
            'cache_hits_avoided_llm_calls': cache_hits,
            'last_cached_goal': self.last_cached_goal,
            'unified_memory_size': {
                'goal_subgraphs': len(self.unified_memory['goal_subgraphs']),
                'object_correlations': len(self.unified_memory['object_correlations']),
                'spatial_reasoning_cache': len(self.unified_memory['spatial_reasoning_cache']),
                'vlm_analysis_history': len(self.unified_memory['vlm_analysis_history'])
            }
        }
    
    # === VLM-LLM 协作核心方法 ===
    
    def _construct_goal_subgraph_via_llm(self, goal: str) -> Dict:
        """
        Enhanced goal subgraph construction with object relationship attributes from UniGoal
        Optimized with caching to avoid redundant LLM calls
        """
        try:
            # Check if we have a cached result for this goal
            if goal in self.goal_subgraph_cache:
                print(f"🎯 Using cached goal subgraph for: {goal}")
                self.last_cached_goal = goal
                return self.goal_subgraph_cache[goal]
            
            # Check if this is the same goal as the last one (quick access optimization)
            if self.last_cached_goal == goal and self.goal_subgraph:
                print(f"🎯 Using last cached goal subgraph for: {goal}")
                return self.goal_subgraph
            
            print(f"🧠 Constructing new goal subgraph via LLM for: {goal}")
            
            subgraph_prompt = f"""
            As a navigation expert, construct a semantic subgraph for finding a {goal.upper()}.

            Complete these sections:
            1. Target rooms: List 2-3 rooms most likely to contain {goal}
            2. Related objects: List 4-6 objects commonly found near {goal}
            3. Spatial relations: How these objects typically relate to {goal}
            4. Navigation sequence: Step-by-step path to find {goal}

            For each related object, assign:
            - Correlation score (0-1): How strongly it indicates {goal}'s presence
            - Co-location probability (0-1): Likelihood of being in same room

            Return in JSON format:
            {{
                "target_rooms": ["bedroom", "living room"],
                "related_objects": [
                    {{"name": "nightstand", "correlation": 0.9, "co_location": 0.95}},
                    {{"name": "pillow", "correlation": 0.8, "co_location": 0.9}}
                ],
                "spatial_relations": ["near wall", "elevated from floor"],
                "navigation_sequence": ["find bedroom/living room", "look for seating areas", "check for {goal}"],
                "confidence": 0.8
            }}
            """
            
            # Call LLM only once for this goal
            response = self.ReasonLLM.call(subgraph_prompt)
            subgraph = self._parse_structured_response(response, 
                expected_keys=["target_rooms", "related_objects", "navigation_sequence"])
            
            # Ensure we have a valid dictionary
            if not isinstance(subgraph, dict):
                print(f"⚠️ LLM response parsing failed, using fallback subgraph")
                subgraph = self._get_fallback_subgraph(goal)
            
            # Validate and enhance subgraph structure
            if not subgraph.get('related_objects'):
                subgraph['related_objects'] = []
            
            # Ensure related_objects have required fields
            for obj in subgraph.get('related_objects', []):
                if isinstance(obj, dict):  # Ensure it's a dict before accessing
                    if 'correlation' not in obj:
                        obj['correlation'] = 0.5
                    if 'co_location' not in obj:
                        obj['co_location'] = 0.5
            
            # Cache the result to avoid future LLM calls
            self.goal_subgraph_cache[goal] = subgraph
            self.last_cached_goal = goal
            
            # Update unified memory
            self.unified_memory['goal_subgraphs'][goal] = subgraph
            
            print(f"✅ Goal subgraph constructed and cached for: {goal}")
            print(f"📋 Target rooms: {subgraph.get('target_rooms', [])}")
            print(f"🔗 Related objects: {len(subgraph.get('related_objects', []))}")
            logging.info(f"Goal subgraph for {goal}: {subgraph}")
            
            return subgraph
            
        except Exception as e:
            print(f"❌ Error constructing goal subgraph: {e}")
            logging.error(f"Goal subgraph construction error: {e}")
            
            # Return cached fallback or construct one
            fallback = self._get_fallback_subgraph(goal)
            self.goal_subgraph_cache[goal] = fallback
            return fallback
    
    def _generate_semantic_hints_for_vlm(self, goal: str, goal_subgraph) -> str:
        """
        Generate focused semantic hints for VLM analysis based on goal context
        """
        try:
            # Handle case where goal_subgraph might be a string instead of dict
            if isinstance(goal_subgraph, str):
                print(f"⚠️ Goal subgraph is string, using fallback: {goal_subgraph[:100]}...")
                goal_subgraph = self._get_fallback_subgraph(goal)
            elif not isinstance(goal_subgraph, dict):
                print(f"⚠️ Goal subgraph invalid type: {type(goal_subgraph)}, using fallback")
                goal_subgraph = self._get_fallback_subgraph(goal)
            
            target_rooms = goal_subgraph.get('target_rooms', [])
            spatial_relations = goal_subgraph.get('spatial_relations', [])
            current_subtask = self.current_subtask or "Initial exploration"
            
            # Get recent failed explorations to avoid redundancy
            recent_failures = []
            if hasattr(self, 'subtask_history') and len(self.subtask_history) > 0:
                for h in self.subtask_history[-2:]:
                    if isinstance(h, dict):
                        # Check if subtask is a dict or string
                        if isinstance(h.get('subtask', ''), dict):
                            failure_desc = h['subtask'].get('description', '')
                        else:
                            failure_desc = str(h.get('subtask', ''))
                        if failure_desc:
                            recent_failures.append(failure_desc)
                    elif isinstance(h, str):
                        recent_failures.append(h)
            
            hints_prompt = f"""
            **Navigation Context for {goal}**:
            
            **Target Locations**: {', '.join(target_rooms) if target_rooms else 'Unknown rooms'}
            **Spatial Layout Clues**: {'; '.join(spatial_relations) if spatial_relations else 'Look for typical furniture arrangements'}
            **Current Focus**: {current_subtask}
            
            **Visual Assessment Priorities**:
            1. **Room Type Recognition**: Identify if current area matches target room types
            2. **Object Relationships**: Note spatial arrangements that match expected layouts
            3. **Accessibility Paths**: Prioritize clear passages to unexplored areas
            4. **Target Indicators**: Watch for objects commonly found near {goal}
            
            Recent unsuccessful attempts: {'; '.join(recent_failures) if recent_failures else 'None'}
            
            **Scoring Focus**: Emphasize directions leading to {', '.join(target_rooms)} with {'; '.join(spatial_relations[:2])} arrangements.
            """
            
            print(f"💡 Focused semantic hints generated: {len(hints_prompt)} characters")
            return hints_prompt.strip()
            
        except Exception as e:
            print(f"❌ Semantic hints generation failed: {e}")
            # Use fallback subgraph if goal_subgraph is invalid
            fallback_subgraph = self._get_fallback_subgraph(goal) if isinstance(goal_subgraph, dict) else {}
            target_rooms = fallback_subgraph.get('target_rooms', ['relevant rooms'])
            return f"Navigate to find {goal}. Look for {', '.join(target_rooms)} with appropriate furniture layout."
    
    def _vlm_enhanced_panoramic_analysis(self, image, goal: str, semantic_hints: str) -> Dict:
        """
        VLM performs focused visual analysis with structured scene graph extraction
        """
        try:
            # Enhanced prompt to extract structured scene information
            enhanced_prompt = f"""The agent has been tasked with navigating to a {goal.upper()}. Analyze the panoramic image with red labels showing angles: 30, 90, 150, 210, 270, 330.

        **MANDATORY: You MUST analyze ALL SIX directions (30, 90, 150, 210, 270, 330). Do not skip any direction.**

        **Object Identification Rules**:
        - A chair must have a backrest (not a stool)
        - A chair is NOT a sofa/couch which is NOT a bed
        - Be precise about object identification to avoid confusion
        - You CANNOT GO THROUGH CLOSED DOORS
        - GOING UP OR DOWN STAIRS are not preferred options

        **Scoring Guidelines** (0-10 scale):
        - Target {goal} visible and accessible: 10
        - High relevance objects + clear path to other rooms: 7-9  
        - Medium relevance objects + clear path: 4-6
        - Low relevance objects + clear path: 2-3
        - Clear path but unrelated room: 1-2
        - Dead end or blocked path: 0

        **Enhanced Context**: {semantic_hints}

        **REQUIRED JSON FORMAT** (copy this structure exactly):
        {{
            "30": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}},
            "90": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}},
            "150": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}},
            "210": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}},
            "270": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}},
            "330": {{"Score": <number 0-10>, "Explanation": "<detailed reasoning>", "objects": ["obj1", "obj2"], "spatial_relations": ["rel1", "rel2"], "room_type": "<room_name>", "accessibility": "<accessible/blocked/unclear>", "target_visible": <true/false>}}
        }}

        **Example Response**:
        {{"30": {{"Score": 8, "Explanation": "Bedroom with nightstand visible, high likelihood for {goal}", "objects": ["bed", "nightstand", "lamp"], "spatial_relations": ["nightstand next to bed", "lamp on nightstand"], "room_type": "bedroom", "accessibility": "accessible", "target_visible": false}}}}

        **IMPORTANT**: Return ONLY the JSON object, no extra text before or after."""
            
            response = self.PredictVLM.call([image], enhanced_prompt)
            print(f"📝 VLM structured analysis response: {len(response)} characters")
            
            # Use standard _eval_response method instead of custom parsing
            vlm_predictions = self._eval_response(response)
            
            # Validate that all 6 directions are present
            expected_directions = ['30', '90', '150', '210', '270', '330']
            missing_directions = [d for d in expected_directions if d not in vlm_predictions]
            
            if missing_directions:
                print(f"⚠️ Missing directions {missing_directions}, filling with fallback data")
                # Add fallback data for missing directions
                for direction in missing_directions:
                    vlm_predictions[direction] = {
                        'Score': 1,
                        'Explanation': f'No clear analysis for direction {direction}°, assigning low score',
                        'objects': [],
                        'spatial_relations': [],
                        'room_type': 'unknown',
                        'accessibility': 'unclear',
                        'target_visible': False
                    }
            
            # Extract and store scene graph information
            self._extract_scene_graph_from_vlm(vlm_predictions)
            
            return vlm_predictions
            
        except Exception as e:
            print(f"❌ VLM enhanced analysis failed: {e}")
            return self._get_basic_vlm_predictions(image, goal)
    
    def _llm_spatial_reasoning(self, goal: str, goal_subgraph: Dict, vlm_predictions: Dict) -> Dict:
        """
        Enhanced spatial reasoning with UniGoal-inspired three-phase strategy
        """
        try:
            # Validate input
            if not isinstance(vlm_predictions, dict):
                print(f"⚠️ Invalid VLM predictions (type: {type(vlm_predictions)}), using fallbacks")
                return {'spatial_analysis': 'Invalid input data', 'recommended_directions': [], 'confidence': 0.3}
                
            # === Extract scene objects for graph matching ===
            scene_objects = []
            for direction, data in vlm_predictions.items():
                if direction not in ['confidence', 'overall_confidence'] and isinstance(data, dict):
                    objects = data.get('objects', [])
                    if isinstance(objects, list):
                        scene_objects.extend(objects)
            
            # === Enhanced UniGoal graph overlap calculation ===
            graph_overlap = self._calculate_unigoal_graph_overlap(scene_objects, goal_subgraph, goal)
            overlap_score = graph_overlap['overlap_score']
            exploration_strategy = graph_overlap['exploration_strategy']
            matched_pairs = graph_overlap['matched_pairs']
            components = graph_overlap.get('components', {})
            target_visible = graph_overlap.get('target_visible', False)
            
            print(f"📊 Enhanced graph analysis: S={overlap_score:.3f} | Strategy: {exploration_strategy} | Target visible: {target_visible}")
            
            # === Phase-specific reasoning prompt ===
            phase_context = self._get_phase_specific_context(exploration_strategy, overlap_score, components)
            
            # Build comprehensive VLM summary
            vlm_summary = []
            high_score_directions = []
            for direction, data in vlm_predictions.items():
                if direction in ['confidence', 'overall_confidence']:
                    continue
                    
                if isinstance(data, dict):
                    score = data.get('Score', 0)
                    objects = data.get('objects', [])
                    room_type = data.get('room_type', 'unknown')
                    accessibility = data.get('accessibility', 'unclear')
                    
                    if score >= 7:
                        high_score_directions.append(direction)
                    
                    vlm_summary.append(f"Direction {direction}°: Score={score}, Room={room_type}, Objects={objects[:3]}, Access={accessibility}")
            
            reasoning_prompt = f"""
            **Advanced Spatial Analysis Task**: Navigate to find {goal}
            
            **Current Exploration Phase**: {exploration_strategy}
            {phase_context}
            
            **Graph Matching Analysis**:
            - Overall Similarity: {overlap_score:.2f}/1.0
            - Node Similarity: {components.get('node_similarity', 0):.2f}
            - Relation Similarity: {components.get('relation_similarity', 0):.2f}  
            - Topology Similarity: {components.get('topology_similarity', 0):.2f}
            - Matched Objects: {len(matched_pairs)} pairs
            - Target Visible: {'Yes' if target_visible else 'No'}
            
            **Visual Analysis Results**:
            {chr(10).join(vlm_summary)}
            
            **High-Score Directions**: {', '.join(high_score_directions) if high_score_directions else 'None above 7'}
            
            **Goal Context**:
            - Expected Rooms: {goal_subgraph.get('target_rooms', ['unknown'])}  
            - Navigation Plan: {goal_subgraph.get('navigation_sequence', ['explore'])}
            - Key Objects: {[obj.get('name', 'unknown') if isinstance(obj, dict) else str(obj) for obj in goal_subgraph.get('related_objects', [])[:3]]}
            
            **Phase-Specific Guidance**:
            {self._get_phase_guidance(exploration_strategy)}
            
            **REQUIRED JSON FORMAT**:
            {{
                "spatial_analysis": "<detailed analysis of current space and strategy>",
                "recommended_directions": ["30", "90"],
                "confidence": 0.8,
                "next_subtask": "<specific next action>"
            }}
            
            **Example**:
            {{"spatial_analysis": "Current living room shows good connectivity to other areas", "recommended_directions": ["30", "90"], "confidence": 0.7, "next_subtask": "Explore hallway connections"}}
            
            **IMPORTANT**: Return ONLY the JSON object, no extra text.
            """
            
            response = self.ReasonLLM.call(reasoning_prompt)
            spatial_data = self._parse_spatial_reasoning_response(response)
            
            # Enhance spatial data with graph overlap context
            spatial_data['exploration_strategy'] = exploration_strategy
            spatial_data['overlap_score'] = overlap_score
            spatial_data['matched_pairs'] = matched_pairs
            spatial_data['graph_components'] = components
            spatial_data['target_visible'] = target_visible
            
            print(f"🤔 Phase-aware spatial reasoning complete: {exploration_strategy} phase with {len(spatial_data.get('recommended_directions', []))} directions")
            return spatial_data
            
        except Exception as e:
            print(f"❌ LLM spatial reasoning failed: {e}")
            return {'spatial_analysis': f'Reasoning failed: {e}', 'recommended_directions': [], 'confidence': 0.3}

    def _get_phase_specific_context(self, phase: str, overlap_score: float, components: dict) -> str:
        """
        Generate phase-specific context for spatial reasoning
        """
        if phase == 'frontier_exploration':
            return f"""
            **ZERO MATCHING PHASE** (S < 0.3, current: {overlap_score:.2f})
            - Focus: Discover new areas and rooms
            - Priority: Breadth-first exploration of unseen spaces
            - Strategy: Follow hallways, open doors, explore room entrances
            """
        elif phase == 'anchor_alignment':
            return f"""
            **PARTIAL MATCHING PHASE** (0.3 ≤ S < 0.7, current: {overlap_score:.2f})
            - Focus: Navigate toward identified anchor objects
            - Priority: Move closer to matched object clusters
            - Strategy: Coordinate alignment based on {len(components)} similarity components
            """
        elif phase == 'target_verification':
            return f"""
            **TARGET VERIFICATION PHASE** (S ≥ 0.7, current: {overlap_score:.2f})
            - Focus: Precise target localization and confirmation
            - Priority: Direct navigation to target location
            - Strategy: Final approach and goal verification
            """
        else:
            return f"Standard exploration with context similarity of {overlap_score:.2f}"

    def _get_phase_guidance(self, phase: str) -> str:
        """
        Get specific guidance for each exploration phase
        """
        if phase == 'frontier_exploration':
            return "Prioritize directions leading to unexplored rooms. Avoid recently visited dead ends."
        elif phase == 'anchor_alignment':
            return "Navigate toward directions with highest object relevance scores. Align trajectory with matched anchor objects."
        elif phase == 'target_verification':
            return "Focus on precise target approach. Verify goal object accessibility and prepare for stopping criteria."
        else:
            return "Use standard exploration heuristics with available visual cues."
    
    def _extract_scene_graph_from_vlm(self, vlm_predictions: Dict):
        """
        Extract and maintain structured scene graph from VLM analysis
        """
        try:
            current_scene_graph = {
                'nodes': set(),
                'spatial_relations': [],
                'room_types': {},
                'target_visibility': False
            }
            
            for direction, data in vlm_predictions.items():
                if direction == 'confidence' or not isinstance(data, dict):
                    continue
                    
                # Extract objects (nodes)
                objects = data.get('objects', [])
                current_scene_graph['nodes'].update(objects)
                
                # Extract spatial relations
                relations = data.get('spatial_relations', [])
                current_scene_graph['spatial_relations'].extend(relations)
                
                # Track room types by direction
                room_type = data.get('room_type', 'unknown')
                current_scene_graph['room_types'][direction] = room_type
                
                # Check target visibility
                if data.get('target_visible', False):
                    current_scene_graph['target_visibility'] = True
            
            # Update scene memory
            self.scene_memory['current_graph'] = current_scene_graph
            self.scene_memory['last_relations'] = current_scene_graph['spatial_relations']
            
            print(f"🔄 Scene graph updated: {len(current_scene_graph['nodes'])} nodes, {len(current_scene_graph['spatial_relations'])} relations")
            
        except Exception as e:
            print(f"❌ Scene graph extraction failed: {e}")
            self.scene_memory['current_graph'] = {'nodes': set(), 'spatial_relations': [], 'room_types': {}, 'target_visibility': False}

    def _get_exploration_phase_description(self, strategy: str, overlap_score: float) -> str:
        """
        Generate natural exploration phase description without technical jargon
        """
        if strategy == 'frontier_exploration':
            return f"""Currently in BROAD EXPLORATION mode - the current environment shows low similarity to typical {self.current_subtask or 'target'} locations. Focus on finding new rooms and areas that might contain the target."""
        elif strategy == 'anchor_alignment':
            return f"""Currently in FOCUSED SEARCH mode - some relevant objects have been identified, suggesting we're getting closer to the target area. Navigate toward regions with higher object relevance."""
        elif strategy == 'target_verification':
            return f"""Currently in TARGET VERIFICATION mode - high similarity detected, likely very close to or at the target location. Focus on precise localization and confirmation."""
        else:
            return f"""Standard exploration mode with context similarity of {overlap_score:.2f}. Use available visual cues to guide navigation decisions."""

    def _collaborative_prediction_fusion(self, vlm_predictions: Dict, spatial_reasoning: Dict, goal_subgraph: Dict) -> Dict:
        """
        融合VLM预测和LLM推理，生成最终的协作预测结果
        """
        try:
            final_predictions = {}
            
            # Safety check: ensure vlm_predictions is a dictionary
            if not isinstance(vlm_predictions, dict):
                print(f"⚠️ Invalid VLM predictions (type: {type(vlm_predictions)}), using fallbacks")
                return self._get_fallback_predictions()
            
            # Safety check: ensure spatial_reasoning has expected structure
            if not isinstance(spatial_reasoning, dict):
                print(f"⚠️ Invalid spatial reasoning (type: {type(spatial_reasoning)}), using VLM only")
                recommended_directions_list = []
                spatial_confidence = 0.3
            elif 'recommended_directions' not in spatial_reasoning:
                print("⚠️ No recommended directions in spatial reasoning, using VLM only")
                recommended_directions_list = []
                spatial_confidence = 0.3
            else:
                # LLM returns a simple list of direction strings
                recommended_directions_list = spatial_reasoning.get('recommended_directions', [])
                spatial_confidence = spatial_reasoning.get('confidence', 0.5)
                print(f"🤔 LLM recommended directions: {recommended_directions_list} (confidence: {spatial_confidence})")
            
            # Process each direction
            for direction, vlm_data in vlm_predictions.items():
                # Skip non-direction keys
                if direction == 'confidence' or direction == 'overall_confidence':
                    continue
                    
                # Ensure we have dictionary data for each direction
                if isinstance(vlm_data, dict):
                    vlm_score = vlm_data.get('Score', vlm_data.get('score', 0))  # Try both cases
                    vlm_reasoning = vlm_data.get('Explanation', vlm_data.get('explanation', f"Direction {direction}° analysis"))
                    vlm_objects = vlm_data.get('objects', [])
                    vlm_room_type = vlm_data.get('room_type', 'unknown')
                    print(f"🔍 Fusion Debug - Dir {direction}: VLM Score={vlm_score} (type: {type(vlm_score)})")
                else:
                    # Convert non-dictionary data
                    vlm_score = self._extract_score_from_any(vlm_data)
                    vlm_reasoning = f"Basic score for direction {direction}°"
                    vlm_objects = []
                    vlm_room_type = 'unknown'
                    print(f"🔍 Fusion Debug - Dir {direction}: Non-dict VLM data, extracted score={vlm_score}")
                
                # Enhance with LLM reasoning if available
                if direction in recommended_directions_list:
                    # Direction is recommended by LLM
                    llm_boost_factor = 1.5  # Boost VLM scores for LLM-recommended directions
                    llm_confidence_bonus = spatial_confidence * 3  # Up to 3 points bonus
                    
                    # Fusion: VLM base score + LLM confidence boost
                    fused_score = min(10, vlm_score * llm_boost_factor + llm_confidence_bonus)
                    
                    combined_reasoning = f"VLM+LLM: {vlm_reasoning} (LLM recommended with confidence {spatial_confidence:.2f})"
                    collaboration_type = 'VLM+LLM'
                    print(f"🚀 Direction {direction}° - LLM recommended! VLM:{vlm_score} → Fused:{fused_score:.1f}")
                else:
                    # VLM only, no LLM recommendation
                    fused_score = vlm_score * 0.8  # Slight penalty without LLM support
                    combined_reasoning = f"VLM only: {vlm_reasoning}"
                    collaboration_type = 'VLM_only'
                
                # Store final prediction for this direction
                final_predictions[direction] = {
                    'Score': round(fused_score, 2),
                    'Explanation': combined_reasoning,
                    'confidence': min(1.0, fused_score / 10),
                    'objects': vlm_objects,
                    'room_type': vlm_room_type,
                    'collaboration_type': collaboration_type
                }
            
            # Update subtask if LLM suggests a change
            if isinstance(spatial_reasoning, dict) and spatial_reasoning.get('subtask_update') and spatial_reasoning['subtask_update'] != 'current':
                self.current_subtask = spatial_reasoning['subtask_update']
                print(f"🎯 Task updated: {self.current_subtask}")
            
            return final_predictions
            
        except Exception as e:
            print(f"❌ Collaboration fusion error: {str(e)}")
            traceback.print_exc()
            return self._get_fallback_predictions()
    
    def _update_collaborative_memory(self, goal: str, goal_subgraph: Dict, vlm_predictions: Dict, spatial_reasoning: Dict):
        """
        更新VLM-LLM协作记忆，替代显式图匹配的记忆机制
        """
        try:
            # 更新场景记忆（VLM部分）
            step_key = f"step_{len(self.scene_memory)}"
            self.scene_memory[step_key] = {
                'vlm_predictions': vlm_predictions,
                'spatial_analysis': spatial_reasoning.get('spatial_analysis', ''),
                'timestamp': len(self.scene_memory),
                'goal': goal
            }
            
            # 更新子任务历史
            if self.current_subtask:
                self.subtask_history.append({
                    'subtask': self.current_subtask,
                    'step': len(self.scene_memory),
                    'confidence': spatial_reasoning.get('overall_confidence', 0.5)
                })
            
            # 应用记忆衰减
            if len(self.scene_memory) > 10:  # 保持最近10步记忆
                oldest_keys = list(self.scene_memory.keys())[:-10]
                for key in oldest_keys:
                    del self.scene_memory[key]
            
            print(f"💾 协作记忆更新: 场景记忆{len(self.scene_memory)}步, 子任务历史{len(self.subtask_history)}项")
            
        except Exception as e:
            print(f"❌ 记忆更新失败: {e}")
    
    # === 辅助解析方法 ===
    
    def _parse_structured_response(self, response: str, expected_keys=None) -> Dict:
        """
        Unified JSON parsing with efficient cascading repair strategy
        """
        # Try direct parsing first (fastest)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if expected_keys and not all(k in result for k in expected_keys):
                    raise ValueError("Missing expected keys")
                return result
        except:
            pass
        
        # Try simple fixes (medium cost)
        try:
            cleaned = self._basic_json_cleanup(response)
            result = json.loads(cleaned)
            return result
        except:
            pass
            
        # Resort to structured extraction (highest cost but most reliable)
        return self._extract_structured_data(response, expected_keys)
    
    def _basic_json_cleanup(self, json_str: str) -> str:
        """Basic JSON cleanup for common issues"""
        # Remove non-JSON text before and after braces
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}')
        if start_idx >= 0 and end_idx >= 0:
            json_str = json_str[start_idx:end_idx+1]
        
        # Fix common issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Single to double quotes for keys
        
        return json_str
    
    def _extract_structured_data(self, response: str, expected_keys=None) -> Dict:
        """Extract structured data when JSON parsing fails"""
        result = {}
        
        if expected_keys:
            for key in expected_keys:
                # Try to extract array data
                if key in ['target_rooms', 'related_objects', 'navigation_sequence']:
                    pattern = rf'{key}["\']?\s*:\s*\[([^\]]*)\]'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        items = [item.strip().strip('"\'') for item in match.group(1).split(',')]
                        result[key] = items
                
                # Try to extract single values
                else:
                    pattern = rf'{key}["\']?\s*:\s*["\']?([^,\}}\n]*)["\']?'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        result[key] = match.group(1).strip().strip('"\'')
        
        return result if result else self._get_fallback_subgraph("unknown")

    def _parse_goal_subgraph_response(self, response: str, goal: str) -> Dict:
        """解析LLM目标子图响应 - delegates to unified parser"""
        return self._parse_structured_response(response, 
                                             ['goal_object', 'target_rooms', 'related_objects', 'reasoning'])
    
    def _eval_response(self, response: str) -> Dict:
        """
        Enhanced unified JSON parser that handles multiple formats and edge cases.
        Combines the best of all parsing strategies.
        """
        import re
        import ast
        import json
        
        if not response or not isinstance(response, str):
            return {}
            
        # Clean the response first
        cleaned_response = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        
        # Strategy 1: Try direct JSON parsing (fastest)
        try:
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                if isinstance(parsed_data, dict):
                    return parsed_data
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
            
        # Strategy 2: Try ast.literal_eval with different brace extraction methods
        try:
            # Method 2a: Extract content between outer braces {{}}
            start_idx = cleaned_response.index('{')
            end_idx = cleaned_response.rindex('}')
            if start_idx < end_idx:
                # Try outer content first
                eval_resp = ast.literal_eval(cleaned_response[start_idx + 1:end_idx])
                if isinstance(eval_resp, dict):
                    return eval_resp
        except:
            try:
                # Method 2b: Extract single brace content {}
                eval_resp = ast.literal_eval(cleaned_response[cleaned_response.rindex('{'):cleaned_response.rindex('}') + 1])
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    # Method 2c: Extract full content from first { to last }
                    eval_resp = ast.literal_eval(cleaned_response[cleaned_response.index('{'):cleaned_response.rindex('}')+1])
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    pass
        
        # Strategy 3: Try aggressive JSON repair
        try:
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                repaired_json = self._aggressive_json_repair(json_str)
                if repaired_json:
                    parsed_data = json.loads(repaired_json)
                    if isinstance(parsed_data, dict):
                        return parsed_data
        except:
            pass
        
        # Strategy 4: Fallback - log error and return empty dict
        print(f"⚠️ All JSON parsing strategies failed for response: {response[:200]}...")
        return {}

    def _parse_vlm_enhanced_response(self, response: str, goal: str) -> Dict:
        """Enhanced VLM response parser using unified JSON parsing"""
        try:
            # Use the unified parser
            parsed_data = self._eval_response(response)
            
            if parsed_data and isinstance(parsed_data, dict):
                print("✅ Unified parser successful! Keys:", list(parsed_data.keys()))
                return self._validate_vlm_response(parsed_data)
            else:
                # Fallback to text extraction if parsing failed
                print("🔧 Unified parser failed, falling back to text extraction...")
                return self._extract_scores_from_text(response, goal)
                    
        except Exception as e:
            print(f"❌ VLM response parsing failed: {e}")
            return self._extract_scores_from_text(response, goal)
    
    def _validate_vlm_response(self, parsed_data: Dict) -> Dict:
        """Validate and normalize VLM response data"""
        valid_response = {}
        expected_directions = ['30', '90', '150', '210', '270', '330']
        
        for direction in expected_directions:
            if direction in parsed_data:
                data = parsed_data[direction]
                if isinstance(data, dict):
                    # Ensure consistent key casing
                    score = data.get('Score', data.get('score', 3))
                    valid_response[direction] = {
                        'Score': score,
                        'score': score,
                        'Explanation': data.get('Explanation', data.get('explanation', f"Direction {direction}°")),
                        'objects': data.get('objects', []),
                        'room_type': data.get('room_type', 'unknown'),
                        'correlation_score': data.get('correlation_score', 0.5)
                    }
                else:
                    # Handle non-dict data
                    score = self._extract_score_from_any(data)
                    valid_response[direction] = {
                        'Score': score,
                        'score': score,
                        'Explanation': f"Direction {direction}°",
                        'objects': [],
                        'room_type': 'unknown',
                        'correlation_score': 0.5
                    }
            else:
                # Create fallback for missing direction
                valid_response[direction] = {
                    'Score': 3,
                    'score': 3,
                    'Explanation': f"Missing analysis for direction {direction}°",
                    'objects': [],
                    'room_type': 'unknown',
                    'correlation_score': 0.5
                }
        
        return valid_response
    
    def _fix_json_syntax(self, json_str: str) -> str:
        """Fix common JSON syntax issues"""
        # First, remove any non-JSON text before the opening brace
        start_idx = json_str.find('{')
        if start_idx > 0:
            json_str = json_str[start_idx:]
        
        # Find the last closing brace and truncate everything after
        end_idx = json_str.rfind('}')
        if end_idx > 0:
            json_str = json_str[:end_idx + 1]
        
        # Handle quotes within explanation fields - a common source of errors
        # Look for patterns like: "Explanation": "text with "quotes" inside it"
        explanation_pattern = r'"Explanation"\s*:\s*"([^"]*)"([^"]*)"([^"]*)"'
        json_str = re.sub(explanation_pattern, 
                         lambda m: f'"Explanation": "{m.group(1)}′{m.group(2)}′{m.group(3)}"', 
                         json_str)
        
        # More comprehensive fix for missing commas in nested objects
        json_str = re.sub(r'(\d+)\s+("Score"|"Explanation"|"objects"|"room_type")', r'\1, \2', json_str)
        json_str = re.sub(r'(\"[^\"]*\")\s+("Score"|"Explanation"|"objects"|"room_type")', r'\1, \2', json_str)
        
        # Fix trailing commas before closing brackets
        json_str = re.sub(r',\s*}', r'}', json_str)
        json_str = re.sub(r',\s*]', r']', json_str)
        
        return json_str
    
    def _aggressive_json_repair(self, json_str: str) -> str:
        """More aggressive JSON repair for heavily malformed JSON"""
        try:
            print(f"🔧 Attempting aggressive repair on JSON: {json_str[:200]}...")
            
            # First try: Simple comma fixes for line 2, column ~126 errors
            # These typically happen when there's a missing comma between key-value pairs
            lines = json_str.split('\n')
            if len(lines) >= 2:
                # Check line 2 around column 126 for missing comma patterns
                line2 = lines[1]
                if len(line2) > 100:  # Only if line is long enough
                    # Pattern: "Score": 8 "Explanation" (missing comma)
                    line2 = re.sub(r'(\d+)\s+("[\w_]+"\s*:)', r'\1, \2', line2)
                    # Pattern: "text" "key": (missing comma)
                    line2 = re.sub(r'(\"[^\"]*\")\s+("[\w_]+"\s*:)', r'\1, \2', line2)
                    # Pattern: ] "key": (missing comma)
                    line2 = re.sub(r'(\])\s+("[\w_]+"\s*:)', r'\1, \2', line2)
                    lines[1] = line2
                    json_str = '\n'.join(lines)
                    print(f"🔧 Applied line-specific comma fixes")
            
            # Method 1: Try to extract direction-based patterns more intelligently
            direction_pattern = r'["\']?(\d{1,3})["\']?\s*:\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
            matches = re.findall(direction_pattern, json_str, re.DOTALL)
            
            if matches:
                print(f"🔧 Found {len(matches)} direction patterns")
                # Rebuild JSON from direction matches
                rebuilt_json = "{"
                for i, (direction, content) in enumerate(matches):
                    # Clean the content within each direction
                    score_match = re.search(r'["\']?[Ss]core["\']?\s*:\s*(\d+(?:\.\d+)?)', content)
                    explanation_match = re.search(r'["\']?[Ee]xplanation["\']?\s*:\s*["\']([^"\']*)["\']?', content)
                    
                    score = score_match.group(1) if score_match else "5"
                    explanation = explanation_match.group(1) if explanation_match else f"Direction {direction}° analysis"
                    
                    # Clean explanation text
                    explanation = explanation.replace('"', '\\"')  # Escape internal quotes
                    
                    rebuilt_json += f'"{direction}": {{"Score": {score}, "Explanation": "{explanation}"}}'
                    if i < len(matches) - 1:
                        rebuilt_json += ", "
                
                rebuilt_json += '}'
                print(f"🔧 Rebuilt JSON: {rebuilt_json[:200]}...")
                return rebuilt_json
            
            # Method 2: Try line-by-line reconstruction
            lines = json_str.split('\n')
            if len(lines) > 1:
                print(f"🔧 Trying line-by-line repair with {len(lines)} lines")
                reconstructed = "{"
                entries = []
                
                for line in lines:
                    # Look for direction entries in each line
                    direction_match = re.search(r'["\']?(\d{1,3})["\']?\s*:\s*.*?["\']?[Ss]core["\']?\s*:\s*(\d+)', line)
                    if direction_match:
                        direction = direction_match.group(1)
                        score = direction_match.group(2)
                        explanation = f"Line-extracted analysis for {direction}°"
                        
                        # Try to extract explanation if present
                        explanation_match = re.search(r'["\']?[Ee]xplanation["\']?\s*:\s*["\']([^"\']*)', line)
                        if explanation_match:
                            explanation = explanation_match.group(1)
                        
                        entries.append(f'"{direction}": {{"Score": {score}, "Explanation": "{explanation}"}}')
                
                if entries:
                    reconstructed += ", ".join(entries) + "}"
                    print(f"🔧 Line-reconstructed JSON: {reconstructed[:200]}...")
                    return reconstructed
            
            # Method 3: Fallback to key-value pair extraction (original method)
            key_value_pattern = r'"([^"]*)"\s*:\s*([^,}]+)'
            matches = re.findall(key_value_pattern, json_str)
            
            if not matches:
                print("🔧 No patterns found, returning original")
                return json_str
            
            # Rebuild a clean JSON object
            rebuilt_json = "{"
            for i, (key, value) in enumerate(matches):
                # Clean the value
                value = value.strip()
                if not (value.startswith('"') or 
                        value.startswith('[') or 
                        value.startswith('{') or
                        value in ['true', 'false', 'null'] or
                        re.match(r'^-?\d+(\.\d+)?$', value)):
                    # Add quotes if this is a bare string
                    value = f'"{value.strip()}"'
                
                rebuilt_json += f'"{key}": {value}'
                if i < len(matches) - 1:
                    rebuilt_json += ", "
            
            rebuilt_json += "}"
            print(f"🔧 Final fallback JSON: {rebuilt_json[:200]}...")
            return rebuilt_json
            
        except Exception as e:
            print(f"⚠️ Aggressive repair failed: {e}")
            return json_str
    
    def _extract_score_from_any(self, value):
        """Extract a numeric score from any value type"""
        if isinstance(value, (int, float)):
            return min(10, max(0, value))  # Clamp to 0-10
        elif isinstance(value, str) and value.isdigit():
            return min(10, max(0, int(value)))
        elif isinstance(value, dict) and 'score' in value:
            return min(10, max(0, value['score']))
        else:
            return 5  # Default middle score
    
    def _extract_scores_from_text(self, response: str, goal: str) -> Dict:
        """Enhanced text extraction with better pattern matching"""
        print(f"🔧 Extracting scores from text (enhanced fallback method)")
        directions = ['30', '90', '150', '210', '270', '330']
        extracted_data = {}
        
        # First, try to extract the JSON-like structure patterns
        for direction in directions:
            score = 5  # Default exploration score (higher than before)
            explanation = f"Exploration potential in direction {direction}°"
            objects = []
            room_type = 'unknown'
            
            # Enhanced patterns to capture scores and content
            direction_patterns = [
                # Pattern 1: "30": {"Score": 5, "Explanation": "text"
                rf'"{direction}":\s*\{{\s*"Score":\s*(\d+)(?:,\s*"Explanation":\s*"([^"]*)")?',
                # Pattern 2: Look for the direction followed by score patterns
                rf'{direction}[°]*[:\s]*.*?[Ss]core[:\s]*(\d+)',
                # Pattern 3: Just the number near the direction
                rf'{direction}[°]*.*?(\d+)(?:/10)?',
                # Pattern 4: In quoted context
                rf'"{direction}"[^}}]*(\d+)'
            ]
            
            found_score = False
            for pattern in direction_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        extracted_score = int(match.group(1))
                        score = min(10, max(0, extracted_score))
                        if len(match.groups()) > 1 and match.group(2):
                            explanation = match.group(2)[:100]  # Limit length
                        print(f"✅ Found score {score} for direction {direction}° with pattern")
                        found_score = True
                        break
                    except (ValueError, IndexError):
                        continue
            
            # If no score found, look for exploration keywords
            if not found_score:
                exploration_keywords = [
                    'doorway', 'door', 'hallway', 'corridor', 'path', 'opening', 
                    'entrance', 'passage', 'room', 'area', 'space'
                ]
                
                direction_context = self._extract_direction_context(response, direction)
                
                # Check for exploration potential
                for keyword in exploration_keywords:
                    if keyword.lower() in direction_context.lower():
                        score = 6  # Higher score for exploration potential
                        explanation = f"Potential {keyword} detected for exploration"
                        print(f"✅ Found exploration keyword '{keyword}' for direction {direction}°, score: {score}")
                        break
                
                # Check for dead end indicators
                dead_end_keywords = ['dead end', 'wall', 'blocked', 'no path', 'obstacle']
                for keyword in dead_end_keywords:
                    if keyword.lower() in direction_context.lower():
                        score = 1  # Low score for dead ends
                        explanation = f"Dead end detected: {keyword}"
                        print(f"⚠️ Found dead end keyword '{keyword}' for direction {direction}°, score: {score}")
                        break
            
            extracted_data[direction] = {
                'Score': score,
                'Explanation': explanation,
                'objects': objects,
                'room_type': room_type,
                'correlation_score': 0.5
            }
        
        # Calculate confidence based on how many scores were found
        successful_extractions = sum(1 for d in directions if extracted_data[d]['Score'] != 5)
        confidence = 0.3 + (successful_extractions / len(directions)) * 0.4
        extracted_data['confidence'] = confidence
        
        high_score_count = len([d for d in directions if extracted_data[d]['Score'] > 5])
        print(f"🔧 Enhanced text extraction complete: found {high_score_count} directions with scores > 5")
        return extracted_data
    
    def _extract_direction_context(self, response: str, direction: str) -> str:
        """Extract context around a specific direction from the response"""
        # Look for text around the direction number
        patterns = [
            rf'{direction}[°]*[:\s]*([^,}}]+)',
            rf'"{direction}"[^}}]*?([^}}]+)',
            rf'direction\s*{direction}[°]*[:\s]*([^.!?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                context = match.group(1).strip()
                if len(context) > 5:  # Only return if substantial
                    return context[:200]  # Limit length
        
        return ""
    
    def _parse_spatial_reasoning_response(self, response: str) -> Dict:
        """Parse LLM spatial reasoning response using unified parser with fallback"""
        try:
            parsed_data = self._eval_response(response)
            
            # Validate that we have a dictionary response
            if not isinstance(parsed_data, dict):
                print(f"⚠️ Spatial reasoning response not a dict: {type(parsed_data)}")
                return {"spatial_analysis": "Invalid response format", "recommended_directions": []}
            
            # Ensure we have the minimum required fields
            if 'spatial_analysis' not in parsed_data:
                parsed_data['spatial_analysis'] = "Spatial analysis not provided"
            if 'recommended_directions' not in parsed_data:
                parsed_data['recommended_directions'] = []
            if 'confidence' not in parsed_data:
                parsed_data['confidence'] = 0.5
                
            return parsed_data
            
        except Exception as e:
            print(f"⚠️ All JSON parsing strategies failed for response: {response[:200]}...")
            return {
                "spatial_analysis": f"Parsing failed: {e}",
                "recommended_directions": [],
                "confidence": 0.3
            }
    
    def _calculate_unigoal_graph_overlap(self, scene_objects: List[str], goal_subgraph: Dict, goal: str = None) -> Dict:
        """
        Enhanced UniGoal-inspired graph matching with node, relation, and topology similarity
        Now includes explicit goal object consideration for more accurate phase determination
        """
        if not scene_objects or not goal_subgraph.get('related_objects'):
            return {'overlap_score': 0.0, 'matched_pairs': [], 'exploration_strategy': 'frontier_exploration', 'num_matches': 0}
        
        try:
            # === 0. Goal Object Detection (Primary Signal) ===
            goal_object_detected = False
            goal_object_score = 0.0
            
            if goal:
                # Check if actual goal object is visible in scene
                for scene_obj in scene_objects:
                    similarity = self._calculate_semantic_similarity(scene_obj.lower(), goal.lower())
                    if similarity >= 0.95:  # Exact or near-exact match
                        goal_object_detected = True
                        goal_object_score = 1.0
                        print(f"🎯 GOAL DETECTED: {scene_obj} matches {goal} with similarity {similarity:.3f}")
                        break
                    elif similarity > 0.8:  # Very high similarity but not exact
                        goal_object_score = max(goal_object_score, similarity * 0.8)  # Partial credit
                        print(f"🔍 Strong goal match: {scene_obj} ~ {goal} (similarity: {similarity:.3f})")
                    elif similarity > 0.6:  # Moderate similarity (related objects)
                        goal_object_score = max(goal_object_score, similarity * 0.5)  # Lower credit
                        print(f"🔍 Partial goal match: {scene_obj} ~ {goal} (similarity: {similarity:.3f})")
            
            # === 1. Node Similarity (S_N) - Enhanced with goal awareness ===
            related_objects = goal_subgraph.get('related_objects', [])
            goal_object_names = []
            correlation_weights = {}
            
            for obj in related_objects:
                if isinstance(obj, dict) and 'name' in obj:
                    name = obj['name']
                    goal_object_names.append(name)
                    correlation_weights[name] = obj.get('correlation', 0.5)
                elif isinstance(obj, str):
                    goal_object_names.append(obj)
                    correlation_weights[obj] = 0.5
            
            # Calculate node matches with semantic similarity
            node_matches = []
            total_node_similarity = 0.0
            high_correlation_matches = 0
            
            for scene_obj in scene_objects:
                best_match_score = 0.0
                best_match_goal = None
                best_correlation = 0.0
                
                for goal_obj in goal_object_names:
                    # Semantic similarity calculation
                    similarity = self._calculate_semantic_similarity(scene_obj, goal_obj)
                    correlation = correlation_weights.get(goal_obj, 0.5)
                    weighted_similarity = similarity * correlation
                    
                    if weighted_similarity > best_match_score and weighted_similarity > 0.6:  # τ threshold
                        best_match_score = weighted_similarity
                        best_match_goal = goal_obj
                        best_correlation = correlation
                
                if best_match_goal:
                    node_matches.append({
                        'scene': scene_obj,
                        'goal': best_match_goal,
                        'score': best_match_score,
                        'correlation': best_correlation
                    })
                    total_node_similarity += best_match_score
                    
                    # Count high-correlation matches (strong indicators)
                    if best_correlation > 0.8:
                        high_correlation_matches += 1
            
            # S_N calculation with goal object bonus
            num_possible_pairs = len(scene_objects) * len(goal_object_names)
            base_S_N = total_node_similarity / max(1, num_possible_pairs) if num_possible_pairs > 0 else 0.0
            S_N = base_S_N + (goal_object_score * 0.3)  # Goal object gives significant boost
            
            # === 2. Relation Similarity (S_E) ===
            scene_relations = self.scene_memory.get('last_relations', [])
            goal_relations = goal_subgraph.get('spatial_relations', [])
            
            relation_matches = 0
            if scene_relations and goal_relations:
                for scene_rel in scene_relations:
                    for goal_rel in goal_relations:
                        if self._relations_match(scene_rel, goal_rel):
                            relation_matches += 1
                            break
            
            S_E = relation_matches / max(len(goal_relations), 1) if goal_relations else 0.0
            
            # === 3. Topology Similarity (S_T) - More conservative ===
            central_object = goal or goal_subgraph.get('goal_object', '')
            if not central_object and related_objects:
                # Use highest correlation object as central
                central_object = max(related_objects, key=lambda x: x.get('correlation', 0) if isinstance(x, dict) else 0.5)
                if isinstance(central_object, dict):
                    central_object = central_object.get('name', '')
            
            # Central object bonus (topology indicator)
            central_bonus = 0.0
            target_visible = self.scene_memory.get('current_graph', {}).get('target_visibility', False) or goal_object_detected
            
            if goal_object_detected:
                central_bonus = 0.5  # Very strong topology signal when goal is visible
            elif target_visible:
                central_bonus = 0.3  # Strong topology signal
            elif high_correlation_matches >= 2:
                central_bonus = 0.2  # Multiple strong indicators
            elif any(self._calculate_semantic_similarity(scene_obj, central_object) > 0.7 for scene_obj in scene_objects):
                central_bonus = 0.1  # Single strong indicator
            
            S_T = min(central_bonus + (relation_matches / max(len(scene_relations), 1) if scene_relations else 0), 1.0)
            
            # === Final Overlap Score (S) - Conservative weighting ===
            # More weight on actual goal detection and high-correlation matches
            if goal_object_detected:
                α, β, γ = 0.5,  0.2, 0.3  # Emphasize node similarity when goal found
            elif high_correlation_matches >= 2:
                α, β, γ = 0.4, 0.3, 0.3  # Balanced when multiple strong indicators
            else:
                α, β, γ = 0.3, 0.3, 0.4  # Emphasize topology when few indicators
                
            S = α * S_N + β * S_E + γ * S_T
            
            # === Conservative Phase Determination ===
            # Stricter thresholds to prevent premature target_verification
            if goal_object_detected and goal_object_score >= 0.9:
                strategy = 'target_verification'  # Only when goal clearly visible
            elif S > 0.8 and high_correlation_matches >= 2 and goal_object_score > 0.3:
                strategy = 'target_verification'  # Very high confidence with multiple strong matches + some goal indication
            elif S >= 0.5 and high_correlation_matches >= 1:
                strategy = 'anchor_alignment'     # Medium confidence with at least one strong match
            elif S >= 0.2:
                strategy = 'anchor_alignment'     # Low-medium confidence
            else:
                strategy = 'frontier_exploration' # Low confidence, keep exploring
            
            result = {
                'overlap_score': min(S, 1.0),
                'matched_pairs': node_matches,
                'exploration_strategy': strategy,
                'num_matches': len(node_matches),
                'components': {
                    'node_similarity': S_N,
                    'relation_similarity': S_E,
                    'topology_similarity': S_T,
                    'goal_object_score': goal_object_score
                },
                'central_object_detected': central_bonus > 0,
                'target_visible': target_visible,
                'goal_object_detected': goal_object_detected,
                'high_correlation_matches': high_correlation_matches
            }
            
            print(f"📊 Enhanced goal-aware overlap: S={S:.3f} (S_N={S_N:.3f}, S_E={S_E:.3f}, S_T={S_T:.3f}) → {strategy}")
            print(f"   🎯 Goal detected: {goal_object_detected}, High-corr matches: {high_correlation_matches}, Goal score: {goal_object_score:.3f}")
            return result
            
        except Exception as e:
            print(f"⚠️ Enhanced graph overlap calculation failed: {e}")
            return {'overlap_score': 0.0, 'matched_pairs': [], 'exploration_strategy': 'frontier_exploration', 'num_matches': 0}

    def _calculate_semantic_similarity(self, scene_obj: str, goal_obj: str) -> float:
        """
        Calculate semantic similarity between scene and goal objects
        """
        scene_obj_lower = scene_obj.lower()
        goal_obj_lower = goal_obj.lower()
        
        # Exact match
        if scene_obj_lower == goal_obj_lower:
            return 1.0
        
        # Substring match
        if goal_obj_lower in scene_obj_lower or scene_obj_lower in goal_obj_lower:
            return 0.9
        
        # Word overlap similarity
        scene_words = set(scene_obj_lower.split())
        goal_words = set(goal_obj_lower.split())
        
        if scene_words & goal_words:  # Has intersection
            overlap_ratio = len(scene_words & goal_words) / len(scene_words | goal_words)
            return 0.5 + 0.4 * overlap_ratio
        
        # Semantic category matching (can be enhanced with actual embeddings)
        furniture_categories = {
            'seating': ['chair', 'sofa', 'couch', 'stool', 'bench'],
            'tables': ['table', 'desk', 'counter'],
            'storage': ['cabinet', 'shelf', 'drawer', 'closet'],
            'lighting': ['lamp', 'light', 'chandelier'],
            'bed': ['bed', 'mattress', 'pillow']
        }
        
        scene_category = goal_category = None
        for category, items in furniture_categories.items():
            if any(item in scene_obj_lower for item in items):
                scene_category = category
            if any(item in goal_obj_lower for item in items):
                goal_category = category
        
        if scene_category and goal_category and scene_category == goal_category:
            return 0.6
        
        return 0.2  # Default low similarity

    def _relations_match(self, scene_rel: str, goal_rel: str) -> bool:
        """
        Check if scene relation matches goal relation
        """
        scene_rel_lower = scene_rel.lower()
        goal_rel_lower = goal_rel.lower()
        
        # Direct match
        if scene_rel_lower == goal_rel_lower:
            return True
        
        # Key spatial terms matching
        spatial_synonyms = {
            'near': ['next to', 'beside', 'adjacent', 'close to'],
            'on': ['on top of', 'above', 'over'],
            'in': ['inside', 'within'],
            'behind': ['back of', 'rear'],
            'front': ['in front of', 'ahead of']
        }
        
        for key, synonyms in spatial_synonyms.items():
            if (key in scene_rel_lower or any(syn in scene_rel_lower for syn in synonyms)) and \
               (key in goal_rel_lower or any(syn in goal_rel_lower for syn in synonyms)):
                return True
        
        return False

    def _calculate_object_correlations(self, objects_list: List[str], goal_object: str) -> Dict[str, float]:
        """
        Efficient object correlation calculation without full graph matching
        """
        scores = {}
        goal_lower = goal_object.lower()
        
        for obj in objects_list:
            obj_lower = obj.lower()
            
            # Direct match bonus
            if goal_lower in obj_lower or obj_lower in goal_lower:
                scores[obj] = 0.9
                continue
            
            # Semantic similarity using simple word overlap
            goal_words = set(goal_lower.split())
            obj_words = set(obj_lower.split())
            
            if goal_words & obj_words:  # word intersection
                overlap = len(goal_words & obj_words) / len(goal_words | obj_words)
                scores[obj] = 0.4 + 0.4 * overlap
            else:
                # Default moderate relevance for spatial reasoning
                scores[obj] = 0.3
        
        return scores

    def _get_fallback_subgraph(self, goal: str) -> Dict:
        """获取后备目标子图"""
        fallback_rooms = {
            'bed': ['bedroom', 'master bedroom'],
            'chair': ['living room', 'dining room', 'bedroom'],
            'toilet': ['bathroom', 'restroom'],
            'sofa': ['living room', 'family room']
        }
        
        return {
            'target_rooms': fallback_rooms.get(goal, ['living room']),
            'spatial_relations': ['adjacent to hallway', 'connected to main area'],
            'navigation_sequence': ['explore hallway', f'find {goal} room', f'locate {goal}'],
            'visual_cues': [f'{goal} or related furniture', 'room entrances', 'hallway connections'],
            'confidence': 0.6
        }
    
    def _get_basic_vlm_predictions(self, image, goal: str) -> Dict:
        """Get basic VLM predictions (fallback when enhanced analysis fails)"""
        basic_predictions = {}
        standard_directions = ['30', '90', '150', '210', '270', '330']  # All 6 directions
        
        for direction in standard_directions:
            basic_predictions[direction] = {
                'Score': 2.0,  # Low score for basic fallback
                'Explanation': f'Fallback analysis for direction {direction}° - moderate exploration potential',
                'objects': [],
                'spatial_relations': [],
                'room_type': 'unknown',
                'accessibility': 'unclear',
                'target_visible': False
            }
        
        print(f"🔄 Using basic VLM predictions for {goal} with {len(standard_directions)} directions")
        return basic_predictions


    def _construct_prompt(self, goal=None, prompt_type=None, subtask='{}', reason='{}', 
                         num_actions=0, scene_graph_context=None, **kwargs):
        """
        Enhanced prompt construction with scene graph context support.
        Compatible with both direct calls and parent class mechanism.
        """
        # Handle both positional and keyword arguments for maximum compatibility
        if goal is None and 'goal' in kwargs:
            goal = kwargs.get('goal')
        if prompt_type is None and 'prompt_type' in kwargs:
            prompt_type = kwargs.get('prompt_type')
        
        # For direct calls to parent implementation
        if goal is None or prompt_type is None:
            return super()._construct_prompt(**kwargs)
            
        if prompt_type == 'goal':
            # Original goal prompt implementation
            return super()._construct_prompt(goal=goal, prompt_type=prompt_type, 
                                            subtask=subtask, reason=reason, 
                                            num_actions=num_actions)
                
        if prompt_type == 'stopping':
            # Original stopping prompt implementation
            return super()._construct_prompt(goal=goal, prompt_type=prompt_type, 
                                            num_actions=num_actions)
                
        if prompt_type == 'predicting':
            # Original predicting prompt implementation with optional scene context
            prompt = super()._construct_prompt(goal=goal, prompt_type=prompt_type)
            if scene_graph_context:
                # Add scene graph context if provided
                prompt = prompt.replace(
                    "analyze this panoramic image", 
                    f"analyze this panoramic image with the following context:\n{scene_graph_context}\n"
                )
            return prompt
                
        if prompt_type == 'planning':
            # Enhanced planning prompt with navigation sequence integration
            nav_sequence = []
            if hasattr(self, 'goal_subgraph') and self.goal_subgraph:
                nav_sequence = self.goal_subgraph.get('navigation_sequence', [])
                    
            # Integrate navigation sequence into planning prompt
            nav_context = ""
            if nav_sequence:
                nav_steps = "\n".join([f"- {step}" for step in nav_sequence])
                nav_context = f"\nNavigation guidance for finding {goal}:\n{nav_steps}\n"
                
            # Add scene graph context if provided
            scene_context = f"\nScene context:\n{scene_graph_context}" if scene_graph_context else ""
            
            if reason != '' and subtask != '{}':
                return f"""The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:
        (1)<The observed image>: The image taken from its current location.
        (2){reason}. This explains why you should go in this direction.
        
        Previous subtask: {subtask}
        Current reasoning: {reason}{nav_context}{scene_context}
        
        Your job is to determine if the goal is visible and describe the next place to go.
        
        To help you plan your best next step, follow these guidelines:
        (1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed.
        (2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask.
        (3) If the {goal} is not found and the previous subtask has already been completed, identify a new subtask based on:
           - What room types typically contain {goal}?
           - Which visible pathways (hallways, open doors) might lead to those rooms?
           - What visual cues suggest promising directions?
        
        Note: Pay special attention to open doors and hallways as they can lead to unseen rooms. GOING UP OR DOWN STAIRS is an option.
        
        Return your answer in this JSON format:
        {{
            "Flag": true/false,  # true if goal object is visible, false otherwise
            "Subtask": {{  # Detailed next steps or empty if goal found
                "description": "Detailed navigation guidance",
                "priority": "high/medium/low"
            }}
        }}
        
        Example responses:
        {{"Flag": false, "Subtask": {{"description": "Go to the hallway to explore bedroom areas", "priority": "high"}}}}
        {{"Flag": true, "Subtask": {{"description": "Go to the {goal}", "priority": "high"}}}}
        {{"Flag": false, "Subtask": {{"description": "Go through the open door to explore new rooms", "priority": "medium"}}}}
        """
            else:
                return f"""The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location.
        
        {nav_context}{scene_context}
        
        Your job is to determine if the goal is visible and describe the next place to go.
        
        To help you plan your best next step, follow these guidelines:
        (1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed.
        (2) If the {goal} is not found, analyze the room type and consider:
           - What room types typically contain {goal}?
           - Which visible pathways (hallways, open doors) might lead to those rooms?
           - What visual cues suggest promising directions?
        
        Note: Pay special attention to open doors and hallways as they can lead to unseen rooms. GOING UP OR DOWN STAIRS is an option.
        
        Return your answer in this JSON format:
        {{
            "Flag": true/false,  # true if goal object is visible, false otherwise
            "Subtask": {{  # Detailed next steps or empty if goal found
                "description": "Detailed navigation guidance",
                "priority": "high/medium/low"
            }}
        }}
        
        Example responses:
        {{"Flag": false, "Subtask": {{"description": "Go to the hallway to explore bedroom areas", "priority": "high"}}}}
        {{"Flag": true, "Subtask": {{"description": "Go to the {goal}", "priority": "high"}}}}
        {{"Flag": false, "Subtask": {{"description": "Go through the open door to explore new rooms", "priority": "medium"}}}}
        """
                
        if prompt_type == 'action':
            # Original action prompt implementation
            return super()._construct_prompt(goal=goal, prompt_type=prompt_type, 
                                            subtask=subtask, num_actions=num_actions)
                
         
        raise ValueError('Prompt type must be goal, stopping, predicting, planning, or action')