from typing import Dict, List, Tuple, Any, Optional, Set
import logging
import json
import numpy as np
import re
import traceback

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
        self.scene_memory = {}  # VLM-based scene graph memory
        self.goal_subgraph = {}  # LLM-constructed goal subgraph
        self.subtask_history = []  # Chain of subtasks
        self.confidence_map = {}  # Confidence-weighted curiosity map
        self.direction_analysis = {}  # Store directional analysis results
        
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
        
        print("🎯 CoTGraphAgent initialized successfully!")
        logging.info("CoTGraphAgent initialized successfully!")
    
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
            print("📋 阶段1: LLM构建目标子图...")
            goal_subgraph = self._construct_goal_subgraph_via_llm(goal)
            self.goal_subgraph.update(goal_subgraph)
            
            # === 阶段2: VLM进行语义增强的全景分析 ===  
            print("👁️ 阶段2: VLM语义增强全景分析...")
            llm_semantic_hints = self._generate_semantic_hints_for_vlm(goal, goal_subgraph)
            vlm_predictions = self._vlm_enhanced_panoramic_analysis(evaluator_image, goal, llm_semantic_hints)
            
            # === 阶段3: LLM基于VLM反馈进行空间推理 ===
            print("🤔 阶段3: LLM空间推理和置信度评估...")
            spatial_reasoning = self._llm_spatial_reasoning(goal, goal_subgraph, vlm_predictions)
            
            # === 协作融合和最终预测 ===
            print("🔄 VLM-LLM协作融合...")
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
        
        try:
            # Create panoramic image for analysis
            angles = (np.arange(len(pano_images))) * 30
            inference_image = self._concat_panoramic(pano_images, angles)
            
            # Get panoramic analysis using CoT methodology
            response = self._predicting_module(inference_image, goal)
            
            # Extract values in the expected format
            explorable_value = {}
            reason = {}
            
            if response and isinstance(response, dict):
                for angle_str, values in response.items():
                    if isinstance(values, dict):
                        explorable_value[angle_str] = values.get('Score', 5)
                        reason[angle_str] = values.get('Explanation', f'Direction {angle_str}° analysis')
            
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
        return {
            'scene_memory_size': len(self.scene_memory),
            'subtask_history_length': len(self.subtask_history),
            'goal_subgraph_size': len(self.goal_subgraph),
            'confidence_map_size': len(self.confidence_map),
            'current_subtask': self.current_subtask,
            'qa_round': self.qa_round,
            'direction_analysis_size': len(self.direction_analysis)
        }
    
    # === VLM-LLM 协作核心方法 ===
    
    def _construct_goal_subgraph_via_llm(self, goal: str) -> Dict:
        """
        Use LLM to construct a goal subgraph, replacing UniGoal's explicit graph matching
        """
        try:
            subgraph_prompt = f"""
        As a navigation expert, please construct a semantic subgraph for the target "{goal}". Your analysis should include:
        
        1. **Target Room Types**: What rooms typically contain {goal}?
        2. **Spatial Relations**: How are these rooms related to other spaces?
        3. **Navigation Path**: What is the typical sequence for finding a {goal} from an unknown position?
        4. **Visual Cues**: What are key visual landmarks during the search process?
        
        Please respond in the following JSON format:
        {{
            "target_rooms": ["room1", "room2"],
            "spatial_relations": ["relation1", "relation2"], 
            "navigation_sequence": ["step1", "step2", "step3"],
            "visual_cues": ["cue1", "cue2"],
            "confidence": 0.8
        }}
        """
            
            response = self.ReasonLLM.call(subgraph_prompt)
            subgraph = self._parse_goal_subgraph_response(response, goal)
            
            print(f"🎯 Goal subgraph construction complete: {len(subgraph.get('target_rooms', []))} target rooms")
            logging.info(f"Goal subgraph for {goal}: {subgraph}")
            
            return subgraph
            
        except Exception as e:
            print(f"❌ Goal subgraph construction failed: {e}")
            return self._get_fallback_subgraph(goal)
            
        except Exception as e:
            print(f"❌ 目标子图构建失败: {e}")
            return self._get_fallback_subgraph(goal)
    
    def _generate_semantic_hints_for_vlm(self, goal: str, goal_subgraph: Dict) -> str:
        """
        LLM generates semantic hints for VLM to enhance panoramic analysis accuracy
        """
        try:
            target_rooms = goal_subgraph.get('target_rooms', [])
            visual_cues = goal_subgraph.get('visual_cues', [])
            current_subtask = self.current_subtask or "Initial exploration"
            
            hints_prompt = f"""
                Based on the target "{goal}" and current subtask "{current_subtask}", provide directional hints for the vision model:
                
                Target rooms: {target_rooms}
                Visual cues: {visual_cues}
                Recent memory: {list(self.scene_memory.keys())[-3:] if self.scene_memory else "None"}
                
                Generate concise directional hints (30-50 words per direction):
                - What visual features indicate the correct direction?
                - Which spatial layouts suggest proximity to the goal?
                - What exploration strategy should be prioritized?
                
                Response format: **Directional hint**: Specific visual assessment criteria
                """
            
            response = self.ReasonLLM.call(hints_prompt)
            print(f"💡 Semantic hints generated: {len(response)} characters")
            return response
            
        except Exception as e:
            print(f"❌ Semantic hints generation failed: {e}")
            return f"Looking for {goal}: Pay attention to relevant room entrances and furniture layout"
    
    def _vlm_enhanced_panoramic_analysis(self, image, goal: str, semantic_hints: str) -> Dict:
        """
        VLM performs semantically enhanced panoramic analysis, receiving LLM hints for more accurate directional scoring
        """
        try:
            # Create streamlined enhanced prompt without redundancy
            enhanced_prompt = f"""The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts.

            Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used:

            To help you describe the layout of your surrounding, please follow my step-by-step instructions:
            (1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed.
            (2) If the {goal} is found, assign a score of 10.
            (3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered.

            **Additional Context from LLM Spatial Analysis**:
            {semantic_hints}

            **Instructions**:
            - Consider the LLM semantic hints when analyzing each direction
            - Pay special attention to room types and spatial relationships mentioned
            - Include objects visible in each direction and identify room type

            For each direction, provide an explanation for your assigned score. Format your answer in the json {{'30': {{'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>, 'objects': [<list of visible objects>], 'room_type': '<identified room type>'}}, '90': {{...}}, '150': {{...}}, '210': {{...}}, '270': {{...}}, '330': {{...}}}}.

            Answer Example: {{'30': {{'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.', 'objects': ['recliner', 'wall'], 'room_type': 'living room'}}, '90': {{'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.', 'objects': ['dining table', 'chairs'], 'room_type': 'dining room'}}, '150': {{'Score': 7, 'Explanation': 'Open doorway leading to a hallway. High potential for finding bedrooms through this corridor.', 'objects': ['doorway', 'hallway'], 'room_type': 'hallway'}}, '210': {{'Score': 1, 'Explanation': 'Kitchen area with appliances. Unlikely to contain a bed.', 'objects': ['refrigerator', 'counter'], 'room_type': 'kitchen'}}, '270': {{'Score': 4, 'Explanation': 'Partially visible room entrance. Could potentially lead to bedroom area.', 'objects': ['door frame', 'partial view'], 'room_type': 'unknown'}}, '330': {{'Score': 2, 'Explanation': 'Living room area with a recliner. Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.', 'objects': ['recliner', 'furniture'], 'room_type': 'living room'}}}}"""
            
            response = self.PredictVLM.call([image], enhanced_prompt)
            print(f"📝 VLM enhanced analysis response: {len(response)} characters")
            
            # Parse VLM response
            vlm_predictions = self._parse_vlm_enhanced_response(response, goal)
            
            return vlm_predictions
            
        except Exception as e:
            print(f"❌ VLM enhanced analysis failed: {e}")
            return self._get_basic_vlm_predictions(image, goal)
    
    def _llm_spatial_reasoning(self, goal: str, goal_subgraph: Dict, vlm_predictions: Dict) -> Dict:
        """
        LLM performs spatial reasoning based on VLM predictions, providing confidence assessment and exploration strategy
        """
        try:
            # Validate input
            if not isinstance(vlm_predictions, dict):
                print(f"⚠️ Invalid VLM predictions type: {type(vlm_predictions)}")
                return {"spatial_analysis": "Invalid VLM predictions", "recommended_directions": []}
                
            # Build VLM prediction summary
            vlm_summary = []
            for direction, data in vlm_predictions.items():
                # Skip non-direction keys
                if direction in ['confidence', 'overall_confidence']:
                    continue
                    
                # Extract data with proper type checking
                if isinstance(data, dict):
                    score = data.get('Score', data.get('score', 0))  # Try both keys for compatibility
                    objects = data.get('objects', [])
                    room_type = data.get('room_type', 'unknown')
                else:
                    score = self._extract_score_from_any(data)
                    objects = []
                    room_type = 'unknown'
                    
                # Only include high-scoring directions in the summary
                if score >= 5:
                    vlm_summary.append(f"{direction}°: {score} points, {room_type}, objects: {objects}")
            
            # Build prompt with appropriate fallbacks
            target_rooms = goal_subgraph.get('target_rooms', []) if isinstance(goal_subgraph, dict) else []
            nav_sequence = goal_subgraph.get('navigation_sequence', []) if isinstance(goal_subgraph, dict) else []
            current_subtask = self.current_subtask or "Initial exploration"
            
            reasoning_prompt = f"""
            As a spatial reasoning expert, analyze these VLM panoramic results:
    
            **Target**: {goal}
            **Target Rooms**: {target_rooms}
            **Navigation Sequence**: {nav_sequence}
            **Current Subtask**: {current_subtask}
    
            **VLM High-Score Directions**:
            {chr(10).join(vlm_summary) if vlm_summary else "All directions scored low"}
    
            Provide:
            1. Spatial Analysis: What type of space am I in?
            2. Target Distance: How far is the {goal}? (far/medium/near)
            3. Recommended Directions: Which 2-3 directions are most promising?
            4. Confidence Assessment: How certain are you?
            5. Subtask Update: Should I change my current subtask?
    
            Return in JSON format:
            {{
                "spatial_analysis": "current space analysis",
                "target_distance": "far/medium/near",
                "recommended_directions": [
                    {{"direction": "30", "priority": 1, "confidence": 0.8, "reasoning": "reasoning"}},
                    {{"direction": "90", "priority": 2, "confidence": 0.6, "reasoning": "reasoning"}}
                ],
                "subtask_update": "new subtask or 'current'",
                "overall_confidence": 0.7
            }}
            """
            
            # Call LLM for reasoning
            response = self.ReasonLLM.call(reasoning_prompt)
            spatial_reasoning = self._parse_spatial_reasoning_response(response)
            
            # Validate result
            if not isinstance(spatial_reasoning, dict):
                print("⚠️ LLM spatial reasoning returned invalid data type")
                return {"spatial_analysis": "Invalid LLM response", "recommended_directions": []}
                
            rec_directions = spatial_reasoning.get('recommended_directions', [])
            print(f"🧠 Spatial reasoning complete: {len(rec_directions)} recommended directions")
            return spatial_reasoning
            
        except Exception as e:
            print(f"❌ LLM spatial reasoning failed: {e}")
            traceback.print_exc()
            return {"spatial_analysis": "Error in reasoning", "recommended_directions": []}
    
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
                recommended_dirs = {}
            elif 'recommended_directions' not in spatial_reasoning:
                print("⚠️ No recommended directions in spatial reasoning, using VLM only")
                recommended_dirs = {}
            else:
                # Create a lookup dictionary for recommended directions
                recommended_dirs = {}
                for rec in spatial_reasoning.get('recommended_directions', []):
                    if isinstance(rec, dict) and 'direction' in rec:
                        recommended_dirs[rec['direction']] = rec
            
            # Process each direction
            for direction, vlm_data in vlm_predictions.items():
                # Skip non-direction keys
                if direction == 'confidence' or direction == 'overall_confidence':
                    continue
                    
                # Ensure we have dictionary data for each direction
                if isinstance(vlm_data, dict):
                    vlm_score = vlm_data.get('score', 0)
                    vlm_reasoning = vlm_data.get('Explanation', 0)
                    vlm_objects = vlm_data.get('objects', [])
                    vlm_room_type = vlm_data.get('room_type', 'unknown')
                else:
                    # Convert non-dictionary data
                    vlm_score = self._extract_score_from_any(vlm_data)
                    vlm_reasoning = f"Basic score for direction {direction}°"
                    vlm_objects = []
                    vlm_room_type = 'unknown'
                
                # Enhance with LLM reasoning if available
                if direction in recommended_dirs:
                    llm_rec = recommended_dirs[direction]
                    llm_confidence = llm_rec.get('confidence', 0.5) 
                    llm_reasoning = llm_rec.get('reasoning', '')
                    priority = llm_rec.get('priority', 3)
                    
                    # Fusion with weights
                    fused_score = vlm_score * 0.6 + llm_confidence * 10 * 0.4
                    priority_bonus = (4 - priority) * 1.0  # priority 1=+3, 2=+2, 3=+1
                    final_score = min(10, fused_score + priority_bonus)
                    
                    combined_reasoning = f"VLM: {vlm_reasoning} | LLM: {llm_reasoning}"
                    collaboration_type = 'VLM+LLM'
                else:
                    # VLM only
                    final_score = vlm_score * 0.7  # Lower confidence without LLM support
                    combined_reasoning = f"VLM only: {vlm_reasoning}"
                    collaboration_type = 'VLM_only'
                
                # Store final prediction for this direction
                final_predictions[direction] = {
                    'Score': round(final_score, 2),
                    'Explanation': combined_reasoning,
                    'confidence': min(1.0, final_score / 10),
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
    
    def _parse_goal_subgraph_response(self, response: str, goal: str) -> Dict:
        """解析LLM目标子图响应"""
        try:
            # 尝试提取JSON
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_fallback_subgraph(goal)
        except:
            return self._get_fallback_subgraph(goal)
    
    def _parse_vlm_enhanced_response(self, response: str, goal: str) -> Dict:
        """解析VLM增强响应"""
        try:
            # First, let's debug what we're getting
            print(f"🔍 VLM Response Length: {len(response)} characters")
            print(f"🔍 First 200 chars: {response[:200]}")
            
            # Try to parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"🔍 Extracted JSON Length: {len(json_str)} characters")
                print(f"🔍 First 300 chars of JSON: {json_str[:300]}")
                
                # Fix common JSON syntax issues before parsing
                original_json = json_str
                
                # Pre-processing: handle common VLM response formatting issues
                # Fix cases like: {"30": {"Score": 8 "Explanation": "text"
                json_str = re.sub(r'(\d+)\s+(")', r'\1, \2', json_str)  # Number followed by quote
                json_str = re.sub(r'(\"[^\"]*\")\s+(")', r'\1, \2', json_str)  # String followed by quote
                
                json_str = self._fix_json_syntax(json_str)
                
                if json_str != original_json:
                    print(f"🔧 JSON was modified during repair")
                    print(f"🔧 After repair (first 300 chars): {json_str[:300]}")
                
                try:
                    parsed_json = json.loads(json_str)
                    print(f"✅ JSON parsing successful! Keys: {list(parsed_json.keys())}")
                except json.JSONDecodeError as json_err:
                    print(f"⚠️ JSON repair failed: {json_err}")
                    print(f"⚠️ Error position: line {json_err.lineno}, column {json_err.colno}")
                    print(f"⚠️ Character position: {json_err.pos}")
                    
                    # Show specific context around the error
                    lines = json_str.split('\n')
                    if json_err.lineno <= len(lines):
                        error_line = lines[json_err.lineno - 1]
                        print(f"⚠️ Error line {json_err.lineno}: {error_line}")
                        if json_err.colno <= len(error_line):
                            pointer = ' ' * (json_err.colno - 1) + '^'
                            print(f"⚠️ Error position: {pointer}")
                    
                    print(f"⚠️ Context around error: {json_str[max(0, json_err.pos-100):json_err.pos+100]}")
                    
                    # Try a more aggressive repair approach
                    json_str = self._aggressive_json_repair(json_str)
                    try:
                        parsed_json = json.loads(json_str)
                        print(f"✅ Aggressive repair successful!")
                    except Exception as repair_err:
                        print(f"⚠️ Aggressive JSON repair also failed: {repair_err}")
                        # Final fallback: try to extract scores manually from text
                        return self._extract_scores_from_text(response, goal)
                
                # Validate the structure - ensure each direction contains a dictionary
                valid_response = {}
                expected_directions = ['30', '90', '150', '210', '270', '330']  # Match parent class format
                
                for direction in expected_directions:
                    if direction in parsed_json:
                        value = parsed_json[direction]
                        if isinstance(value, dict):
                            valid_response[direction] = value
                        else:
                            # Create proper dictionary structure if we got flat values
                            valid_response[direction] = {
                                'Score': self._extract_score_from_any(value),
                                'Explanation': f"Direction {direction}°",
                                'objects': [],
                                'room_type': 'unknown'
                            }
                    else:
                        # Create fallback for missing direction
                        valid_response[direction] = {
                            'Score': 3,
                            'Explanation': f"Analysis for direction {direction}°",
                            'objects': [],
                            'room_type': 'unknown'
                        }
                
                # Copy confidence if available
                if 'confidence' in parsed_json:
                    valid_response['confidence'] = parsed_json['confidence']
                
                return valid_response
            else:
                print("⚠️ No valid JSON found in VLM response")
                return self._extract_scores_from_text(response, goal)
                
        except Exception as e:
            print(f"⚠️ Failed to parse VLM response: {e}")
            return self._extract_scores_from_text(response, goal)
    
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
        
        # Handle specific early-line issues that cause line 2, column 126 type errors
        # Fix malformed first entries like: {"30": {"Score": 8 "Explanation": ...
        json_str = re.sub(r'(\{\s*"[^"]+"\s*:\s*\{\s*"[^"]+"\s*:\s*\d+)\s+(")', r'\1, \2', json_str)
        
        # More comprehensive fix for missing commas in nested objects
        # Fix patterns like: "Score": 8 "Explanation": "text"
        json_str = re.sub(r'(\d+)\s+("Score"|"Explanation"|"objects"|"room_type")', r'\1, \2', json_str)
        json_str = re.sub(r'(\"[^\"]*\")\s+("Score"|"Explanation"|"objects"|"room_type")', r'\1, \2', json_str)
        
        # Fix specific pattern: "Score": 8, "Explanation": "text" "objects":
        json_str = re.sub(r'(\"[^\"]*\")\s+("objects"|"room_type")', r'\1, \2', json_str)
        
        # Replace missing commas between objects
        json_str = re.sub(r'}\s*{', r'},{', json_str)
        
        # Fix trailing commas before closing brackets
        json_str = re.sub(r',\s*}', r'}', json_str)
        json_str = re.sub(r',\s*]', r']', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Fix single quotes used instead of double quotes
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Fix missing comma between value and next key - more comprehensive
        # Handle cases like: "Score": 8 "Explanation": "text"
        json_str = re.sub(r'(\d+)\s+("[\w_]+"\s*:)', r'\1, \2', json_str)
        json_str = re.sub(r'(\"[^\"]*\")\s+("[\w_]+"\s*:)', r'\1, \2', json_str)
        json_str = re.sub(r'(\])\s+("[\w_]+"\s*:)', r'\1, \2', json_str)
        json_str = re.sub(r'(})\s+("[\w_]+"\s*:)', r'\1, \2', json_str)
        
        # Fix missing quotes around unquoted string values
        json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*[a-zA-Z0-9_])\s*([,}])', 
                         lambda m: f': "{m.group(1).strip()}"{m.group(2)}' 
                         if m.group(1).strip() not in ['true', 'false', 'null'] and not m.group(1).strip().isdigit()
                         else f': {m.group(1).strip()}{m.group(2)}', json_str)
        
        # Fix missing comma between string and opening brace
        json_str = re.sub(r'"\s*{', r'",{', json_str)
        
        # Fix array formatting issues
        json_str = re.sub(r'\[\s*([^,\]]+)\s*([^,\]]+)\s*\]', lambda m: f'["{m.group(1).strip()}", "{m.group(2).strip()}"]', json_str)
        
        # Additional fix for nested object structure issues
        # Handle cases where there's missing comma after nested objects
        json_str = re.sub(r'(}\s*)"(\d+)":', r'\1, "\2":', json_str)
        
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
        """Extract scores from VLM response text when JSON parsing fails"""
        print(f"🔧 Extracting scores from text (fallback method)")
        directions = ['30', '90', '150', '210', '270', '330']  # Match parent class format
        extracted_data = {}
        
        # Try to extract score patterns for each direction
        for direction in directions:
            # Look for patterns like "30°: score 8" or "Direction 30: 8/10" etc.
            patterns = [
                rf'{direction}[°]*\s*:\s*(?:score\s*)?(\d+)',
                rf'{direction}[°]*.*?(\d+)/10',
                rf'{direction}[°]*.*?score[:\s]*(\d+)',
                rf'direction\s*{direction}[°]*.*?(\d+)',
                rf'"{direction}":\s*\{{[^}}]*score["\']?\s*:\s*(\d+)'  # JSON-like pattern
            ]
            
            score = 3  # Default score
            explanation = f"Text analysis for direction {direction}°"
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        extracted_score = int(match.group(1))
                        score = min(10, max(0, extracted_score))
                        print(f"🔧 Found score {score} for direction {direction}° using pattern")
                        break
                    except:
                        continue
            
            # Try to extract explanation text near the direction
            explanation_patterns = [
                rf'{direction}[°]*[:\s]*[^.!?]*?([^.!?]*[.!?])',
                rf'direction\s*{direction}[°]*[:\s]*([^.!?]*[.!?])',
                rf'"{direction}":\s*\{{[^}}]*explanation["\']?\s*:\s*["\']([^"\']*)',
            ]
            
            for pattern in explanation_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_explanation = match.group(1).strip()
                    if len(extracted_explanation) > 10:  # Only use if substantial
                        explanation = extracted_explanation[:100]  # Limit length
                        break
            
            extracted_data[direction] = {
                'Score': score,
                'Explanation': explanation,
                'objects': [],
                'room_type': 'unknown'
            }
        
        extracted_data['confidence'] = 0.4  # Lower confidence for text extraction
        print(f"🔧 Text extraction complete: found {len([d for d in directions if extracted_data[d]['Score'] > 3])} directions with scores > 3")
        return extracted_data
    
    def _parse_spatial_reasoning_response(self, response: str) -> Dict:
        """解析LLM空间推理响应"""
        try:
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"spatial_analysis": "解析失败", "recommended_directions": []}
        except:
            return {"spatial_analysis": "解析失败", "recommended_directions": []}
    
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
        """获取基础VLM预测（无LLM增强）"""
        basic_predictions = {}
        standard_directions = ['30', '90', '150', '210', '270', '330']  # Match parent class format
        for direction in standard_directions:
            basic_predictions[direction] = {
                'Score': 5.0,  # 中等分数
                'Explanation': f'基础分析方向{direction}度',
                'objects': [],
                'room_type': '未知'
            }
        basic_predictions['confidence'] = 0.5
        return basic_predictions


    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', reason: str = '{}', 
                        num_actions: int = 0, scene_graph_context: str = None):
        """
        Enhanced prompt construction with scene graph context support
        """
        if prompt_type == 'goal':
            # Original goal prompt implementation
            return super()._construct_prompt(goal, prompt_type, subtask=subtask, reason=reason, num_actions=num_actions)
            
        if prompt_type == 'stopping':
            # Original stopping prompt implementation
            return super()._construct_prompt(goal, prompt_type, num_actions=num_actions)
            
        if prompt_type == 'predicting':
            # Original predicting prompt implementation with optional scene context
            prompt = super()._construct_prompt(goal, prompt_type)
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
            return super()._construct_prompt(goal, prompt_type, subtask=subtask, num_actions=num_actions)
            
        raise ValueError('Prompt type must be goal, stopping, predicting, planning, or action')
