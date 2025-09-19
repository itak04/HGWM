from typing import Dict, List, Tuple, Any, Optional, Set
import time
import re
import json
import traceback
import numpy as np
import logging
import ast
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
        
        # === 语义体素映射系统 ===
        self.semantic_voxel_map = {}  # 存储物体的语义信息和位置
        self.scene_graph_global = {  # 全局场景图
            'nodes': [],
            'edges': [],
            'spatial_relationships': {},
            'room_mapping': {}
        }
        self.object_position_tracker = {}  # 追踪物体在体素空间中的位置历史
        
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
            
            # Enhanced LLM system instruction for hierarchical room-aware navigation
            llm_system_instruction = (
                "You are an expert in residential navigation and spatial hierarchies. Your role is to construct "
                "HIERARCHICAL room-aware semantic subgraphs for efficient home navigation. "
                "CORE EXPERTISE: "
                "1. Room Hierarchy Analysis: Classify rooms into primary/secondary/adjacent categories based on goal objects "
                "2. Spatial Anchor Detection: Identify key wayfinding objects with room-indicator properties "
                "3. Hallway Navigation: Map room connections and transition strategies "
                "4. Goal-Object Correlation: Analyze spatial relationships between objects and target goals "
                "5. Provide multi-level spatial analysis from room-level to object-level and navigation strategies. Include confidence scores and spatial relationship annotations."
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
            
            # Also assign to self.llm for compatibility
            self.llm = self.ReasonLLM
                
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
    
    def _get_fallback_predictions(self):
        """获取默认的VLM预测结果"""
        fallback_predictions = {}
        directions = [30, 90, 150, 210, 270, 330]
        
        for direction in directions:
            fallback_predictions[str(direction)] = {
                'Score': 1,  # 低分但不是0，保持探索能力
                'Room': 'unknown',
                'Objects': {},
                'Explanation': f'Fallback prediction for direction {direction}°'
            }
        
        fallback_predictions['confidence'] = 0.3
        fallback_predictions['overall_confidence'] = 0.3
        
        return fallback_predictions
    
    def _get_fallback_goal_subgraph(self, goal):
        """获取默认的目标子图"""
        return {
            'target_room_hierarchy': {
                'primary_room': 'unknown',
                'secondary_rooms': [],
                'adjacent_zones': []
            },
            'spatial_anchor_objects': [goal],
            'exploration_strategy': {
                'frontier_priorities': ['forward', 'left', 'right']
            },
            'hallway_navigation': {
                'room_connections': {},
                'wayfinding_objects': []
            }
        }
    
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
        
        图匹配赋分：LLM构建目标子图，VLM进行地图探索
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
            
            vlm_predictions = self._vlm_enhanced_panoramic_analysis(evaluator_image, goal, goal_subgraph)
            
            # 更新记忆系统（只使用VLM预测结果）
            self._update_collaborative_memory(goal, goal_subgraph, vlm_predictions, {})
            
            # 不再存储_last_vlm_predictions，直接返回给调用者
            print(f"📊 VLM预测完成:")
            for direction, data in vlm_predictions.items():
                if direction not in ['confidence', 'overall_confidence'] and isinstance(data, dict):
                    score = data.get('Score', 0)
                    room_type = data.get('Room', data.get('room_type', 'unknown'))
                    objects = data.get('Objects', data.get('objects', []))
                    print(f"   {direction}°: Score={score}, Room={room_type}, Objects={len(objects) if isinstance(objects, (list, dict)) else 0}")
            
            return vlm_predictions, goal_subgraph  # 同时返回VLM预测和目标子图
            
        except Exception as e:
            error_msg = f"VLM-LLM协作预测失败: {str(e)}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            # 返回默认的回退预测结果
            return self._get_fallback_predictions(), self._get_fallback_goal_subgraph(goal)
    
    def make_curiosity_value(self, pano_images, goal):
        """
        Enhanced curiosity value generation using CoT panoramic analysis.
        Integrates with the existing navigation flow and provides data for enhanced update_curiosity_value.
        """
        print(f"\n🎯 CoT Curiosity Value Generation - Goal: {goal}")
        
        # 存储当前目标供后续使用
        self._current_goal = goal
        
        # Initialize variables to prevent UnboundLocalError
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)  # Use instance method
        explorable_value = {}
        reason = {}
        vlm_predictions = None
        goal_subgraph = None
        
        try:
            # Get panoramic analysis using CoT methodology with both VLM predictions and goal subgraph
            vlm_predictions, goal_subgraph = self._predicting_module(inference_image, goal)
            
            # Extract values in the expected format
            # Define expected direction keys to filter out metadata fields
            expected_directions = ['30', '90', '150', '210', '270', '330']
            metadata_fields = ['spatial_analysis', 'recommended_directions', 'overall_confidence', 
                             'exploration_strategy', 'overlap_score', 'confidence']
            
            if vlm_predictions and isinstance(vlm_predictions, dict):
                for angle_str, values in vlm_predictions.items():
                    # Skip metadata fields that are not directional data
                    if angle_str in metadata_fields:
                        continue
                        
                    # Only process expected direction keys or numeric angle strings
                    if angle_str in expected_directions or (angle_str.isdigit() and angle_str in ['30', '90', '150', '210', '270', '330']):
                        if isinstance(values, dict):
                            score = values.get('Score', values.get('score', 5))  # Try both cases
                            explorable_value[angle_str] = score
                            reason[angle_str] = values.get('Explanation', values.get('explanation', f'Direction {angle_str}° analysis'))
                            print(f"🔍 Debug: Direction {angle_str}° - Score: {score}, Type: {type(score)}")
                        else:
                            print(f"⚠️ Non-dict values for {angle_str}: {values}")
                    # Silently skip non-direction keys that are not in metadata_fields (for future extensibility)
            
        except Exception as e:
            print(f"❌ Error in CoT curiosity generation: {e}")
            logging.error(f"make_curiosity_value error: {str(e)}")
            # Reset variables for fallback processing
            vlm_predictions = None
            goal_subgraph = None
            
        # Fallback if extraction failed or error occurred
        if not explorable_value:
            print("⚠️ Using fallback curiosity values")
            for i in range(1, 12, 2):  # 30, 90, 150, 210, 270, 330
                angle_str = str(i * 30)
                explorable_value[angle_str] = 5
                reason[angle_str] = f'Fallback analysis for direction {angle_str}°'
        
        print(f"📊 Curiosity Values Generated:")
        for angle, score in explorable_value.items():
            print(f"  🧭 {angle}°: {score}/10")
        
        # 存储VLM预测和目标子图供update_curiosity_value使用
        self._current_vlm_predictions = vlm_predictions
        self._current_goal_subgraph = goal_subgraph
        
        return inference_image, explorable_value, reason
    
    def make_plan(self, pano_images, previous_subtask, goal_reason, goal, direction_scene_data=None):
        """
        
        Args:
            pano_images: Panoramic images (usually the target direction image)
            previous_subtask: Previous subtask information
            goal_reason: Reasoning for the current goal direction
            goal: Target object to navigate to
            direction_scene_data: Scene graph data from the target direction (from update_curiosity_value)
                                 Expected format: {
                                     'spatial_objects': [...],
                                     'current_graph': {...},
                                     'vlm_predictions': {...},
                                     'goal_rotate': int
                                 }
        """
        try:
            # Extract scene graph context from goal subgraph and direction-specific data
            scene_context = ""
            
            # Add direction-specific scene graph information
            if direction_scene_data:
                spatial_objects = direction_scene_data.get('spatial_objects', [])
                current_graph = direction_scene_data.get('current_graph', {})
                vlm_predictions = direction_scene_data.get('vlm_predictions', {})
                goal_rotate = direction_scene_data.get('goal_rotate', 0)
                
                scene_context += f"\nCurrent direction ({goal_rotate * 30}°) scene analysis:\n"
                
                # Add spatial objects information
                if spatial_objects:
                    obj_names = [obj.get('name', 'unknown') for obj in spatial_objects if isinstance(obj, dict)]
                    scene_context += f"- Detected objects: {', '.join(obj_names[:5])}\n"  # Show first 5 objects
                
                # Add scene graph information
                if current_graph:
                    room_nodes = current_graph.get('room_nodes', [])
                    object_nodes = current_graph.get('object_nodes', [])
                    if room_nodes:
                        room_types = [room.get('type', 'unknown') for room in room_nodes if isinstance(room, dict)]
                        scene_context += f"- Room context: {', '.join(room_types)}\n"
                    if object_nodes:
                        nearby_objects = [obj.get('name', 'unknown') for obj in object_nodes[:3] if isinstance(obj, dict)]
                        scene_context += f"- Nearby objects: {', '.join(nearby_objects)}\n"
                
                # Store direction scene data for potential use in action selection
                self._current_direction_scene_data = direction_scene_data
            
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
                    
                subtask_desc = dct.get('Subtask', '')
                if isinstance(subtask_desc, str):
                    subtask = {'description': subtask_desc, 'priority': 'medium'}
                elif isinstance(subtask_desc, dict):
                    subtask = subtask_desc
                else:
                    subtask = {'description': str(subtask_desc), 'priority': 'medium'}
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
    
    # === VLM-LLM 协作核心方法 ===
    def _construct_goal_subgraph_via_llm(self, goal: str) -> Dict:
        """
        hierarchical room-aware goal subgraph construction for efficient navigation
        Includes room hierarchy, hallway connections, and spatial relationship awareness
        """
        try:
            # Check if we have a cached result for this goal
            if goal in self.goal_subgraph_cache:
                print(f"🎯 Using cached hierarchical goal subgraph for: {goal}")
                self.last_cached_goal = goal
                return self.goal_subgraph_cache[goal]
            
            print(f"🏠 Constructing hierarchical room-aware subgraph for: {goal}")
            
            # Build the prompt using string concatenation to avoid f-string brace conflicts
            subgraph_prompt = (
                "You are an expert in residential navigation and spatial understanding. "
                f"Create a comprehensive navigation strategy for efficiently locating a {goal.upper()} in a typical home environment.\n\n"
                
                f"NAVIGATION STRATEGY ANALYSIS FOR {goal.upper()}:\n\n"
                
                "1. TARGET ROOM ANALYSIS:\n"
                f"- Primary room: The most likely room type where {goal} is typically found\n"
                f"- Secondary rooms: 1-2 alternative room types where {goal} might also be located\n"
                "- Adjacent zones: Connected areas that provide visual access or pathways to target rooms\n\n"
                
                "2. PATHWAY and CONNECTIVITY MAPPING:\n"
                "- Hallway connections: Common routes to reach target rooms from main areas\n"
                "- Visual access points: Doorways/openings that help identify target rooms\n"
                "- Navigation landmarks: Key reference points for wayfinding\n\n"
                
                "3. TARGET-ORIENTED SPATIAL ANCHORS and NAVIGATION CUES:\n"
                f"- Directional anchors: 3-4 objects that DIRECTLY indicate where {goal} is likely located (not just co-occurring objects)\n"
                f"- Visual pathways: Objects that form visual lines leading toward typical {goal} placement areas\n"
                f"- Distance indicators: Objects that help estimate proximity to {goal} based on typical room layouts\n"
                f"- Positioning cues: Furniture/fixtures that indicate the specific area within a room where {goal} is usually found\n\n"
                
                "4. SYSTEMATIC EXPLORATION APPROACH:\n"
                "- Exploration priorities: Which areas to check first when searching\n"
                "- Recovery strategies: What to do when initial search paths fail\n\n"
                
                "Return EXACT JSON format:\n"
                "{\n"
                "    'target_room_hierarchy': {\n"
                "        'primary_room': 'specific_room_type',\n"
                "        'secondary_rooms': ['alternative1', 'alternative2'],\n"
                "        'adjacent_zones': ['connecting_area1', 'connecting_area2']\n"
                "    },\n"
                "    'hallway_navigation': {\n"
                "        'room_connections': {\n"
                "            'hallway_to_primary': {'from': 'hallway', 'to': 'primary_room', 'visual_cues': ['doorway', 'opening']},\n"
                "            'living_to_primary': {'from': 'living_area', 'to': 'primary_room', 'visual_cues': ['arch', 'passage']}\n"
                "        },\n"
                "        'wayfinding_objects': ['stairs', 'kitchen_island', 'main_door']\n"
                "    },\n"
                "    'spatial_anchor_objects': {\n"
                "        'object1': {'name': 'directional_guide_object1', 'guidance_type': 'points_toward_target', 'proximity_indicator': 'close', 'direction_confidence': 0.95},\n"
                "        'object2': {'name': 'pathway_marker_object2', 'guidance_type': 'forms_visual_line', 'proximity_indicator': 'medium', 'direction_confidence': 0.90},\n"
                "        'object3': {'name': 'positioning_reference_object3', 'guidance_type': 'indicates_target_zone', 'proximity_indicator': 'adjacent', 'direction_confidence': 0.88}\n"
                "    },\n"
                "    'exploration_strategy': {\n"
                "        'frontier_priorities': ['unexplored_doorways', 'partial_room_views', 'connecting_passages'],\n"
                "    },\n"
                "    'overlap_weights': {\n"
                "        'room_hierarchy_match': 0.35,\n"
                "        'anchor_objects_detected': 0.30,\n"
                "        'hallway_connectivity': 0.20,\n"
                "        'spatial_arrangement': 0.15\n"
                "    },\n"
                "    'confidence': 0.92\n"
                "}\n\n"
                
                "EXAMPLE for 'chair':\n"
                "{\n"
                "    'target_room_connection': {\n"
                "        'primary_room': 'living_room',\n"
                "        'secondary_rooms': ['dining_room', 'bedroom'],\n"
                "        'adjacent_zones': ['hallway', 'kitchen_area']\n"
                "    },\n"
                "    'hallway_navigation': {\n"
                "        'room_connections': {\n"
                "            'hallway_to_living': {'from': 'hallway', 'to': 'living_room', 'visual_cues': ['wide_opening', 'no_door']},\n"
                "            'kitchen_to_dining': {'from': 'kitchen', 'to': 'dining_room', 'visual_cues': ['archway', 'partial_wall']}\n"
                "        },\n"
                "        'wayfinding_objects': ['staircase', 'kitchen_counter', 'front_door']\n"
                "    },\n"
                "    'spatial_anchor_objects': {\n"
                "        'sofa': {'name': 'sofa', 'guidance_type': 'points_toward_target', 'proximity_indicator': 'adjacent', 'direction_confidence': 0.95},\n"
                "        'coffee_table': {'name': 'coffee_table', 'guidance_type': 'forms_visual_line', 'proximity_indicator': 'between_agent_and_target', 'direction_confidence': 0.90},\n"
                "        'side_table': {'name': 'side_table', 'guidance_type': 'indicates_target_zone', 'proximity_indicator': 'immediately_beside', 'direction_confidence': 0.88}\n"
                "    },\n"
                "    'exploration_strategy': {\n"
                "        'frontier_priorities': ['living_room_doorways', 'dining_area_openings', 'bedroom_entrances'],\n"
                "    },\n"
                "    'overlap_weights': {\n"
                "        'room_hierarchy_match': 0.30,\n"
                "        'anchor_objects_detected': 0.35,\n"
                "        'hallway_connectivity': 0.20,\n"
                "        'spatial_arrangement': 0.15\n"
                "    },\n"
                "    'confidence': 0.92\n"
                "}\n\n"
                
                "Provide ONE definitive navigation analysis based on typical residential layouts and common object placement patterns."
            )

            
            # Call LLM only once for this goal
            response = self.ReasonLLM.call(subgraph_prompt)
            subgraph = self._eval_response(response)
            
            # Ensure we have a valid dictionary
            if not isinstance(subgraph, dict):
                print(f"⚠️ LLM response parsing failed for goal: {goal}, got {type(subgraph)}")
                logging.error(f"LLM response parsing failed for goal: {goal}, got {type(subgraph)}")
                # Try to get a fallback goal subgraph
                subgraph = self._get_fallback_goal_subgraph(goal)
                if not isinstance(subgraph, dict):
                    print(f"❌ Fallback subgraph also invalid, returning empty dict")
                    return {}
            
            # Cache the result to avoid future LLM calls
            self.goal_subgraph_cache[goal] = subgraph
            self.last_cached_goal = goal
            
            # Update unified memory with hierarchical data - ensure it's a dict
            try:
                if isinstance(subgraph, dict):
                    self.unified_memory['goal_subgraphs'][goal] = {
                        'data': subgraph,
                        'timestamp': time.time(),
                        'access_count': self.unified_memory['goal_subgraphs'].get(goal, {}).get('access_count', 0) + 1,
                        'type': 'hierarchical_room_aware'
                    }
                    # Also store directly for backward compatibility
                    self.unified_memory['goal_subgraphs'][f"{goal}_direct"] = subgraph
                else:
                    print(f"⚠️ Cannot update unified memory: subgraph is not a dict ({type(subgraph)})")
            except Exception as e:
                print(f"⚠️ Error updating unified memory: {e}")
                logging.error(f"Unified memory update error: {e}")
            
            print(f"✅ Goal subgraph constructed and cached for: {goal}")
            
            # Debug: Print key parts of the subgraph with correct field names and safe access
            try:
                target_room_hierarchy = subgraph.get('target_room_hierarchy', {}) if isinstance(subgraph, dict) else {}
                spatial_anchors = subgraph.get('spatial_anchor_objects', {}) if isinstance(subgraph, dict) else {}
                primary_room = target_room_hierarchy.get('primary_room', 'unknown') if isinstance(target_room_hierarchy, dict) else 'unknown'
                secondary_rooms = target_room_hierarchy.get('secondary_rooms', []) if isinstance(target_room_hierarchy, dict) else []
                
                print(f"📋 Primary room: {primary_room}")
                print(f"🏠 Secondary rooms: {list(secondary_rooms) if hasattr(secondary_rooms, '__iter__') and not isinstance(secondary_rooms, str) else [secondary_rooms]}")
                print(f"🔗 Spatial anchor objects: {len(spatial_anchors) if isinstance(spatial_anchors, dict) else 0}")
                
                # Show anchor objects for debugging  
                if isinstance(spatial_anchors, dict) and spatial_anchors:
                    anchor_names = []
                    for key, value in spatial_anchors.items():
                        if isinstance(value, dict):
                            anchor_names.append(value.get('name', key))
                        else:
                            anchor_names.append(str(key))
                    print(f"⚓ Anchor objects: {anchor_names[:3]}...")
            except Exception as e:
                print(f"⚠️ Error in debug output: {e}")
            
            logging.info(f"Goal subgraph for {goal}: {subgraph}")
            
            return subgraph
            
        except Exception as e:
            print(f"❌ Error constructing goal subgraph: {e}")
            logging.error(f"Goal subgraph construction error: {e}")
            return {}
    
    def llm_call(self, prompt: str) -> str:
        """
        Make LLM call with proper error handling
        """
        try:
            if hasattr(self, 'ReasonLLM') and self.ReasonLLM:
                return self.ReasonLLM.call(prompt)
            elif hasattr(self, 'llm') and self.llm:
                return self.llm.generate(prompt)
            else:
                print("⚠️ No LLM instance available")
                return '{"spatial_analysis": "LLM not available", "confidence": 0.3}'
        except Exception as e:
            print(f"⚠️ LLM call error: {e}")
            return '{"spatial_analysis": "LLM call failed", "confidence": 0.3}'

    def _normalize_subgraph_data(self, data, data_type='list'):
        """
        Normalize subgraph data to handle both old and new formats
        
        Args:
            data: The data to normalize (can be list, set, dict, or other)
            data_type: Expected type - 'list', 'dict', or 'anchor_objects'
            
        Returns:
            Normalized data in the expected format
        """
        if data_type == 'list':
            # Convert sets to lists, keep lists as-is
            if isinstance(data, set):
                return list(data)
            elif isinstance(data, list):
                return data
            else:
                return []
                
        elif data_type == 'anchor_objects':
            # Handle spatial_anchor_objects in both old list and new dict formats
            if isinstance(data, dict):
                # New dictionary format: convert to list of objects
                return [obj_data for obj_data in data.values() if isinstance(obj_data, dict)]
            elif isinstance(data, list):
                # Old list format: return as-is
                return data
            else:
                return []
                
        elif data_type == 'dict':
            # Ensure data is a dictionary
            if isinstance(data, dict):
                return data
            else:
                return {}
        
        return data

    def _vlm_enhanced_panoramic_analysis(self, image, goal: str, goal_subgraph: Dict) -> Dict:
        """
        VLM performs hierarchical room-aware analysis with detailed spatial object information
        for goal graph matching. Generates a single evaluator prompt that embeds
        semantic hints derived from the goal_subgraph and asks the VLM to score each
        panoramic direction.
        """
        # 1. Extract subgraph fields
        th = goal_subgraph.get('target_room_hierarchy', {})
        hn = goal_subgraph.get('hallway_navigation', {})
        anchors = goal_subgraph.get('spatial_anchor_objects', [])
        es = goal_subgraph.get('exploration_strategy', {})

        # Helper function to handle both sets and lists
        def _ensure_list(value):
            """Convert sets to lists for compatibility, keep lists as-is"""
            return self._normalize_subgraph_data(value, 'list')

        primary = th.get('primary_room', 'unknown')
        secondary = _ensure_list(th.get('secondary_rooms', []))
        adjacent = _ensure_list(th.get('adjacent_zones', []))

        connections = hn.get('room_connections', {})
        wayfinding = _ensure_list(hn.get('wayfinding_objects', []))

        # Handle both old list format and new dict format for anchor objects
        normalized_anchors = self._normalize_subgraph_data(anchors, 'anchor_objects')
        if isinstance(anchors, dict):
            anchor_names = list(anchors.keys())
        else:
            anchor_names = [obj['name'] for obj in normalized_anchors if isinstance(obj, dict) and 'name' in obj]

        frontiers = _ensure_list(es.get('frontier_priorities', []))
        
        # Build connections string - handle both dict and list formats
        connections_str = []
        if isinstance(connections, dict):
            # New dictionary format
            for conn_key, conn_data in connections.items():
                if isinstance(conn_data, dict):
                    from_room = conn_data.get('from', '')
                    to_room = conn_data.get('to', '')
                    visual_cues = conn_data.get('visual_cues', set())
                    cues_str = ','.join(_ensure_list(visual_cues)) if visual_cues else 'none'
                    connections_str.append(f"{from_room}→{to_room} via {cues_str}")
        elif isinstance(connections, list):
            # Old list format (backward compatibility)
            for c in connections:
                if isinstance(c, dict):
                    from_room = c.get('from', '')
                    to_room = c.get('to', '')
                    visual_cues = c.get('visual_cues', [])
                    cues_str = ','.join(visual_cues) if visual_cues else 'none'
                    connections_str.append(f"{from_room}→{to_room} via {cues_str}")

        evaluator_prompt = (
            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f"Context for {goal.upper()}: Primary room is {primary}, secondary rooms are {', '.join(secondary) if secondary else 'none'}, adjacent zones include {', '.join(adjacent) if adjacent else 'none'}. "
            f"Navigation paths: {', '.join(connections_str)}. "
            f"Key landmarks: {', '.join(wayfinding) if wayfinding else 'none'}. "
            f"Spatial anchors to detect: {', '.join(anchor_names) if anchor_names else 'none'}. "
            f"Search priorities: {', '.join(frontiers)}. "
            f"Your job is to assign a score to each direction (0 to 10) indicating how promising it is for finding the {goal}. "
            f"Follow these instructions:\n"
            # f"(1) List all visible objects with their detailed spatial positions:\n"
            # f"   - Use precise positioning: far-left, center-left, center, center-right, far-right\n"
            # f"   - Include depth information: foreground (1-2m), mid-ground (2-4m), background (4m+)\n"
            # f"   - Note spatial relationships between objects (e.g., 'beside', 'on top of', 'under')\n"
            f"(1) Infer the current room type from furniture and features.\n"
            # f"(2) Identify anchor objects that can guide navigation toward the {goal}. Focus on objects that indicate proximity to target rooms or provide directional cues. \n"
            f"(2) Judge path clarity—ignore closed doors and stairs.\n\n"
            f"(3) Scoring Guidelines:\n"
            f"   (4a) Only if the {goal} is found in image, assign a score of 10.\n"
            f"   (4b) If the path is a dead end and it is clear that the target is not in sight, assign a score of 0.\n"
            f"   (4c) If no goal-related objects are visible but the view leads to an unexplored area (clear turn, open door, hallway), However, if you see objects related to the {goal}, first infer the current room type, then assign a score 1-9 based on how strongly these objects suggest the {goal} can be found nearby. assign a score reflecting your common-sense estimate of finding the {goal} there.Note: a chair must have a backrest and is not a stool; a chair is not a sofa (couch), and a sofa is not a bed.\n"
            f"   (4d) Treat any clear turn, open door, or unobstructed hallway as an opportunity to explore further—score higher for more promising paths.\n\n"
            f"Json Format:\n"
            f"{{'30': {{'Score': <0-10>, 'Objects': {{'object1': 'object1 position with depth and relationships', 'object2': 'object2 position with depth and relationships'}}, "
            f"'ObjectRelationships': [spatial relationships between objects], "
            f"'DirectionalAnchors': ['anchor_objects_that_guide_toward_target'], 'Room': 'room_type', 'Path': 'accessibility_assessment', "
            f"'Explanation': 'An explanation for your assigned score.'}},\n"
            f" '90': {{...}}, '150': {{...}}, '210': {{...}}, '270': {{...}}, '330': {{...}}}}\n\n"
            f"Answer Example:\n"
            f"{{\n"
            f" '30': {{'Score': 1, 'Objects': {{'recliner': 'center-right foreground beside sofa', 'side_table': 'right_edge mid-ground'}}, "
            f"'ObjectRelationships': ['recliner beside sofa'], 'DirectionalAnchors': ['side_table points toward bedroom'], 'Room': 'living_room', 'Path': 'Dead end', "
            f"'Explanation': 'Seating corner in the living room with furniture but no pathways. No access to adjacent rooms.'}},\n"
            f" '90': {{'Score': 3, 'Objects': {{'dining_table': 'center mid-ground', 'chairs': 'perimeter foreground'}}, "
            f"'ObjectRelationships': ['dining_table surrounded by chairs'], 'DirectionalAnchors': ['dining_table indicates dining_room area'], 'Room': 'dining_area', 'Path': 'Clear doorway', "
            f"'Explanation': 'Dining room with proper furniture and accessible doorway.'}},\n"
            f" '150': {{'Score': 6, 'Objects': {{'flooring': 'bottom background', 'door_frame': 'far_end mid-ground'}}, "
            f"'ObjectRelationships': ['door_frame facing corridor'], 'DirectionalAnchors': ['door_frame leads toward target rooms'], 'Room': 'hallway', 'Path': 'Unobstructed passage', "
            f"'Explanation': 'Hallway connecting multiple rooms, with clear passage.'}},\n"
            f" ...\n"
            f"}}"
        )


        # 4. Send to VLM and process response
        response = self.PredictVLM.call([image], evaluator_prompt)
        print(f"📝 VLM analysis response: {len(response)} characters")
        print(f"🔍 First 500 chars of VLM response: {response[:500]}")
        
        # Debug: Check if response contains expected direction patterns
        expected_directions = ['30', '90', '150', '210', '270', '330']
        direction_mentions = sum(1 for d in expected_directions if d in response)
        print(f"🎯 Direction angle mentions in response: {direction_mentions}/6")
        
        # Check for JSON structure indicators
        json_indicators = ['{', '}', ':', 'Score', 'Objects', 'Room']
        json_count = sum(1 for indicator in json_indicators if indicator in response)
        print(f"📊 JSON structure indicators found: {json_count}/6")
        
        # Parse response using enhanced method
        vlm_predictions = self._eval_response(response)
        print(f"🎯 VLM parsing result: {len(vlm_predictions)} keys found")
        print(f"📊 Parsed keys: {list(vlm_predictions.keys())}")
        
        # Enhanced debugging for parsed data structure
        if isinstance(vlm_predictions, dict):
            # Check for direction keys vs object lists
            direction_keys = [k for k in vlm_predictions.keys() if k.isdigit() and int(k) in [30, 90, 150, 210, 270, 330]]
            object_keys = [k for k in vlm_predictions.keys() if not k.isdigit() and k not in ['Score', 'Room', 'Likelihood', 'Path', 'Explanation']]
            
            print(f"🧭 Direction keys found: {direction_keys}")
            print(f"🏠 Object keys found: {object_keys}")
            
            if direction_keys:
                print(f"✅ Successfully parsing directional data structure")
                # Show sample direction data
                sample_dir = direction_keys[0]
                sample_data = vlm_predictions.get(sample_dir, {})
                print(f"📍 Sample direction {sample_dir}: {str(sample_data)[:100]}...")
            elif object_keys:
                print(f"⚠️ Warning: Detected object-based structure instead of directional")
                print(f"🔄 Attempting directional extraction...")
                # Try directional extraction with expected directions parameter
                directional_data = self._extract_directional_json(response, expected_directions)
                if directional_data != vlm_predictions:
                    print(f"🎯 Directional extraction produced different result")
                    print(f"📊 New keys: {list(directional_data.keys())}")
                    vlm_predictions = directional_data
                else:
                    print(f"❌ Directional extraction couldn't improve result")
        
        # Validate that we got directional data, not object data
        expected_directions = ['30', '90', '150', '210', '270', '330']
        valid_directions = [k for k in vlm_predictions.keys() if k in expected_directions]
        
        if len(valid_directions) == 0:
            print(f"❌ VLM response parsing failed - got object data instead of directional data")
            print(f"🔍 Raw response sample: {response[:300]}...")
            
            # Try to extract directional JSON from the response
            vlm_predictions = self._extract_directional_json(response, expected_directions)
            
            if not vlm_predictions:
                print("🔧 Directional extraction failed, using fallback predictions")
                vlm_predictions = self._get_fallback_predictions()
        
        print(f"🎯 Final VLM predictions: {len(vlm_predictions)} directions")
        
        # Validate and enhance predictions with spatial object extraction
        for direction in expected_directions:
            if direction not in vlm_predictions:
                print(f"⚠️ Missing direction {direction}, creating default entry")
                vlm_predictions[direction] = {
                    'Score': 3.0,
                    'Objects': {},
                    'Anchors': [],
                    'Room': 'unknown',
                    'Likelihood': 'moderate',
                    'Path': 'unclear',
                    'Explanation': f'Default entry for missing direction {direction}'
                }
            else:
                # Extract detailed spatial information from structured VLM data
                direction_data = vlm_predictions[direction]
                
                # Ensure direction_data is a dictionary before processing
                if not isinstance(direction_data, dict):
                    print(f"⚠️ Direction {direction} data is not a dict (type: {type(direction_data)}), creating default structure")
                    vlm_predictions[direction] = {
                        'Score': 3.0,
                        'Objects': {},
                        'Anchors': [],
                        'Room': 'unknown',
                        'Likelihood': 'moderate',
                        'Path': 'unclear',
                        'Explanation': f'Non-dict data converted for direction {direction}: {str(direction_data)[:100]}...'
                    }
                    direction_data = vlm_predictions[direction]
                
                objects, spatial_relations, room_type, spatial_anchors_detected = self._extract_spatial_details(direction_data)
                
                # Get explanation for compatibility checks
                explanation = direction_data.get('Explanation', '') if isinstance(direction_data, dict) else ''
                likelihood = direction_data.get('Likelihood', '') if isinstance(direction_data, dict) else ''
                path_info = direction_data.get('Path', '') if isinstance(direction_data, dict) else ''
                
                # Normalize structured data using existing normalization methods
                objects_dict = direction_data.get('Objects', {}) if isinstance(direction_data, dict) else {}
                objects_list = self._normalize_subgraph_data(objects_dict, 'list') if objects_dict else objects
                
                anchors_set = direction_data.get('Anchors', set()) if isinstance(direction_data, dict) else set()
                anchors_list = self._normalize_subgraph_data(anchors_set, 'list')
                
                # Enhance prediction with extracted spatial data
                if isinstance(direction_data, dict):
                    direction_data.update({
                        'objects': objects_list,
                        'spatial_relations': spatial_relations, 
                        'room_type': direction_data.get('Room', room_type),
                        'accessibility': 'accessible' if any(keyword in path_info.lower() + explanation.lower() 
                                                           for keyword in ['clear', 'unobstructed', 'open']) else 'unclear',
                        'target_visible': goal.lower() in explanation.lower(),
                        'spatial_anchors_detected': spatial_anchors_detected if spatial_anchors_detected else anchors_list,
                        'goal_likelihood': self._extract_likelihood_from_data(direction_data, explanation)
                    })
        
        print(f"🏠 VLM analysis complete: {len([d for d in vlm_predictions.keys() if isinstance(vlm_predictions[d], dict) and vlm_predictions[d].get('Score', 0) > 5])} high-score directions")
        
        return vlm_predictions

    def _extract_directional_json(self, response: str, expected_directions: List[str]) -> Dict:
        """
        Extract directional JSON data from VLM response when standard parsing fails.
        Specifically looks for direction angle keys (30, 90, 150, etc.)
        """
        import re
        
        print(f"🔍 Attempting directional JSON extraction from response...")
        
        # Strategy 1: Look for patterns like '30': {... or "30": {...
        directional_data = {}
        
        for direction in expected_directions:
            # Enhanced pattern to match nested structures: '30': {... complex content ...}
            pattern = rf"['\"]?{direction}['\"]?\s*:\s*\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}"
            matches = re.finditer(pattern, response, re.DOTALL)
            
            for match in matches:
                match_text = match.group(0)
                try:
                    # Extract just the direction part with better parsing
                    direction_content = match_text.split(':', 1)[1].strip()
                    
                    # Try to parse the content as a complete direction entry
                    parsed = self._eval_response(f"{{'{direction}': {direction_content}}}")
                    if parsed and direction in parsed:
                        directional_data[direction] = parsed[direction]
                        print(f"✅ Successfully extracted direction {direction}")
                        break
                    else:
                        # Try alternative format
                        parsed = self._eval_response(f'{{{direction}: {direction_content}}}')
                        if parsed and direction in parsed:
                            directional_data[direction] = parsed[direction]
                            print(f"✅ Successfully extracted direction {direction} (alt format)")
                            break
                except Exception as e:
                    print(f"⚠️ Failed to parse direction {direction}: {e}")
                    # Try a simpler extraction
                    try:
                        # Extract just the content and create a basic structure
                        if 'Score' in match_text:
                            score_match = re.search(r'Score[\'"]?\s*:\s*(\d+(?:\.\d+)?)', match_text)
                            if score_match:
                                score = float(score_match.group(1))
                                directional_data[direction] = {
                                    'Score': score,
                                    'Objects': {},
                                    'Anchors': [],
                                    'Room': 'unknown',
                                    'Likelihood': 'moderate',
                                    'Path': 'unclear',
                                    'Explanation': f'Partially extracted data for direction {direction}'
                                }
                                print(f"✅ Partially extracted direction {direction} with score {score}")
                                break
                    except:
                        continue
        
        # Strategy 2: Look for complete JSON blocks containing direction keys
        if len(directional_data) < 3:  # Need at least some directions
            print(f"🔍 Strategy 1 yielded only {len(directional_data)} directions, trying Strategy 2...")
            
            # Find all JSON-like blocks in the response
            json_blocks = re.findall(r'\{[^}]*(?:\{[^}]*\}[^}]*)*\}', response, re.DOTALL)
            
            for block in json_blocks:
                try:
                    parsed_block = self._eval_response(block)
                    if parsed_block and isinstance(parsed_block, dict):
                        # Check if this block contains direction keys
                        block_directions = [k for k in parsed_block.keys() if k in expected_directions]
                        if len(block_directions) >= 3:  # Found a good directional block
                            print(f"✅ Found directional block with {len(block_directions)} directions")
                            return parsed_block
                except Exception as e:
                    continue
        
        # Strategy 3: Regex-based extraction for each direction
        if len(directional_data) < 3:
            print(f"🔍 Strategy 2 failed, trying regex extraction...")
            
            for direction in expected_directions:
                if direction not in directional_data:
                    # Look for patterns like: 30: Score: 5, Objects: {...}, Room: kitchen
                    pattern = rf"{direction}[:\s]*(?:.*?Score[:\s]*(\d+(?:\.\d+)?))?"
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    
                    if match:
                        score = float(match.group(1)) if match.group(1) else 3.0
                        directional_data[direction] = {
                            'Score': score,
                            'Objects': {},
                            'Anchors': [],
                            'Room': 'unknown',
                            'Likelihood': 'moderate',
                            'Path': 'unclear',
                            'Explanation': f'Regex-extracted data for direction {direction}'
                        }
                        print(f"✅ Regex-extracted direction {direction} with score {score}")
        
        print(f"🎯 Directional extraction complete: {len(directional_data)} directions extracted")
        return directional_data

    def _extract_likelihood_from_data(self, direction_data: Dict, explanation: str) -> str:
        """
        Extract goal likelihood from structured VLM data or explanation
        """
        # Ensure direction_data is a dictionary
        if not isinstance(direction_data, dict):
            print(f"⚠️ Warning: direction_data is not a dict in _extract_likelihood_from_data (type: {type(direction_data)})")
            direction_data = {}
            
        # First try structured Likelihood field
        likelihood = direction_data.get('Likelihood', '').lower() if isinstance(direction_data, dict) else ''
        
        if 'high' in likelihood:
            return 'high'
        elif 'moderate' in likelihood or 'medium' in likelihood:
            return 'medium'  
        elif 'low' in likelihood:
            return 'low'
        else:
            # Fallback to explanation analysis
            explanation_lower = explanation.lower() if isinstance(explanation, str) else ''
            if 'high' in explanation_lower:
                return 'high'
            elif 'moderate' in explanation_lower or 'medium' in explanation_lower:
                return 'medium'
            else:
                return 'low'

    def _extract_spatial_details(self, vlm_direction_data: Dict) -> Tuple[List[str], List[str], str, List[str]]:
        """
        Extract detailed spatial information from VLM direction data using structured format
        Now uses the new structured VLM output format with dictionaries and sets
        """
        try:
            objects = []
            spatial_relations = []
            room_type = 'unknown'
            spatial_anchors_detected = []
            
            # Extract from structured VLM output
            if isinstance(vlm_direction_data, dict):
                # Extract room type directly
                room_type = vlm_direction_data.get('Room', 'unknown')
                
                # Extract objects and positions from Objects dict
                objects_dict = vlm_direction_data.get('Objects', {})
                if isinstance(objects_dict, dict):
                    # New structured format: {'object': 'position'}
                    objects = list(objects_dict.keys())
                    spatial_relations = [f"{obj}_at_{pos}" for obj, pos in objects_dict.items()]
                elif isinstance(objects_dict, (list, set)):
                    # Fallback: normalize to list
                    objects = self._normalize_subgraph_data(objects_dict, 'list')
                
                # Extract spatial anchors from Anchors set
                anchors_set = vlm_direction_data.get('Anchors', set())
                spatial_anchors_detected = self._normalize_subgraph_data(anchors_set, 'list')
                
                # Fallback: try to extract from Explanation field if structured data missing
                explanation = vlm_direction_data.get('Explanation', '')
                if not objects and explanation:
                    # Fallback parsing from explanation text
                    objects, additional_relations, fallback_room, fallback_anchors = self._parse_explanation_text(explanation)
                    if not spatial_relations:
                        spatial_relations = additional_relations
                    if room_type == 'unknown':
                        room_type = fallback_room
                    if not spatial_anchors_detected:
                        spatial_anchors_detected = fallback_anchors
            
            # Limit results to avoid overwhelming downstream processing
            return objects[:5], spatial_relations[:3], room_type, spatial_anchors_detected[:3]
            
        except Exception as e:
            print(f"⚠️ Spatial details extraction failed: {e}")
            return [], [], 'unknown', []
    
    def _parse_explanation_text(self, explanation: str) -> Tuple[List[str], List[str], str, List[str]]:
        """
        Fallback method to parse explanation text for spatial information
        """
        import re
        objects = []
        spatial_relations = []
        room_type = 'unknown'
        spatial_anchors_detected = []
        
        explanation_lower = explanation.lower()
        
        # Extract room type
        room_pattern = r'room:\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        room_match = re.search(room_pattern, explanation_lower)
        if room_match:
            room_type = room_match.group(1)
        
        # Extract objects from legacy format
        objects_pattern = r'objects:\s*[\[{](.*?)[\]}]'
        objects_match = re.search(objects_pattern, explanation_lower)
        if objects_match:
            object_list = objects_match.group(1).split(',')
            for obj_desc in object_list:
                obj_desc = obj_desc.strip()
                if ' at ' in obj_desc:
                    obj_name = obj_desc.split(' at ')[0].strip()
                    obj_position = obj_desc.split(' at ')[1].strip()
                    if obj_name:
                        objects.append(obj_name)
                        spatial_relations.append(f"{obj_name}_at_{obj_position}")
                elif obj_desc:
                    objects.append(obj_desc.strip())
        
        # Extract anchors from legacy format
        anchors_pattern = r'anchors:\s*[\[{](.*?)[\]}]'
        anchors_match = re.search(anchors_pattern, explanation_lower)
        if anchors_match:
            anchor_list = anchors_match.group(1).split(',')
            for anchor in anchor_list:
                anchor_name = anchor.strip()
                if anchor_name and 'detected' not in anchor_name:
                    spatial_anchors_detected.append(anchor_name)
        
        return objects, spatial_relations, room_type, spatial_anchors_detected

    def _build_hierarchical_context(self, hierarchical_overlap: Dict, goal_subgraph: Dict) -> str:
        """
        构建层级化导航上下文，优化为连续字符串格式
        """
        # 安全地提取层级重叠结果，使用.get()方法避免KeyError
        exploration_strategy = hierarchical_overlap.get('exploration_strategy', 'initial_frontier_exploration')
        room_hierarchy_match = hierarchical_overlap.get('room_hierarchy_match', 0.0)
        complete_goal_overlap = hierarchical_overlap.get('complete_goal_overlap', 0.0)
        local_goal_overlap = hierarchical_overlap.get('local_goal_overlap', 0.0)
        exploration_progress = hierarchical_overlap.get('exploration_progress', 0.0)
        stuck_indicator = hierarchical_overlap.get('stuck_indicator', False)
        
        hallway_navigation = goal_subgraph.get('hallway_navigation', {})
        spatial_anchors = self._get_anchor_names_from_subgraph(goal_subgraph)[:3]
        strategic_focus = self._get_strategic_focus_explanation(exploration_strategy, room_hierarchy_match, complete_goal_overlap, exploration_progress, stuck_indicator)
        
        hierarchical_context = (f"Current Navigation Strategy: {exploration_strategy.upper().replace('_', ' ')}. "
            f"Target room confidence: {'High' if room_hierarchy_match > 0.6 else 'Moderate' if room_hierarchy_match > 0.3 else 'Low'} - {'We have identified rooms likely to contain the target' if room_hierarchy_match > 0.6 else 'Some relevant rooms discovered' if room_hierarchy_match > 0.3 else 'Still searching for target room types'}. "
            f"Scene familiarity: {'High' if complete_goal_overlap > 0.5 else 'Moderate' if complete_goal_overlap > 0.3 else 'Low'} - {'Environment matches target expectations well' if complete_goal_overlap > 0.5 else 'Some expected features found' if complete_goal_overlap > 0.3 else 'Environment differs from expected target context'}. "
            f"Current view relevance: {'High' if local_goal_overlap > 0.4 else 'Moderate' if local_goal_overlap > 0.2 else 'Low'} - {'Current location shows strong target indicators' if local_goal_overlap > 0.4 else 'Some target clues in current view' if local_goal_overlap > 0.2 else 'Limited target-relevant information visible'}. "
            f"Overall progress: {'Advanced' if exploration_progress > 0.6 else 'Moderate' if exploration_progress > 0.3 else 'Early stage'} - {'Most target-relevant areas have been explored' if exploration_progress > 0.6 else 'Making steady progress toward target areas' if exploration_progress > 0.3 else 'Beginning systematic exploration of environment'}. "
            f"Navigation status: {'RECOVERY MODE - Agent seems stuck, need new exploration paths' if stuck_indicator else 'NORMAL - Progressing through environment effectively'}. "
            f"Primary spatial anchors: {spatial_anchors}. "
            f"Hallway connections: {hallway_navigation.get('wayfinding_objects', ['standard navigation paths'])}. "
            f"Strategic Focus: {strategic_focus}")
        
        return hierarchical_context


    def _extract_scene_graph_from_vlm(self, vlm_predictions: Dict):
        """
        Extract and maintain hierarchical scene graph from enhanced VLM analysis
        """
        try:
            current_scene_graph = {
                'nodes': set(),
                'spatial_relations': [],
                'room_types': {},
                'spatial_anchors': set(),
                'goal_likelihood': {},
                'accessibility': {},
                'target_visibility': False,
                'hierarchical_context': {}
            }
            
            for direction, data in vlm_predictions.items():
                # Define metadata fields to exclude from direction processing  
                metadata_fields = {'confidence', 'overall_confidence', 'spatial_analysis', 'recommended_directions', 
                                 'exploration_strategy', 'overlap_score'}
                if direction in metadata_fields or not isinstance(data, dict):
                    continue
                    
                # Extract objects (nodes) with enhanced information
                objects = data.get('objects', [])
                current_scene_graph['nodes'].update(objects)
                
                # Extract spatial relations with positioning details
                relations = data.get('spatial_relations', [])
                current_scene_graph['spatial_relations'].extend(relations)
                
                # Track room types by direction
                room_type = data.get('room_type', 'unknown')
                current_scene_graph['room_types'][direction] = room_type
                
                # Extract spatial anchors detected
                spatial_anchors = data.get('spatial_anchors_detected', [])
                current_scene_graph['spatial_anchors'].update(spatial_anchors)
                
                # Track goal likelihood per direction
                goal_likelihood = data.get('goal_likelihood', 'low')
                current_scene_graph['goal_likelihood'][direction] = goal_likelihood
                
                # Track accessibility per direction
                accessibility = data.get('accessibility', 'unclear')
                current_scene_graph['accessibility'][direction] = accessibility
                
                # Check target visibility
                if data.get('target_visible', False):
                    current_scene_graph['target_visibility'] = True
                
                # Store hierarchical context for each direction
                current_scene_graph['hierarchical_context'][direction] = {
                    'score': data.get('Score', 0),
                    'explanation': data.get('Explanation', '')[:100],  # Truncate for memory efficiency
                    'spatial_anchor_count': len(spatial_anchors),
                    'object_count': len(objects)
                }
            
            # Add summary statistics
            current_scene_graph['summary'] = {
                'total_objects': len(current_scene_graph['nodes']),
                'total_spatial_anchors': len(current_scene_graph['spatial_anchors']),
                'room_diversity': len(set(current_scene_graph['room_types'].values())),
                'high_likelihood_directions': [d for d, likelihood in current_scene_graph['goal_likelihood'].items() if likelihood == 'high'],
                'accessible_directions': [d for d, access in current_scene_graph['accessibility'].items() if access == 'accessible']
            }
            
            # Update scene memory with enhanced structure
            self.scene_memory['current_graph'] = current_scene_graph
            self.scene_memory['last_relations'] = current_scene_graph['spatial_relations']
            self.scene_memory['spatial_anchors'] = current_scene_graph['spatial_anchors']
            
            summary = current_scene_graph['summary']
            print(f"🏠 Enhanced scene graph updated: {summary['total_objects']} objects, {summary['total_spatial_anchors']} anchors, {summary['room_diversity']} room types, {len(summary['high_likelihood_directions'])} high-likelihood directions")
            
        except Exception as e:
            print(f"❌ Enhanced scene graph extraction failed: {e}")
            # Fallback to basic structure
            self.scene_memory['current_graph'] = {
                'nodes': set(), 
                'spatial_relations': [], 
                'room_types': {}, 
                'spatial_anchors': set(),
                'target_visibility': False,
                'summary': {'total_objects': 0, 'total_spatial_anchors': 0}
            }

    def _get_exploration_phase_description(self, avg_overlap: float, max_overlap: float) -> dict:
        """
        根据重叠分数确定探索阶段和推理策略
        
        Args:
            avg_overlap: 各方向平均重叠分数 (0.0-1.0)
            max_overlap: 最高重叠分数 (0.0-1.0)
            
        Returns:
            dict containing:
                - phase_name: exploration phase identifier
                - reasoning_strategy: detailed strategy description
                - weight_adjustments: suggested weight modifications
        """
        if max_overlap >= 0.7:
            # High overlap - target verification mode
            return {
                'phase_name': 'TARGET_VERIFICATION',
                'reasoning_strategy': 'High semantic similarity detected - likely very close to target. Focus on precise localization and target confirmation.',
                'weight_adjustments': {
                    'semantic_weight': 0.5,  # Increase semantic influence
                    'spatial_weight': 0.8,   # Reduce spatial weight
                    'llm_weight': 0.2        # Moderate LLM weight
                }
            }
        elif avg_overlap >= 0.4 or max_overlap >= 0.5:
            # Medium overlap - focused navigation
            return {
                'phase_name': 'FOCUSED_SEARCH',
                'reasoning_strategy': 'Moderate semantic alignment found - target area likely nearby. Navigate toward directions with highest semantic relevance.',
                'weight_adjustments': {
                    'semantic_weight': 0.4,  # Balanced semantic influence
                    'spatial_weight': 0.6,   # Balanced spatial weight  
                    'llm_weight': 0.4        # Standard LLM weight
                }
            }
        else:
            # Low overlap - frontier exploration
            return {
                'phase_name': 'FRONTIER_EXPLORATION', 
                'reasoning_strategy': 'Low semantic similarity - broad exploration needed. Prioritize discovering new areas and rooms that might contain target.',
                'weight_adjustments': {
                    'semantic_weight': 0.1,  # Reduce semantic influence
                    'spatial_weight': 0.6,   # Increase spatial exploration
                    'llm_weight': 0.5        # Increase LLM reasoning weight
                }
            }

    def _update_collaborative_memory(self, goal: str, goal_subgraph: Dict, vlm_predictions: Dict, final_predictions: Dict):
        """
        简化版协作记忆更新 - 保留核心跟踪功能
        """
        try:
            # Extract current agent position if available
            agent_position = None
            if hasattr(self, 'sim') and hasattr(self.sim, 'get_agent_state'):
                try:
                    agent_state = self.sim.get_agent_state()
                    agent_position = [agent_state.position[0], agent_state.position[2]]  # x, z coordinates
                except:
                    agent_position = None
            
            # Update enhanced scene memory
            step_key = f"step_{len(self.scene_memory)}"
            self.scene_memory[step_key] = {
                'vlm_data': vlm_predictions,
                'final_predictions': final_predictions,
                'goal_subgraph': goal_subgraph,
                'agent_position': agent_position,
                'timestamp': time.time(),
                'step_number': len(self.scene_memory),
                'goal': goal,
                'exploration_strategy': final_predictions.get('exploration_strategy', 'unknown'),
                'stuck_indicator': final_predictions.get('stuck_indicator', False)
            }
            
            # Update subtask history with enhanced context
            if hasattr(self, 'current_subtask') and self.current_subtask:
                self.subtask_history.append({
                    'subtask': self.current_subtask,
                    'step': len(self.scene_memory),
                    'confidence': final_predictions.get('overall_confidence', 0.5),
                    'overlap_score': final_predictions.get('overlap_score', 0.0),
                    'exploration_strategy': final_predictions.get('exploration_strategy', 'unknown')
                })
            
            # Enhanced memory management: keep important information
            if len(self.scene_memory) > 15:  # Extended memory window for better stuck detection
                # Keep recent steps and important transitions
                keys_to_keep = list(self.scene_memory.keys())[-10:]  # Last 10 steps
                
                # Keep steps with high importance (room transitions, high confidence, etc.)
                for key in list(self.scene_memory.keys())[:-10]:
                    step_data = self.scene_memory[key]
                    if (step_data.get('final_predictions', {}).get('overall_confidence', 0) > 0.8 or
                        step_data.get('exploration_strategy') == 'goal_focused_navigation' or
                        step_data.get('vlm_data', {}).get('room_type', '') != 'unknown'):
                        keys_to_keep.append(key)
                
                # Remove non-important old steps
                for key in list(self.scene_memory.keys()):
                    if key not in keys_to_keep:
                        del self.scene_memory[key]
            
            print(f"💾 Enhanced collaborative memory updated: {len(self.scene_memory)} steps, {len(self.subtask_history)} subtasks")
            
        except Exception as e:
            print(f"❌ Enhanced memory update failed: {e}")
            traceback.print_exc()
    
    def _eval_response(self, response: str):
        """Enhanced VLM response parsing with robust nested brace handling for LLM/VLM JSON outputs"""
        import re
        import json
        
        if not response or not isinstance(response, str):
            logging.error('Empty or invalid response')
            return {}
        
        # Clean response - normalize quotes to single quotes, handle escaping
        result = response.strip()
        # Fix common quote issues in VLM responses
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", result)
        # Normalize double quotes to single quotes for consistency
        result = re.sub(r'"([^"]*)":', r"'\1':", result)
        result = re.sub(r':\s*"([^"]*)"', r": '\1'", result)
        
        print(f"🔧 Cleaned response for parsing: {result[:200]}...")
        
        # Strategy 1: Try multiple nested brace extraction patterns
        brace_patterns = [
            # Pattern 1: Most outer braces {{...}}
            (lambda s: s[s.index('{') + 1:s.rindex('}')], "outer_double_braces"),
            # Pattern 2: Single braces {...}
            (lambda s: s[s.rindex('{'):s.rindex('}') + 1], "single_braces_last"),
            # Pattern 3: First complete brace set {{...}, {...}}
            (lambda s: s[s.index('{'):s.rindex('}')+1], "complete_braces"),
            # Pattern 4: First inner content after outer brace removal
            (lambda s: s[s.index('{', s.index('{') + 1):s.rindex('}', 0, s.rindex('}'))], "inner_braces"),
            # Pattern 5: Extract between triple braces {{{...}}}
            (lambda s: s[s.index('{') + 2:s.rindex('}') - 1] if s.count('{') >= 3 else s[s.index('{'):s.rindex('}')+1], "triple_braces"),
            # Pattern 6: Find largest valid JSON section
            (lambda s: self._extract_largest_json_section(s), "largest_json_section")
        ]
        
        for extract_func, pattern_name in brace_patterns:
            try:
                extracted = extract_func(result)
                if not extracted:
                    continue
                    
                print(f"🎯 Trying {pattern_name}: {extracted[:100]}...")
                
                # Try ast.literal_eval first (handles Python dict format)
                try:
                    eval_resp = ast.literal_eval(extracted)
                    if isinstance(eval_resp, dict):
                        print(f"✅ Successfully parsed with {pattern_name} using ast.literal_eval")
                        
                        # Check what type of data we got
                        expected_directions = ['30', '90', '150', '210', '270', '330']
                        direction_keys = [k for k in eval_resp.keys() if k in expected_directions]
                        
                        # Check for goal subgraph indicators
                        subgraph_indicators = ['target_room_hierarchy', 'spatial_anchor_objects', 'hallway_navigation', 'exploration_strategy']
                        subgraph_keys = [k for k in eval_resp.keys() if k in subgraph_indicators]
                        
                        if len(direction_keys) >= 3:  # Directional VLM data
                            print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                            return eval_resp
                        elif len(subgraph_keys) >= 2:  # Goal subgraph data
                            print(f"✅ Confirmed goal subgraph data with keys: {subgraph_keys}")
                            return eval_resp
                        elif len(eval_resp) > 0:  # Any valid dictionary
                            print(f"✅ Valid dictionary data with {len(eval_resp)} keys: {list(eval_resp.keys())[:5]}")
                            return eval_resp
                        else:
                            print(f"⚠️ Empty dictionary, continuing search...")
                except (ValueError, SyntaxError) as e:
                    print(f"   ast.literal_eval failed for {pattern_name}: {e}")
                
                # Try JSON parsing as backup (after converting single quotes to double quotes)
                try:
                    json_str = extracted.replace("'", '"')
                    eval_resp = json.loads(json_str)
                    if isinstance(eval_resp, dict):
                        print(f"✅ Successfully parsed with {pattern_name} using json.loads")
                        
                        # Check what type of data we got
                        expected_directions = ['30', '90', '150', '210', '270', '330']
                        direction_keys = [k for k in eval_resp.keys() if k in expected_directions]
                        
                        # Check for goal subgraph indicators
                        subgraph_indicators = ['target_room_hierarchy', 'spatial_anchor_objects', 'hallway_navigation', 'exploration_strategy']
                        subgraph_keys = [k for k in eval_resp.keys() if k in subgraph_indicators]
                        
                        if len(direction_keys) >= 3:  # Directional VLM data
                            print(f"✅ Confirmed directional data with {len(direction_keys)} directions")
                            return eval_resp
                        elif len(subgraph_keys) >= 2:  # Goal subgraph data
                            print(f"✅ Confirmed goal subgraph data with keys: {subgraph_keys}")
                            return eval_resp
                        elif len(eval_resp) > 0:  # Any valid dictionary
                            print(f"✅ Valid dictionary data with {len(eval_resp)} keys: {list(eval_resp.keys())[:5]}")
                            return eval_resp
                        else:
                            print(f"⚠️ Empty dictionary, continuing search...")
                except json.JSONDecodeError as e:
                    print(f"   json.loads failed for {pattern_name}: {e}")
                    
            except (ValueError, IndexError) as e:
                print(f"   Pattern {pattern_name} extraction failed: {e}")
                continue
        
        # Strategy 2: Regex-based key-value extraction for malformed JSON (only as last resort)
        print("🔧 Attempting regex-based key-value extraction as last resort...")
        try:
            extracted_dict = self._regex_extract_dict(result)
            if extracted_dict:
                # Check if we got valid directional data
                expected_directions = ['30', '90', '150', '210', '270', '330']
                direction_keys = [k for k in extracted_dict.keys() if k in expected_directions]
                if len(direction_keys) >= 3:
                    print(f"✅ Successfully extracted via regex: {len(extracted_dict)} keys with {len(direction_keys)} directions")
                    return extracted_dict
                else:
                    print(f"⚠️ Regex extraction didn't yield directional data")
        except Exception as e:
            print(f"   Regex extraction failed: {e}")
        
        # Strategy 3: Line-by-line parsing for structured responses
        print("🔧 Attempting line-by-line structured parsing...")
        try:
            line_parsed = self._parse_structured_lines(result)
            if line_parsed:
                print(f"✅ Successfully parsed via line parsing: {len(line_parsed)} keys")
                return line_parsed
        except Exception as e:
            print(f"   Line parsing failed: {e}")
        
        logging.error(f'All parsing strategies failed for response: {response[:200]}...')
        return {}

    def _extract_largest_json_section(self, text: str) -> str:
        """Extract the largest valid JSON-like section from text, preferring directional data"""
        max_content = ""
        max_length = 0
        best_directional_content = ""
        
        # Find all potential JSON sections
        import re
        
        # Expected direction keys
        expected_directions = ['30', '90', '150', '210', '270', '330']
        
        # Strategy 1: Look for the outermost JSON block that contains all directions
        # Pattern to match complete JSON with all directions
        full_json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_candidates = re.findall(full_json_pattern, text, re.DOTALL)
        
        for candidate in json_candidates:
            direction_count = sum(1 for direction in expected_directions 
                                if f"'{direction}'" in candidate or f'"{direction}"' in candidate)
            
            if direction_count >= 5:  # Look for nearly complete directional data
                if len(candidate) > len(best_directional_content):
                    best_directional_content = candidate
                    print(f"🎯 Found comprehensive directional JSON with {direction_count} directions, length: {len(candidate)}")
        
        # Strategy 2: Fallback to brace matching for nested structures
        if not best_directional_content:
            brace_positions = [(m.start(), '{') for m in re.finditer(r'\{', text)]
            brace_positions.extend([(m.start(), '}') for m in re.finditer(r'\}', text)])
            brace_positions.sort()
            
            stack = []
            for pos, brace in brace_positions:
                if brace == '{':
                    stack.append(pos)
                elif brace == '}' and stack:
                    start_pos = stack.pop()
                    section = text[start_pos:pos + 1]
                    
                    # Check if this section contains directional data
                    direction_count = sum(1 for direction in expected_directions 
                                        if f"'{direction}'" in section or f'"{direction}"' in section)
                    
                    if direction_count >= 3:  # Has multiple directions - prefer this
                        if len(section) > len(best_directional_content):
                            best_directional_content = section
                            print(f"🎯 Found directional JSON section with {direction_count} directions, length: {len(section)}")
                    
                    # Also track the largest section as fallback
                    if len(section) > max_length:
                        max_length = len(section)
                        max_content = section
        
        # Strategy 3: If still no good directional content, try to reconstruct from parts
        if not best_directional_content:
            print(f"🔧 Attempting to reconstruct complete directional JSON from parts...")
            
            # Extract individual direction blocks
            direction_blocks = {}
            for direction in expected_directions:
                # Look for patterns like '30': { ... }
                direction_pattern = rf"['\"]?{direction}['\"]?\s*:\s*\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}"
                match = re.search(direction_pattern, text, re.DOTALL)
                if match:
                    direction_blocks[direction] = match.group(0)
                    print(f"✅ Found block for direction {direction}")
            
            # If we found multiple direction blocks, combine them
            if len(direction_blocks) >= 3:
                combined_json = "{" + ", ".join(direction_blocks.values()) + "}"
                print(f"🔨 Reconstructed JSON with {len(direction_blocks)} directions")
                best_directional_content = combined_json
        
        # Return directional content if found, otherwise return largest content
        if best_directional_content:
            print(f"✅ Using directional JSON section (length: {len(best_directional_content)})")
            return best_directional_content
        else:
            print(f"⚠️ No directional JSON found, using largest section (length: {len(max_content)})")
            return max_content

    def _regex_extract_dict(self, text: str) -> Dict:
        """Extract dictionary from text using regex patterns with proper nested structure handling"""
        import re
        import ast
        result_dict = {}
        
        # Enhanced pattern for better nested structure matching
        kv_pattern = r"'([^']+)':\s*(?:'([^']*)'|(\d+\.?\d*)|(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})|(\[[^\]]*\]))"
        matches = re.findall(kv_pattern, text, re.DOTALL)
        
        for match in matches:
            key = match[0]
            if match[1]:  # string value
                result_dict[key] = match[1]
            elif match[2]:  # numeric value
                try:
                    result_dict[key] = float(match[2]) if '.' in match[2] else int(match[2])
                except ValueError:
                    result_dict[key] = match[2]
            elif match[3]:  # dict value - parse it properly
                dict_str = match[3]
                try:
                    # Clean the dict string before parsing to fix common VLM errors
                    dict_str_cleaned = self._clean_dict_string(dict_str)
                    parsed_dict = ast.literal_eval(dict_str_cleaned)
                    result_dict[key] = parsed_dict
                    print(f"   ✅ Successfully parsed nested dict for key '{key}': {type(parsed_dict)}")
                except (ValueError, SyntaxError) as e:
                    print(f"   ⚠️ Failed to parse nested dict for key '{key}': {e}")
                    # Try alternative parsing approaches
                    try:
                        # Remove problematic patterns and try again
                        dict_str_fixed = self._fix_malformed_dict_string(dict_str)
                        if dict_str_fixed:
                            parsed_dict = ast.literal_eval(dict_str_fixed)
                            result_dict[key] = parsed_dict
                            print(f"   ✅ Successfully parsed after fixing for key '{key}': {type(parsed_dict)}")
                        else:
                            # Store as string if cannot parse
                            result_dict[key] = dict_str
                    except Exception as e2:
                        print(f"   ❌ Final attempt failed for key '{key}': {e2}")
                        result_dict[key] = dict_str
            elif match[4]:  # list value - parse it properly
                list_str = match[4]
                try:
                    parsed_list = ast.literal_eval(list_str)
                    result_dict[key] = parsed_list
                    print(f"   ✅ Successfully parsed list for key '{key}': {type(parsed_list)}")
                except (ValueError, SyntaxError) as e:
                    print(f"   ⚠️ Failed to parse list for key '{key}': {e}")
                    # Fallback: store as string
                    result_dict[key] = list_str
        
        # Additional validation: check if we got direction keys with dict values
        expected_directions = ['30', '90', '150', '210', '270', '330']
        direction_keys = [k for k in result_dict.keys() if k in expected_directions]
        
        if direction_keys:
            print(f"🎯 Found {len(direction_keys)} direction keys in regex extraction")
            for direction in direction_keys:
                direction_data = result_dict[direction]
                if isinstance(direction_data, str):
                    print(f"⚠️ Direction {direction} is still a string, attempting enhanced parsing...")
                    # Use the new enhanced parsing method
                    try:
                        parsed_data = self._clean_and_parse_direction_string(direction_data)
                        if isinstance(parsed_data, dict):
                            result_dict[direction] = parsed_data
                            print(f"   ✅ Successfully converted direction {direction} to dict")
                        else:
                            print(f"   ⚠️ Enhanced parsing for {direction} returned non-dict: {type(parsed_data)}")
                    except Exception as e:
                        print(f"   ❌ Enhanced parsing failed for direction {direction}: {e}")
                        # Fallback to original method
                        try:
                            parsed_data = ast.literal_eval(direction_data)
                            if isinstance(parsed_data, dict):
                                result_dict[direction] = parsed_data
                                print(f"   ✅ Fallback parsing worked for direction {direction}")
                        except Exception as e2:
                            print(f"   ❌ All parsing methods failed for direction {direction}: {e2}")
                elif isinstance(direction_data, dict):
                    print(f"   ✅ Direction {direction} is already a dict with keys: {list(direction_data.keys())}")
        
        return result_dict

    def _parse_structured_lines(self, text: str) -> Dict:
        """Parse structured multi-line responses into dictionary"""
        import re
        lines = text.split('\n')
        result_dict = {}
        current_key = None
        current_value = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for key: value pattern
            key_match = re.match(r"'([^']+)':\s*(.*)", line)
            if key_match:
                # Save previous key-value if exists
                if current_key and current_value:
                    result_dict[current_key] = current_value.strip().strip("'\"")
                
                current_key = key_match.group(1)
                current_value = key_match.group(2).strip().strip("'\"")
            elif current_key and line:
                # Continue building current value
                current_value += " " + line.strip().strip("'\"")
        
        # Save final key-value
        if current_key and current_value:
            result_dict[current_key] = current_value.strip().strip("'\"")
        
        return result_dict

    def _clean_dict_string(self, dict_str: str) -> str:
        """Clean dictionary string to fix common VLM formatting issues"""
        import re
        
        # Fix duplicate keys like 'Score': 6, 'Score': 'Score': 6
        dict_str = re.sub(r"'Score':\s*\d+,\s*'Score':\s*'Score':\s*\d+", 
                         lambda m: "'Score': " + re.search(r'\d+', m.group()).group(), dict_str)
        
        # Fix other duplicate key patterns  
        dict_str = re.sub(r"'(\w+)':\s*([^,}]+),\s*'\1':\s*'\1':\s*([^,}]+)", r"'\1': \2", dict_str)
        
        # Fix trailing commas before closing braces
        dict_str = re.sub(r',\s*}', '}', dict_str)
        
        # Fix multiple consecutive commas
        dict_str = re.sub(r',\s*,+', ',', dict_str)
        
        return dict_str.strip()

    def _fix_malformed_dict_string(self, dict_str: str) -> str:
        """Try to fix malformed dictionary strings with more aggressive cleaning"""
        import re
        
        try:
            # More aggressive cleaning
            dict_str = dict_str.strip()
            
            # Remove problematic duplicate key patterns entirely
            dict_str = re.sub(r"'Score':\s*'Score':\s*\d+", "'Score': 0", dict_str)
            dict_str = re.sub(r"'(\w+)':\s*'\1':\s*([^,}]+)", r"'\1': \2", dict_str)
            
            # Fix missing quotes around keys (be more careful with context)
            dict_str = re.sub(r'(\w+):\s*\{', r"'\1': {", dict_str)  # Fix keys before nested dicts
            dict_str = re.sub(r'(\w+):\s*\'([^\']*)\'\s*([,}])', r"'\1': '\2'\3", dict_str)  # quoted values
            dict_str = re.sub(r'(\w+):\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r"'\1': '\2'\3", dict_str)  # unquoted values
            
            # Fix nested object issues - ensure quotes around object keys/values
            dict_str = re.sub(r'\{\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)', r"{ '\1': '\2'", dict_str)
            dict_str = re.sub(r',\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)', r", '\1': '\2'", dict_str)
            
            # Fix trailing issues
            dict_str = re.sub(r',\s*}', '}', dict_str)
            dict_str = re.sub(r',\s*,+', ',', dict_str)
            
            # Basic validation - must start with { and end with }
            if not dict_str.startswith('{') or not dict_str.endswith('}'):
                return None
                
            return dict_str.strip()
            
        except Exception as e:
            print(f"   Error in fix_malformed_dict_string: {e}")
            return None

    def _clean_and_parse_direction_string(self, direction_str: str) -> dict:
        """Enhanced cleaning and parsing for direction data strings"""
        import re
        import ast
        
        try:
            # Apply all cleaning strategies
            cleaned = self._clean_dict_string(direction_str)
            
            # Try parsing the cleaned version
            try:
                return ast.literal_eval(cleaned)
            except:
                # If that fails, try more aggressive fixing
                fixed = self._fix_malformed_dict_string(direction_str)
                if fixed:
                    return ast.literal_eval(fixed)
                else:
                    # Final attempt: extract key-value pairs manually
                    return self._manual_dict_extraction(direction_str)
                    
        except Exception as e:
            print(f"   Error in clean_and_parse_direction_string: {e}")
            return direction_str  # Return original if all fails

    def _manual_dict_extraction(self, text: str) -> dict:
        """Manually extract dictionary from problematic text"""
        import re
        
        result = {}
        
        # Extract Score (handle various formats)
        score_match = re.search(r"(?:'Score'|Score):\s*(\d+)", text)
        if score_match:
            result['Score'] = int(score_match.group(1))
        
        # Extract Objects (handle various formats)
        objects_match = re.search(r"(?:'Objects'|Objects):\s*\{([^}]+)\}", text)
        if objects_match:
            objects_text = objects_match.group(1)
            objects = {}
            
            # Try different patterns for object entries
            for obj_match in re.finditer(r"'([^']+)':\s*'([^']*)'", objects_text):
                objects[obj_match.group(1)] = obj_match.group(2)
            
            # Handle unquoted keys/values
            if not objects:  # If no quoted matches found
                for obj_match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*):\s*([a-zA-Z_][a-zA-Z0-9_]*)", objects_text):
                    objects[obj_match.group(1)] = obj_match.group(2)
            
            result['Objects'] = objects
        
        # Extract Room (handle various formats)
        room_match = re.search(r"(?:'Room'|Room):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
        if room_match:
            result['Room'] = room_match.group(1) if room_match.group(1) else room_match.group(2).strip()
        
        # Extract Path (handle various formats)
        path_match = re.search(r"(?:'Path'|Path):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
        if path_match:
            result['Path'] = path_match.group(1) if path_match.group(1) else path_match.group(2).strip()
            
        # Extract Explanation (handle various formats)
        explanation_match = re.search(r"(?:'Explanation'|Explanation):\s*(?:'([^']*)'|([a-zA-Z\s]+))(?:,|$|\})", text)
        if explanation_match:
            result['Explanation'] = explanation_match.group(1) if explanation_match.group(1) else explanation_match.group(2).strip()
        
        return result if result else text


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
            return prompt
                
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The observed image>: The image taken from its current location. "
                f"(2){reason}. This explains why you should go in this direction. "
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
                
        if prompt_type == 'action':
            # Enhanced action prompt with VLM object position information
            scene_objects_context = ""
            
            # Check if we have current direction scene data
            if hasattr(self, '_current_direction_scene_data') and self._current_direction_scene_data:
                direction_data = self._current_direction_scene_data
                spatial_objects = direction_data.get('spatial_objects', [])
                vlm_predictions = direction_data.get('vlm_predictions', {})
                goal_rotate = direction_data.get('goal_rotate', 0)
                
                # Build scene objects context
                if spatial_objects or vlm_predictions:
                    scene_objects_context = f"\n\n🎯 SCENE ANALYSIS for current direction ({goal_rotate * 30}°):\n"
                    
                    # Add VLM detected objects with positions
                    vlm_objects = vlm_predictions.get('Objects', vlm_predictions.get('objects', {}))
                    if isinstance(vlm_objects, dict) and vlm_objects:
                        scene_objects_context += "📍 Detected Objects and Their Locations:\n"
                        for obj_name, obj_data in vlm_objects.items():
                            if isinstance(obj_data, dict):
                                position = obj_data.get('position', 'unknown')
                                likelihood = obj_data.get('likelihood', obj_data.get('score', 'unknown'))
                                scene_objects_context += f"  - {obj_name}: position={position}, confidence={likelihood}\n"
                    
                    # Add spatial objects information
                    if spatial_objects:
                        scene_objects_context += "\n🗺️ Spatial Object Mapping:\n"
                        for i, obj in enumerate(spatial_objects[:5]):  # Show first 5 objects
                            if isinstance(obj, dict):
                                obj_name = obj.get('name', f'object_{i}')
                                obj_pos = obj.get('position', 'unknown')
                                obj_type = obj.get('type', 'unknown')
                                scene_objects_context += f"  - {obj_name} ({obj_type}): spatial_pos={obj_pos}\n"
                    
                    # Add goal-related analysis if available
                    current_graph = direction_data.get('current_graph', {})
                    if current_graph:
                        room_context = current_graph.get('room_nodes', [])
                        if room_context:
                            room_types = [room.get('type', 'unknown') for room in room_context if isinstance(room, dict)]
                            scene_objects_context += f"\n🏠 Room Context: {', '.join(room_types)}\n"
                    
                    scene_objects_context += f"\n💡 Use this spatial information to make more informed navigation decisions!\n"
            
            # Original action prompt with enhanced context
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                f"{scene_objects_context}"
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    f"{scene_objects_context}"
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt
                
    def update_curiosity_value(self, explorable_value, reason, vlm_predictions=None, goal_subgraph=None):
        """
        增强版好奇心值更新函数：整合空间-语义推理与UniGoal图匹配
        
        Args:
            explorable_value: 父类方法所需的可探索值字典
            reason: 父类方法所需的推理字典  
            vlm_predictions: VLM预测数据（直接传入，避免使用缓存）
            goal_subgraph: LLM构建的目标子图（直接传入，避免使用缓存）
            
        5阶段处理流程：
        1. 基础值图更新（继承原始功能）
        2. 对象空间定位与映射（结合voxel_map准确性）
        3. 场景图构建与匹配
        4. UniGoal风格重叠计算（使用LLM目标子图）
        5. LLM空间推理与智能评分
        """
        # 如果没有直接传入的数据，尝试使用当前存储的数据
        if vlm_predictions is None and hasattr(self, '_current_vlm_predictions'):
            vlm_predictions = self._current_vlm_predictions
        if goal_subgraph is None and hasattr(self, '_current_goal_subgraph'):
            goal_subgraph = self._current_goal_subgraph
            
        # 如果仍然没有增强数据，使用基础功能
        if vlm_predictions is None or goal_subgraph is None:
            print("⚠️ 未发现VLM预测数据或目标子图，使用基础好奇心更新")
            return super().update_curiosity_value(explorable_value, reason)
        
        print(f"\n🧠 增强好奇心更新 (空间-语义整合)")
        
        # === 先进行语义信息回填 ===
        self._backfill_semantic_to_spatial_positions(vlm_predictions)
        
        # 对每个方向进行增强处理 - 先收集所有数据
        direction_data = {}
        for direction_str, base_score in explorable_value.items():
            direction = int(direction_str)
            
            print(f"\n   方向 {direction}°:")
            print(f"   基础评分: {base_score}")
            
            # === 阶段2: 对象空间定位与映射（结合voxel_map准确性）===
            spatial_objects = self._map_objects_to_spatial_coordinates_with_voxel(
                direction, vlm_predictions, None  # action_result不需要，因为我们已经有了base_score
            )
            
            # === 阶段3: 场景图构建 ===
            current_graph = self._build_current_scene_graph(spatial_objects, direction)
            
            # === 阶段4: UniGoal风格图重叠计算（直接使用LLM目标子图）===
            # 提取当前方向的VLM对象数据
            direction_vlm_data = vlm_predictions.get(direction_str, {}) if isinstance(vlm_predictions, dict) else {}
            direction_objects = direction_vlm_data.get('Objects', direction_vlm_data.get('objects', {})) if isinstance(direction_vlm_data, dict) else {}
            
            # 为重叠计算创建方向特定的vlm_objects字典
            direction_specific_vlm_objects = {direction_str: direction_vlm_data} if direction_vlm_data else {}
            
            overlap_result = self._calculate_graph_overlap_score_with_subgraph(
                direction_specific_vlm_objects, goal_subgraph, direction, reason=reason
            )
            overlap_score = overlap_result.get('overlap_score', 0.0)
            
            # 存储每个方向的数据
            direction_data[direction_str] = {
                'base_score': base_score,
                'spatial_objects': spatial_objects,
                'current_graph': current_graph,
                'overlap_score': overlap_score,
                'vlm_predictions': direction_vlm_data,  # 添加VLM预测数据
                'goal_rotate': direction  # 添加方向信息
            }
            
            print(f"   空间对象: {len(spatial_objects)}")
            print(f"   图重叠: {overlap_score:.3f}")
        
        # === Overlap-Based Reasoning Strategy Selection ===
        # 计算各方向的平均重叠分数以确定探索阶段
        overlap_scores = [data['overlap_score'] for data in direction_data.values()]
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        max_overlap = max(overlap_scores) if overlap_scores else 0.0
        
        # 根据重叠分数确定探索策略
        exploration_phase = self._get_exploration_phase_description(avg_overlap, max_overlap)
        
        print(f"\n📊 重叠分析: 平均={avg_overlap:.3f}, 最大={max_overlap:.3f}")
        print(f"🎯 探索阶段: {exploration_phase['phase_name']}")
        print(f"💡 推理策略: {exploration_phase['reasoning_strategy']}")
        
        # === 阶段5: LLM综合空间推理与方向推荐 ===
        current_goal = getattr(self, 'last_cached_goal', 'unknown_target')
        llm_result = self._perform_llm_spatial_reasoning(
            direction_data, current_goal, vlm_predictions, exploration_phase
        )
        
        recommended_direction = llm_result.get('recommended_direction', 90)
        llm_confidence = llm_result.get('confidence_score', 0.5)
        reasoning = llm_result.get('reasoning_summary', 'No reasoning provided')
        
        print(f"\n🧠 LLM综合推理结果:")
        print(f"   推荐方向: {recommended_direction}°")
        print(f"   置信度: {llm_confidence:.3f}")
        print(f"   推理摘要: {reasoning}")
        
        # 根据LLM推荐调整各方向得分
        enhanced_scores = {}
        
        # 获取exploration phase的权重调整
        weight_adjustments = exploration_phase.get('weight_adjustments', {})
        spatial_weight = weight_adjustments.get('spatial_weight', 0.6)
        semantic_weight = weight_adjustments.get('semantic_weight', 0.2)
        llm_weight = weight_adjustments.get('llm_weight', 0.3)
        
        print(f"🎯 应用 {exploration_phase['phase_name']} 权重策略:")
        print(f"   空间权重: {spatial_weight}, 语义权重: {semantic_weight}, LLM权重: {llm_weight}")
        
        for direction_str, data in direction_data.items():
            direction = int(direction_str)
            base_score = data['base_score']
            overlap_score = data['overlap_score']
            
            # LLM推荐方向的加成
            if direction == recommended_direction:
                llm_boost = llm_confidence # 推荐方向获得confidence加成
            else:
                llm_boost = max(0.2, llm_confidence * 0.5)  # 其他方向获得部分加成
            
            if base_score >= 10:
                enhanced_score = 10  #目标定向
            else:
                enhanced_score = (
                    base_score * spatial_weight +
                    overlap_score * 10 * semantic_weight +  # 转换到0-10范围
                    llm_boost * 10 * llm_weight     # LLM推理加成转换到0-10范围
                )
                # 确保分数在合理范围内，并保持数值稳定性
                enhanced_score = max(0.0, min(10.0, enhanced_score))
                # 四舍五入到两位小数，减少浮点精度误差
                enhanced_score = round(enhanced_score, 2)
            
            enhanced_scores[direction_str] = enhanced_score
            
            print(f"   方向{direction}°: 基础{base_score:.2f}×{spatial_weight} + 语义{overlap_score:.3f}×{semantic_weight} + LLM{llm_boost:.3f}×{llm_weight} = {enhanced_score:.3f}")
        
        print(f"\n✅ 空间-语义-LLM整合完成，推荐方向: {recommended_direction}°")
        final_result = super().update_curiosity_value(enhanced_scores, reason)
        
        # 存储direction_data供make_plan使用
        self._direction_scene_data = direction_data
        
        # 清理临时数据
        if hasattr(self, '_current_vlm_predictions'):
            delattr(self, '_current_vlm_predictions')
        if hasattr(self, '_current_goal_subgraph'):
            delattr(self, '_current_goal_subgraph')
            
        return final_result
        
    def update_voxel(self, r: float, theta: float, agent_state, temp_map: np.ndarray, effective_dist: float=3):
        """
        重写父类的update_voxel方法，增加语义信息存储
        结合父类的空间映射功能与语义物体信息
        
        注意：只在有VLM语义数据时才进行语义映射，避免无意义的更新
        """
        # 调用父类方法进行基础体素更新
        super().update_voxel(r, theta, agent_state, temp_map, effective_dist)
        
        # 只在有VLM预测数据时才添加语义信息
        if hasattr(self, '_current_vlm_predictions') and self._current_vlm_predictions:
            self._update_semantic_voxel_mapping(r, theta, agent_state, temp_map)
        # 移除_record_spatial_position调用 - 由_map_objects_to_spatial_coordinates_with_voxel统一处理
        
    def _update_semantic_voxel_mapping(self, r: float, theta: float, agent_state, temp_map: np.ndarray):
        """
        更新语义体素映射：将VLM检测到的物体信息映射到体素空间
        """
        try:
            # 计算空间坐标
            agent_coords = self._global_to_grid(agent_state.position)
            clipped = min(r, 3.0)  # 限制在有效范围内
            
            # 计算局部坐标和全局坐标
            local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
            global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
            point = self._global_to_grid(global_coords)
            
            # 获取当前方向的角度字符串
            direction_deg = int(np.degrees(theta))
            direction_key = str(((direction_deg + 15) // 30) * 30)  # 转换为30度间隔
            
            # 检查是否有该方向的VLM预测数据
            if hasattr(self, '_current_vlm_predictions') and self._current_vlm_predictions:
                direction_data = self._current_vlm_predictions.get(direction_key, {})
                
                if isinstance(direction_data, dict):
                    objects = direction_data.get('Objects', direction_data.get('objects', []))
                    room_type = direction_data.get('Room', direction_data.get('room_type', 'unknown'))
                    confidence = direction_data.get('Score', direction_data.get('score', 0.5))
                    
                    # 更新语义体素映射
                    voxel_key = f"{point[0]}_{point[1]}"
                    
                    if voxel_key not in self.semantic_voxel_map:
                        self.semantic_voxel_map[voxel_key] = {
                            'position': point,
                            'global_coords': global_coords,
                            'objects': [],
                            'room_type': room_type,
                            'confidence': confidence,
                            'observations': 1,
                            'last_updated': self.step_ndx if hasattr(self, 'step_ndx') else 0
                        }
                    else:
                        # 更新现有条目
                        existing = self.semantic_voxel_map[voxel_key]
                        existing['observations'] += 1
                        existing['confidence'] = (existing['confidence'] + confidence) / 2
                        existing['last_updated'] = self.step_ndx if hasattr(self, 'step_ndx') else 0
                        
                        # 如果房间类型更确定，更新房间类型
                        if confidence > existing.get('room_confidence', 0):
                            existing['room_type'] = room_type
                            existing['room_confidence'] = confidence
                    
                    # 添加物体信息
                    if isinstance(objects, list):
                        for obj in objects:
                            obj_name = obj if isinstance(obj, str) else obj.get('name', str(obj))
                            if obj_name not in self.semantic_voxel_map[voxel_key]['objects']:
                                self.semantic_voxel_map[voxel_key]['objects'].append(obj_name)
                                
                                # 更新全局场景图
                                self._update_global_scene_graph(obj_name, point, room_type, voxel_key)
            
            print(f"   📍 语义体素更新: 方向{direction_key}°, 位置({point[0]}, {point[1]})")
                                
        except Exception as e:
            print(f"   ⚠️ 语义体素映射更新失败: {e}")
            
    def _backfill_semantic_to_spatial_positions(self, vlm_predictions: Dict):
        """
        简化版语义信息回填函数
        由于_map_objects_to_spatial_coordinates_with_voxel已经处理了空间-语义映射，
        这个函数现在主要用于验证和清理
        """
        if not isinstance(vlm_predictions, dict):
            print(f"   ⚠️ VLM预测数据格式无效 (type: {type(vlm_predictions)})")
            return
            
        print(f"   💭 VLM语义数据已可用，包含{len(vlm_predictions)}个方向的预测")
        
        # 清理过期的数据结构（如果存在）
        if hasattr(self, '_pending_spatial_positions'):
            delattr(self, '_pending_spatial_positions')
            
    def _update_global_scene_graph(self, obj_name: str, position: tuple, room_type: str, voxel_key: str):
        """
        更新全局场景图，添加新的节点和边
        """
        obj_id = f"{obj_name}_{voxel_key}"
        
        # 检查节点是否已存在
        existing_node = next((node for node in self.scene_graph_global['nodes'] 
                            if node['id'] == obj_id), None)
        
        if not existing_node:
            # 添加新节点
            new_node = {
                'id': obj_id,
                'name': obj_name,
                'type': 'object',
                'room_type': room_type,
                'position': position,
                'voxel_key': voxel_key,
                'confidence': 0.8,
                'observations': 1
            }
            self.scene_graph_global['nodes'].append(new_node)
            
            # 更新物体位置追踪
            if obj_name not in self.object_position_tracker:
                self.object_position_tracker[obj_name] = []
            self.object_position_tracker[obj_name].append({
                'position': position,
                'voxel_key': voxel_key,
                'timestamp': self.step_ndx if hasattr(self, 'step_ndx') else 0,
                'room_type': room_type
            })
            
            # 创建空间关系边
            self._create_spatial_relationships(obj_id, obj_name, position, room_type)
        else:
            # 更新现有节点
            existing_node['observations'] += 1
            existing_node['confidence'] = min(1.0, existing_node['confidence'] + 0.1)
            
    def _create_spatial_relationships(self, new_obj_id: str, new_obj_name: str, new_position: tuple, room_type: str):
        """
        基于空间邻近性和语义关系创建场景图的边
        """
        PROXIMITY_THRESHOLD = 100  # 像素距离阈值
        
        for existing_node in self.scene_graph_global['nodes']:
            if existing_node['id'] == new_obj_id:
                continue
                
            existing_pos = existing_node['position']
            distance = np.sqrt((new_position[0] - existing_pos[0])**2 + 
                             (new_position[1] - existing_pos[1])**2)
            
            if distance < PROXIMITY_THRESHOLD:
                # 创建空间邻近边
                spatial_edge = {
                    'source': new_obj_id,
                    'target': existing_node['id'],
                    'type': 'spatial_near',
                    'distance': distance,
                    'confidence': max(0.1, 1.0 - distance / PROXIMITY_THRESHOLD)
                }
                
                # 检查边是否已存在
                edge_exists = any(
                    edge['source'] == spatial_edge['source'] and 
                    edge['target'] == spatial_edge['target'] and
                    edge['type'] == spatial_edge['type']
                    for edge in self.scene_graph_global['edges']
                )
                
                if not edge_exists:
                    self.scene_graph_global['edges'].append(spatial_edge)
                
                # 检查语义关系
                semantic_relation = self._check_semantic_relationship(new_obj_name, existing_node['name'])
                if semantic_relation:
                    semantic_edge = {
                        'source': new_obj_id,
                        'target': existing_node['id'],
                        'type': f'semantic_{semantic_relation}',
                        'confidence': 0.7
                    }
                    
                    semantic_edge_exists = any(
                        edge['source'] == semantic_edge['source'] and 
                        edge['target'] == semantic_edge['target'] and
                        edge['type'] == semantic_edge['type']
                        for edge in self.scene_graph_global['edges']
                    )
                    
                    if not semantic_edge_exists:
                        self.scene_graph_global['edges'].append(semantic_edge)
                        
    def get_enhanced_scene_graph_for_overlap(self, direction: int) -> Dict:
        """
        获取增强的场景图用于overlap计算
        基于当前方向和semantic_voxel_map构建更精确的局部场景图
        """
        enhanced_graph = {
            'nodes': [],
            'edges': [],
            'direction_focus': direction,
            'semantic_coverage': {}
        }
        
        # 基于方向筛选相关的体素区域
        direction_rad = np.radians(direction)
        agent_pos = getattr(self, 'agent_state', None)
        
        if agent_pos and hasattr(agent_pos, 'position'):
            agent_coords = self._global_to_grid(agent_pos.position)
            
            # 定义方向扇形区域
            DIRECTION_CONE_ANGLE = np.radians(60)  # 60度锥形
            MAX_DISTANCE = 200  # 最大距离（像素）
            
            for voxel_key, voxel_data in self.semantic_voxel_map.items():
                voxel_pos = voxel_data['position']
                
                # 计算相对位置和角度
                relative_pos = np.array([voxel_pos[0] - agent_coords[0], voxel_pos[1] - agent_coords[1]])
                distance = np.linalg.norm(relative_pos)
                
                if distance > MAX_DISTANCE:
                    continue
                    
                # 计算角度
                voxel_angle = np.arctan2(relative_pos[0], -relative_pos[1])  # 注意坐标系转换
                angle_diff = abs(np.arctan2(np.sin(voxel_angle - direction_rad), 
                                          np.cos(voxel_angle - direction_rad)))
                
                if angle_diff <= DIRECTION_CONE_ANGLE / 2:
                    # 在方向锥形内，添加到场景图
                    for obj_name in voxel_data['objects']:
                        node_id = f"{obj_name}_{voxel_key}"
                        enhanced_graph['nodes'].append({
                            'id': node_id,
                            'name': obj_name,
                            'room_type': voxel_data['room_type'],
                            'position': voxel_pos,
                            'distance': distance,
                            'angle_offset': angle_diff,
                            'confidence': voxel_data['confidence'],
                            'observations': voxel_data['observations']
                        })
            
            # 从全局场景图中提取相关的边
            node_ids = {node['id'] for node in enhanced_graph['nodes']}
            for edge in self.scene_graph_global['edges']:
                if edge['source'] in node_ids and edge['target'] in node_ids:
                    enhanced_graph['edges'].append(edge)
                    
        print(f"   🌐 构建方向{direction}°增强场景图: {len(enhanced_graph['nodes'])} 节点, {len(enhanced_graph['edges'])} 边")
        return enhanced_graph
        
    def _check_semantic_relationship(self, obj1: str, obj2: str) -> str:
        """
        检查两个物体之间的语义关系
        返回关系类型字符串或None
        """
        obj1_lower = obj1.lower()
        obj2_lower = obj2.lower()
        
        # 首先检查容器关系 (优先级高)
        containers = ['shelf', 'cabinet', 'drawer', 'box', 'basket', 'bookshelf']
        if obj1_lower in containers and obj2_lower not in containers:
            return 'contains'
        if obj2_lower in containers and obj1_lower not in containers:
            return 'contained_by'
            
        # 检查支撑关系
        surfaces = ['table', 'desk', 'counter']
        if obj1_lower in surfaces and obj2_lower not in surfaces:
            return 'supports'
        if obj2_lower in surfaces and obj1_lower not in surfaces:
            return 'supported_by'
        
        # 最后检查功能关系 (精确匹配)
        functional_pairs = [
            ('bed', 'pillow'), ('bed', 'blanket'), ('bed', 'nightstand'),
            ('chair', 'table'), ('chair', 'desk'),
            ('sofa', 'coffee_table'), ('sofa', 'cushion'),
            ('toilet', 'toilet_paper'), ('toilet', 'sink'),
            ('stove', 'pan'), ('stove', 'pot'), 
            ('refrigerator', 'food'), ('microwave', 'food'),
            ('tv', 'remote'), ('tv', 'sofa'),
            ('lamp', 'table'), ('lamp', 'nightstand'),
            ('book', 'bookshelf'), ('book', 'table'),
        ]
        
        for item1, item2 in functional_pairs:
            if (obj1_lower == item1 and obj2_lower == item2) or (obj1_lower == item2 and obj2_lower == item1):
                return 'functional'
            
        return None
        
    def apply_scene_graph_corrections(self, scene_graph: Dict, reason: Dict = None, goal_context: str = None) -> Dict:
        """
        应用UniGoal风格的场景图修正
        使用LLM进行常识性错误检测和修正
        
        Args:
            scene_graph: 待修正的场景图
            reason: 来自update_curiosity_value的推理上下文
            goal_context: 当前目标的上下文信息
        """
        if not scene_graph['nodes']:
            return scene_graph
            
        # 构建场景描述
        scene_description = self._build_scene_description(scene_graph)
        
        # 构建增强的上下文信息
        context_info = ""
        if reason and isinstance(reason, dict):
            # 提取reasoning context
            spatial_analysis = reason.get('spatial_analysis', {})
            if spatial_analysis:
                context_info += f"\nSpatial Analysis Context:\n{spatial_analysis}\n"
            
            # 提取推理信息
            reasoning_text = reason.get('reasoning', '') or str(reason)
            if reasoning_text:
                context_info += f"\nPrevious Reasoning:\n{reasoning_text}\n"
        
        if goal_context:
            context_info += f"\nCurrent Goal: {goal_context}\n"
        
        # 增强的UniGoal风格修正提示
        correction_prompt = f"""Scene Graph Correction: You are an AI assistant with commonsense knowledge about indoor environments. 
        
        Given the following scene graph representation:

        {scene_description}

        {context_info}

        Based on the spatial analysis and reasoning context above, please identify any logical inconsistencies or implausible object arrangements and provide corrections. Consider:

        1. Spatial Context: Use the previous spatial analysis to validate object positions
        2. Typical object relationships: pillows are usually ON beds, books are usually IN/ON shelves
        3. Spatial constraints: large furniture items shouldn't overlap, objects need physical support
        4. Functional relationships: remote controls near TVs, lamps near tables/beds
        5. Room context appropriateness: kitchen objects in kitchens, bedroom objects in bedrooms
        6. Goal-oriented logic: prioritize corrections that help with navigation to the current goal

        Respond with a JSON object containing:
        {{
            "errors_found": ["specific issues identified"],
            "corrections": ["specific corrections to make"],
            "confidence": 0.8,
            "corrected_relationships": [
                {{"source": "object1", "target": "object2", "type": "relationship_type", "confidence": 0.9}}
            ],
            "reasoning_integration": "how the spatial context influenced the corrections"
        }}

        Focus on practical, actionable corrections that improve navigation planning and scene understanding."""

        try:
            # 使用当前配置的ReasonLLM进行修正
            if hasattr(self, 'ReasonLLM') and self.ReasonLLM:
                print("   🧠 调用ReasonLLM进行场景图修正...")
                llm_response = self.ReasonLLM.call(correction_prompt)
                
                # 解析LLM响应
                corrected_response = self._eval_response(llm_response)
                
                if corrected_response and isinstance(corrected_response, dict):
                    if corrected_response.get('corrected_relationships'):
                        # 应用修正
                        scene_graph = self._apply_corrections(scene_graph, corrected_response)
                        errors_found = corrected_response.get('errors_found', [])
                        reasoning_integration = corrected_response.get('reasoning_integration', '')
                        
                        print(f"   ✅ 场景图修正完成: 发现 {len(errors_found)} 个问题")
                        if reasoning_integration:
                            print(f"   🔄 推理整合: {reasoning_integration}")
                    else:
                        print("   ℹ️ LLM未发现需要修正的关系")
                else:
                    print("   ⚠️ LLM响应解析失败，保持原场景图")
            else:
                print("   ⚠️ ReasonLLM未初始化，跳过场景图修正")
                
        except Exception as e:
            print(f"   ⚠️ 场景图修正失败: {e}")
            # 在出错时也记录错误但继续执行
            
        return scene_graph
        
    def _build_scene_description(self, scene_graph: Dict) -> str:
        """
        构建场景图的文字描述用于LLM分析
        """
        descriptions = []
        
        # 描述节点
        room_objects = {}
        for node in scene_graph['nodes']:
            room_type = node.get('room_type', 'unknown')
            if room_type not in room_objects:
                room_objects[room_type] = []
            room_objects[room_type].append(node['name'])
            
        for room_type, objects in room_objects.items():
            descriptions.append(f"Room: {room_type}, Objects: {', '.join(objects)}")
            
        # 描述关系
        relationships = []
        for edge in scene_graph['edges']:
            source_name = next((node['name'] for node in scene_graph['nodes'] if node['id'] == edge['source']), 'unknown')
            target_name = next((node['name'] for node in scene_graph['nodes'] if node['id'] == edge['target']), 'unknown')
            relationships.append(f"{source_name} --{edge['type']}--> {target_name}")
            
        if relationships:
            descriptions.append(f"Relationships: {'; '.join(relationships)}")
            
        return '\n'.join(descriptions)
        
    def _apply_corrections(self, scene_graph: Dict, corrections: Dict) -> Dict:
        """
        应用LLM建议的修正到场景图
        """
        corrected_rels = corrections.get('corrected_relationships', [])
        
        # 创建修正后的场景图副本
        corrected_graph = {
            'nodes': scene_graph['nodes'].copy(),
            'edges': [],
            'corrections_applied': True,
            'correction_confidence': corrections.get('confidence', 0.5)
        }
        
        # 应用修正后的关系
        for rel in corrected_rels:
            if isinstance(rel, dict) and 'source' in rel and 'target' in rel:
                # 查找对应的节点ID
                source_node = next((node for node in corrected_graph['nodes'] 
                                  if node['name'].lower() == rel['source'].lower()), None)
                target_node = next((node for node in corrected_graph['nodes'] 
                                  if node['name'].lower() == rel['target'].lower()), None)
                
                if source_node and target_node:
                    corrected_edge = {
                        'source': source_node['id'],
                        'target': target_node['id'],
                        'type': rel.get('type', 'corrected'),
                        'confidence': rel.get('confidence', 0.8),
                        'corrected': True
                    }
                    corrected_graph['edges'].append(corrected_edge)
                    
        # 保留未修正的高置信度边
        for edge in scene_graph['edges']:
            if edge.get('confidence', 0.5) > 0.7:
                edge_exists = any(
                    existing['source'] == edge['source'] and 
                    existing['target'] == edge['target']
                    for existing in corrected_graph['edges']
                )
                if not edge_exists:
                    corrected_graph['edges'].append(edge)
    def _calculate_graph_overlap_score_with_subgraph(self, vlm_objects: Dict, goal_subgraph: Dict, direction: int = 0, reason: Dict = None) -> Dict:
        """
        增强版分层图重叠计算：充分利用深度信息和空间关系精度
        基于房间-对象分层架构，集成深度感知和关系推理
        """
        try:
            # 获取当前方向的增强版分层场景图
            hierarchical_scene_graph = self.get_enhanced_scene_graph_for_overlap(direction)
            
            # 应用场景图修正（传入reason上下文）
            goal_context = goal_subgraph.get('target_object', '') if isinstance(goal_subgraph, dict) else ''
            corrected_scene_graph = self.apply_scene_graph_corrections(hierarchical_scene_graph, reason, goal_context)
            
            # === 阶段1: 增强版房间级别匹配（考虑深度分布） ===
            room_level_scores = self._calculate_enhanced_room_level_overlap(corrected_scene_graph, goal_subgraph)
            
            # === 阶段2: 深度感知对象级别匹配 ===
            object_level_scores = self._calculate_depth_aware_object_overlap(
                vlm_objects, corrected_scene_graph, goal_subgraph, room_level_scores
            )
            
            # === 阶段3: 空间关系匹配（深度兼容性增强） ===
            spatial_relationship_scores = self._calculate_depth_enhanced_spatial_match(
                corrected_scene_graph, goal_subgraph
            )
            
            # === 阶段4: 语义一致性验证（空间精度考虑） ===
            semantic_consistency_scores = self._calculate_precision_aware_semantic_consistency(
                corrected_scene_graph, goal_subgraph, vlm_objects
            )
            
            # === 阶段5: 深度分布一致性评分 ===
            depth_consistency_scores = self._calculate_depth_distribution_consistency(
                corrected_scene_graph, goal_subgraph
            )
            
            # === 综合评分计算（动态权重 + 深度增强） ===
            room_match_score = room_level_scores['room_match_score']
            object_match_score = object_level_scores['object_match_score']
            spatial_match_score = spatial_relationship_scores['spatial_match_score']
            semantic_consistency = semantic_consistency_scores['consistency_score']
            depth_consistency = depth_consistency_scores.get('depth_consistency', 0.5)
            
            # 基于场景图质量的动态权重调整
            scene_quality = self._assess_scene_graph_quality(corrected_scene_graph)
            
            # 检查是否检测到目标物体
            target_detected = object_level_scores.get('target_detected', False)
            target_objects_found = object_level_scores.get('target_objects_found', [])
            
            if target_detected:
                # 🎯 目标检测加成：当发现目标物体时大幅提升对象权重
                print(f"   🎯 检测到目标物体: {target_objects_found}")
                room_weight = 0.25
                object_weight = 0.50  # 显著提升对象权重
                spatial_weight = 0.20
                semantic_weight = 0.03
                depth_weight = 0.02
            elif scene_quality > 0.8:
                # 高质量场景图：更信任空间关系
                room_weight = 0.35
                object_weight = 0.30
                spatial_weight = 0.25
                semantic_weight = 0.05
                depth_weight = 0.05
            elif room_match_score > 0.7:
                # 房间匹配很好时
                room_weight = 0.45
                object_weight = 0.25
                spatial_weight = 0.20
                semantic_weight = 0.05
                depth_weight = 0.05
            elif object_match_score > 0.7:
                # 对象匹配很好时
                room_weight = 0.30
                object_weight = 0.40
                spatial_weight = 0.20
                semantic_weight = 0.05
                depth_weight = 0.05
            else:
                # 平衡权重
                room_weight = 0.35
                object_weight = 0.30
                spatial_weight = 0.20
                semantic_weight = 0.10
                depth_weight = 0.05
            
            # 基础综合评分
            base_final_score = (
                room_match_score * room_weight +
                object_match_score * object_weight +
                spatial_match_score * spatial_weight +
                semantic_consistency * semantic_weight +
                depth_consistency * depth_weight
            )
            
            # === 增强版部分匹配加成逻辑 ===
            # 深度感知的部分匹配加成
            depth_enhanced_bonus = 0.0
            
            # 如果房间部分匹配但对象匹配度不错且深度一致性好
            if 0.3 <= room_match_score < 0.8 and object_match_score > 0.5 and depth_consistency > 0.6:
                depth_enhanced_bonus += min(0.15, object_match_score * depth_consistency * 0.2)
            
            # 如果对象部分匹配但房间匹配度不错且空间精度高
            if 0.3 <= object_match_score < 0.8 and room_match_score > 0.5:
                spatial_precision_bonus = scene_quality * 0.1
                depth_enhanced_bonus += min(0.15, room_match_score * 0.15 + spatial_precision_bonus)
            
            # 空间关系质量加成
            if len(corrected_scene_graph.get('spatial_relationships', [])) > 0:
                relation_quality = sum(r.get('confidence', 0.5) for r in corrected_scene_graph['spatial_relationships'])
                relation_quality /= len(corrected_scene_graph['spatial_relationships'])
                if relation_quality > 0.7:
                    depth_enhanced_bonus += min(0.1, relation_quality * 0.1)
            
            base_final_score += depth_enhanced_bonus
            
            # 应用方向相关性和深度方向性衰减
            if hasattr(self, 'semantic_voxel_map') and self.semantic_voxel_map:
                direction_relevance = self._calculate_direction_relevance(direction)
                depth_direction_factor = self._calculate_depth_direction_factor(corrected_scene_graph, direction)
                final_score = base_final_score * direction_relevance * depth_direction_factor
            else:
                direction_relevance = 1.0
                depth_direction_factor = 1.0
                final_score = base_final_score
            
            # 构建详细结果（增强版）
            overlap_result = {
                'overlap_score': min(1.0, final_score),
                
                # 房间级别指标（增强）
                'room_match_score': room_level_scores['room_match_score'],
                'matched_rooms': room_level_scores['matched_rooms'],
                'room_confidence': room_level_scores['room_confidence'],
                'room_depth_diversity': room_level_scores.get('depth_diversity_avg', 0.0),
                
                # 对象级别指标（深度感知）
                'object_match_score': object_level_scores['object_match_score'],
                'vlm_target_match': object_level_scores['vlm_target_match'],
                'scene_target_match': object_level_scores['scene_target_match'],
                'matched_objects': object_level_scores['matched_objects'],
                'depth_layer_matches': object_level_scores.get('depth_layer_matches', {}),
                
                # 空间关系指标（深度增强）
                'spatial_match_score': spatial_relationship_scores['spatial_match_score'],
                'room_spatial_score': spatial_relationship_scores['room_spatial_score'],
                'object_spatial_score': spatial_relationship_scores['object_spatial_score'],
                'depth_compatibility_score': spatial_relationship_scores.get('depth_compatibility_score', 0.5),
                
                # 语义一致性指标（精度感知）
                'consistency_score': semantic_consistency_scores['consistency_score'],
                'vlm_scene_consistency': semantic_consistency_scores['vlm_scene_consistency'],
                'semantic_richness': semantic_consistency_scores['semantic_richness'],
                'spatial_precision_score': semantic_consistency_scores.get('spatial_precision_score', 0.5),
                
                # 深度分布指标
                'depth_consistency': depth_consistency,
                'depth_layer_distribution': depth_consistency_scores.get('layer_distribution', {}),
                'depth_scene_alignment': depth_consistency_scores.get('scene_alignment', 0.5),
                
                # 方向和图结构指标（增强）
                'direction_relevance': direction_relevance,
                'depth_direction_factor': depth_direction_factor,
                'scene_graph_corrected': corrected_scene_graph.get('corrections_applied', False),
                'scene_graph_quality': scene_quality,
                'hierarchical_structure': True,
                'total_rooms': len(corrected_scene_graph.get('room_nodes', {})),
                'total_objects': len(corrected_scene_graph.get('object_nodes', [])),
                'spatial_relationships_count': len(corrected_scene_graph.get('spatial_relationships', [])),
                'voxel_validation_rate': self._calculate_voxel_validation_rate(corrected_scene_graph),
                'optimization_used': 'hierarchical_depth_enhanced',
                'depth_enhanced_bonus': depth_enhanced_bonus
            }
            
            print(f"   🏗️ 深度增强分层重叠: 总分{final_score:.3f} "
                  f"(房间:{room_match_score:.3f}, 对象:{object_match_score:.3f}, "
                  f"空间:{spatial_match_score:.3f}, 深度:{depth_consistency:.3f}) "
                  f"质量:{scene_quality:.3f}")
            
            return overlap_result
            
        except Exception as e:
            print(f"   ⚠️ 深度增强重叠计算失败: {e}")
            # 回退到基础计算
            return self._fallback_overlap_calculation(vlm_objects, goal_subgraph)
    
    def _calculate_enhanced_room_level_overlap(self, scene_graph: Dict, goal_subgraph: Dict) -> Dict:
        """增强版房间级别重叠计算，考虑深度分布"""
        # 调用原有的房间级别重叠计算，然后增强
        base_result = self._calculate_room_level_overlap(scene_graph, goal_subgraph)
        
        # 计算深度多样性平均值
        room_nodes = scene_graph.get('room_nodes', {})
        if room_nodes:
            depth_diversities = [node.get('depth_diversity', 0.0) for node in room_nodes.values()]
            base_result['depth_diversity_avg'] = sum(depth_diversities) / len(depth_diversities)
        else:
            base_result['depth_diversity_avg'] = 0.0
        
        return base_result
    
    def _calculate_depth_aware_object_overlap(self, vlm_objects: Dict, scene_graph: Dict, goal_subgraph: Dict, room_scores: Dict) -> Dict:
        """深度感知的对象级别重叠计算"""
        # 调用原有的对象级别重叠计算
        base_result = self._calculate_object_level_overlap(vlm_objects, scene_graph, goal_subgraph, room_scores)
        
        # 添加深度层次匹配统计
        depth_layer_matches = {'foreground': 0, 'mid-ground': 0, 'background': 0}
        
        object_nodes = scene_graph.get('object_nodes', [])
        for obj_node in object_nodes:
            depth_layer = obj_node.get('depth_info', {}).get('depth_category', 'mid-ground')
            if depth_layer in depth_layer_matches:
                depth_layer_matches[depth_layer] += 1
        
        base_result['depth_layer_matches'] = depth_layer_matches
        return base_result
    
    def _calculate_depth_enhanced_spatial_match(self, scene_graph: Dict, goal_subgraph: Dict) -> Dict:
        """深度兼容性增强的空间关系匹配"""
        # 调用原有的空间关系匹配
        base_result = self._calculate_hierarchical_spatial_match(scene_graph, goal_subgraph)
        
        # 计算深度兼容性评分
        object_edges = scene_graph.get('object_edges', [])
        if object_edges:
            depth_compatible_edges = sum(1 for edge in object_edges if edge.get('depth_compatible', 0.5) > 0.7)
            depth_compatibility_score = depth_compatible_edges / len(object_edges)
        else:
            depth_compatibility_score = 0.5
        
        base_result['depth_compatibility_score'] = depth_compatibility_score
        return base_result
    
    def _calculate_precision_aware_semantic_consistency(self, scene_graph: Dict, goal_subgraph: Dict, vlm_objects: Dict) -> Dict:
        """空间精度感知的语义一致性计算"""
        # 调用原有的语义一致性计算
        base_result = self._calculate_semantic_consistency(scene_graph, goal_subgraph, vlm_objects)
        
        # 计算空间精度评分
        object_nodes = scene_graph.get('object_nodes', [])
        if object_nodes:
            precision_scores = []
            for obj_node in object_nodes:
                precision = obj_node.get('spatial_precision', 'medium')
                score = {'high': 1.0, 'medium': 0.6, 'low': 0.3}.get(precision, 0.5)
                precision_scores.append(score)
            spatial_precision_score = sum(precision_scores) / len(precision_scores)
        else:
            spatial_precision_score = 0.5
        
        base_result['spatial_precision_score'] = spatial_precision_score
        return base_result
    
    def _calculate_depth_distribution_consistency(self, scene_graph: Dict, goal_subgraph: Dict) -> Dict:
        """计算深度分布一致性"""
        depth_layers = scene_graph.get('depth_layers', {})
        
        # 计算当前场景的深度分布
        total_objects = sum(len(objects) for objects in depth_layers.values())
        if total_objects > 0:
            layer_distribution = {
                layer: len(objects) / total_objects 
                for layer, objects in depth_layers.items()
            }
        else:
            layer_distribution = {'foreground': 0.33, 'mid-ground': 0.34, 'background': 0.33}
        
        # 理想的深度分布（可以根据目标调整）
        ideal_distribution = {'foreground': 0.3, 'mid-ground': 0.4, 'background': 0.3}
        
        # 计算分布相似性
        similarity = 0.0
        for layer in ['foreground', 'mid-ground', 'background']:
            current = layer_distribution.get(layer, 0.0)
            ideal = ideal_distribution.get(layer, 0.33)
            similarity += 1.0 - abs(current - ideal)
        
        depth_consistency = similarity / 3.0
        
        return {
            'depth_consistency': depth_consistency,
            'layer_distribution': layer_distribution,
            'scene_alignment': min(1.0, depth_consistency + 0.1)
        }
    
    def _assess_scene_graph_quality(self, scene_graph: Dict) -> float:
        """评估场景图质量"""
        quality_score = 0.0
        
        # 节点质量
        object_nodes = scene_graph.get('object_nodes', [])
        if object_nodes:
            voxel_validated = sum(1 for obj in object_nodes if obj.get('voxel_validated', False))
            high_precision = sum(1 for obj in object_nodes if obj.get('spatial_precision') == 'high')
            node_quality = (voxel_validated * 0.6 + high_precision * 0.4) / len(object_nodes)
            quality_score += node_quality * 0.4
        
        # 关系质量
        spatial_relationships = scene_graph.get('spatial_relationships', [])
        if spatial_relationships:
            avg_confidence = sum(r.get('confidence', 0.5) for r in spatial_relationships) / len(spatial_relationships)
            quality_score += avg_confidence * 0.3
        
        # 房间质量
        room_nodes = scene_graph.get('room_nodes', {})
        if room_nodes:
            avg_room_confidence = sum(node.get('avg_confidence', 0.5) for node in room_nodes.values()) / len(room_nodes)
            quality_score += avg_room_confidence * 0.3
        
        return min(1.0, quality_score)
    
    def _calculate_depth_direction_factor(self, scene_graph: Dict, direction: int) -> float:
        """计算深度-方向因子"""
        # 简单实现：根据方向和深度分布的一致性
        depth_layers = scene_graph.get('depth_layers', {})
        
        # 前向方向更适合前景对象，侧向更适合背景
        if direction in [30, 330]:  # 前向
            foreground_weight = 1.2
            background_weight = 0.8
        elif direction in [90, 270]:  # 侧向
            foreground_weight = 0.9
            background_weight = 1.1
        else:  # 其他方向
            foreground_weight = 1.0
            background_weight = 1.0
        
        total_objects = sum(len(objects) for objects in depth_layers.values())
        if total_objects > 0:
            foreground_ratio = len(depth_layers.get('foreground', [])) / total_objects
            background_ratio = len(depth_layers.get('background', [])) / total_objects
            
            factor = (foreground_ratio * foreground_weight + 
                     background_ratio * background_weight + 
                     (1 - foreground_ratio - background_ratio))
            return min(1.2, factor)
        
        return 1.0
    
    def _calculate_voxel_validation_rate(self, scene_graph: Dict) -> float:
        """计算voxel验证率"""
        object_nodes = scene_graph.get('object_nodes', [])
        if object_nodes:
            validated = sum(1 for obj in object_nodes if obj.get('voxel_validated', False))
            return validated / len(object_nodes)
        return 0.0
            
    def _calculate_spatial_relationship_match(self, scene_graph: Dict, goal_subgraph: Dict) -> float:
        """
        计算空间关系匹配度
        """
        if not scene_graph.get('edges') or not isinstance(goal_subgraph, dict):
            return 0.0
            
        scene_relationships = set()
        for edge in scene_graph['edges']:
            source_name = next((node['name'] for node in scene_graph['nodes'] 
                              if node['id'] == edge['source']), '')
            target_name = next((node['name'] for node in scene_graph['nodes'] 
                              if node['id'] == edge['target']), '')
            if source_name and target_name:
                rel_key = f"{source_name.lower()}_{edge['type']}_{target_name.lower()}"
                scene_relationships.add(rel_key)
        
        # 从目标子图中提取预期关系
        target_relationships = set()
        goal_edges = goal_subgraph.get('edges', [])
        goal_nodes = goal_subgraph.get('nodes', [])
        
        if isinstance(goal_edges, list) and isinstance(goal_nodes, list):
            for edge in goal_edges:
                if isinstance(edge, dict):
                    source_name = edge.get('source', '')
                    target_name = edge.get('target', '')
                    rel_type = edge.get('type', 'related')
                    if source_name and target_name:
                        rel_key = f"{source_name.lower()}_{rel_type}_{target_name.lower()}"
                        target_relationships.add(rel_key)
        
        if not target_relationships:
            return 0.5  # 中性分数，如果没有目标关系信息
            
        # 计算匹配度
        matches = len(scene_relationships.intersection(target_relationships))
        total = len(scene_relationships.union(target_relationships))
        
        return matches / max(total, 1)
        
    def _calculate_semantic_richness_score(self, scene_graph: Dict) -> float:
        """
        计算场景图的语义丰富度
        """
        nodes = scene_graph.get('nodes', [])
        edges = scene_graph.get('edges', [])
        
        if not nodes:
            return 0.0
            
        # 节点丰富度：不同物体类型数量
        unique_objects = len(set(node['name'].lower() for node in nodes))
        node_richness = min(1.0, unique_objects / 10)  # 归一化到10个物体
        
        # 关系丰富度：不同关系类型数量
        unique_relations = len(set(edge.get('type', 'unknown') for edge in edges))
        relation_richness = min(1.0, unique_relations / 5)  # 归一化到5种关系类型
        
        # 置信度加权
        avg_confidence = np.mean([node.get('confidence', 0.5) for node in nodes])
        
        return (node_richness * 0.4 + relation_richness * 0.4 + avg_confidence * 0.2)
        
    def _calculate_direction_relevance(self, target_direction: int) -> float:
        """
        基于语义体素映射计算方向相关性
        """
        if not hasattr(self, 'semantic_voxel_map') or not self.semantic_voxel_map:
            return 1.0
            
        direction_rad = np.radians(target_direction)
        total_relevance = 0.0
        total_weight = 0.0
        
        agent_pos = getattr(self, 'agent_state', None)
        if not agent_pos or not hasattr(agent_pos, 'position'):
            return 1.0
            
        agent_coords = self._global_to_grid(agent_pos.position)
        
        for voxel_key, voxel_data in self.semantic_voxel_map.items():
            voxel_pos = voxel_data['position']
            
            # 计算相对角度
            relative_pos = np.array([voxel_pos[0] - agent_coords[0], voxel_pos[1] - agent_coords[1]])
            distance = np.linalg.norm(relative_pos)
            
            if distance < 200:  # 只考虑附近的体素
                voxel_angle = np.arctan2(relative_pos[0], -relative_pos[1])
                angle_diff = abs(np.arctan2(np.sin(voxel_angle - direction_rad), 
                                          np.cos(voxel_angle - direction_rad)))
                
                # 基于角度差计算相关性 (越小越相关)
                angle_relevance = max(0, 1 - (angle_diff / np.pi))
                
                # 基于距离的权重 (越近权重越高)
                distance_weight = max(0.1, 1 - (distance / 200))
                
                # 基于观察次数的权重
                observation_weight = min(1.0, voxel_data['observations'] / 5)
                
                weighted_relevance = angle_relevance * distance_weight * observation_weight
                total_relevance += weighted_relevance
                total_weight += distance_weight * observation_weight
                
        return total_relevance / max(total_weight, 0.1)
        
    def _calculate_room_level_overlap(self, hierarchical_graph: Dict, goal_subgraph: Dict) -> Dict:
        """
        计算房间级别的重叠匹配度
        """
        try:
            # 提取目标子图中的房间信息
            goal_rooms = set()
            goal_spatial_regions = set()
            
            if isinstance(goal_subgraph, dict):
                # 从节点中提取房间相关信息
                nodes = goal_subgraph.get('nodes', [])
                for node in nodes:
                    if isinstance(node, dict):
                        node_name = node.get('name', '').lower()
                        # 识别房间类型节点
                        if any(room_word in node_name for room_word in 
                               ['kitchen', 'bathroom', 'bedroom', 'living', 'dining', 'office']):
                            goal_rooms.add(node_name)
                        
                        # 识别空间区域
                        spatial_region = self._classify_spatial_region(node.get('position', {}))
                        if spatial_region != 'unknown':
                            goal_spatial_regions.add(spatial_region)
            
            # 提取当前场景图中的房间节点
            current_rooms = set()
            current_spatial_regions = set()
            
            room_nodes = hierarchical_graph.get('room_nodes', {})
            for room_id, room_data in room_nodes.items():
                room_name = room_data.get('name', room_id).lower()
                current_rooms.add(room_name)
                
                # 获取房间的空间区域
                room_region = room_data.get('spatial_region', 'unknown')
                if room_region != 'unknown':
                    current_spatial_regions.add(room_region)
            
            # 计算房间名称匹配度
            room_name_overlap = len(current_rooms.intersection(goal_rooms))
            room_name_total = len(current_rooms.union(goal_rooms))
            room_name_score = room_name_overlap / max(room_name_total, 1) if room_name_total > 0 else 0
            
            # 计算空间区域匹配度
            spatial_overlap = len(current_spatial_regions.intersection(goal_spatial_regions))
            spatial_total = len(current_spatial_regions.union(goal_spatial_regions))
            spatial_score = spatial_overlap / max(spatial_total, 1) if spatial_total > 0 else 0
            
            # 房间连接性评分
            connectivity_score = self._evaluate_room_connectivity(room_nodes)
            
            # 综合房间匹配分数 (名称匹配50% + 空间区域匹配30% + 连接性20%)
            room_match_score = (room_name_score * 0.5 + spatial_score * 0.3 + connectivity_score * 0.2)
            
            return {
                'room_match_score': min(1.0, room_match_score),
                'matched_rooms': list(current_rooms.intersection(goal_rooms)),
                'room_confidence': room_name_score,
                'spatial_region_match': spatial_score,
                'connectivity_score': connectivity_score,
                'total_current_rooms': len(current_rooms),
                'total_goal_rooms': len(goal_rooms)
            }
            
        except Exception as e:
            print(f"   ⚠️ 房间级别匹配计算失败: {e}")
            return {
                'room_match_score': 0.0,
                'matched_rooms': [],
                'room_confidence': 0.0,
                'spatial_region_match': 0.0,
                'connectivity_score': 0.0
            }
    
    def _calculate_object_level_overlap(self, vlm_objects: Dict, hierarchical_graph: Dict, goal_subgraph: Dict, room_scores: Dict) -> Dict:
        """
        计算对象级别的重叠匹配度，利用房间级别筛选优化
        特别加强对目标物体的检测和评分
        """
        try:
            # 从VLM预测中提取物体
            current_objects = set()
            if isinstance(vlm_objects, dict):
                for direction_key, direction_data in vlm_objects.items():
                    if isinstance(direction_data, dict):
                        objects = direction_data.get('Objects', direction_data.get('objects', {}))
                        
                        # 处理字典格式的Objects（主要格式）
                        if isinstance(objects, dict):
                            for obj_name, position_info in objects.items():
                                current_objects.add(obj_name.lower())
                        
                        # 处理列表格式的Objects（备用格式）
                        elif isinstance(objects, list):
                            for obj in objects:
                                obj_name = obj if isinstance(obj, str) else obj.get('name', str(obj))
                                current_objects.add(obj_name.lower())
            
            # 从目标子图中提取目标物体名称
            target_objects = set()
            current_goal = getattr(self, '_current_goal', '').lower() if hasattr(self, '_current_goal') else ''
            
            # 首先添加当前目标
            if current_goal:
                target_objects.add(current_goal)
                print(f"   🎯 当前目标物体: {current_goal}")
            
            # 然后从goal_subgraph中提取
            if isinstance(goal_subgraph, dict):
                # 从spatial_anchor_objects提取真实的物体名称
                spatial_anchors = goal_subgraph.get('spatial_anchor_objects', {})
                if isinstance(spatial_anchors, dict):
                    for obj_key, obj_info in spatial_anchors.items():
                        if isinstance(obj_info, dict):
                            # 提取实际的物体名称，而不是占位符键名
                            actual_name = obj_info.get('name', obj_key)
                            target_objects.add(str(actual_name).lower())
                        else:
                            target_objects.add(str(obj_info).lower())
                elif isinstance(spatial_anchors, list):
                    for obj_name in spatial_anchors:
                        target_objects.add(str(obj_name).lower())
                
                # 从nodes中提取（备用）
                nodes = goal_subgraph.get('nodes', [])
                if isinstance(nodes, list):
                    for node in nodes:
                        if isinstance(node, dict):
                            obj_name = node.get('name', node.get('id', ''))
                            if obj_name:
                                target_objects.add(str(obj_name).lower())
            
            print(f"   📊 对象匹配分析: VLM检测={list(current_objects)}, 目标={list(target_objects)}")
            
            # 从分层场景图中提取物体（优先考虑匹配房间中的物体）
            scene_graph_objects = set()
            prioritized_objects = set()
            
            object_nodes = hierarchical_graph.get('object_nodes', [])
            matched_rooms = set(room_scores.get('matched_rooms', []))
            
            for obj_node in object_nodes:
                obj_name = obj_node.get('name', '').lower()
                scene_graph_objects.add(obj_name)
                
                # 如果物体在匹配的房间中，给予优先权重
                obj_room = obj_node.get('room', '').lower()
                if obj_room in matched_rooms or matched_rooms == set():
                    prioritized_objects.add(obj_name)
            
            # 计算VLM-目标重叠（关键！）
            vlm_target_intersection = current_objects.intersection(target_objects)
            vlm_target_overlap = len(vlm_target_intersection)
            vlm_target_total = len(current_objects.union(target_objects))
            
            # 当发现目标物体时，显著提升分数
            if vlm_target_overlap > 0:
                # 基础匹配分数
                base_vlm_target_score = vlm_target_overlap / max(vlm_target_total, 1)
                
                # 目标检测加成：每个目标物体+0.5分，上限0.9
                target_detection_bonus = min(0.6, vlm_target_overlap * 0.5)
                
                # 完全匹配奖励：如果所有目标都找到了
                if len(vlm_target_intersection) == len(target_objects) and len(target_objects) > 0:
                    perfect_match_bonus = 0.2
                else:
                    perfect_match_bonus = 0.0
                
                vlm_target_score = min(1.0, base_vlm_target_score + target_detection_bonus + perfect_match_bonus)
                
                print(f"   🎯 目标检测成功! 发现: {list(vlm_target_intersection)}")
                print(f"      基础分数: {base_vlm_target_score:.3f}, 检测奖励: {target_detection_bonus:.3f}, 完美匹配奖励: {perfect_match_bonus:.3f}")
                print(f"      最终VLM-目标分数: {vlm_target_score:.3f}")
            else:
                vlm_target_score = vlm_target_overlap / max(vlm_target_total, 1) if vlm_target_total > 0 else 0.0
                print(f"   ❌ 未检测到目标物体，VLM-目标分数: {vlm_target_score:.3f}")
            
            # 计算场景图-目标重叠（优先权重加成）
            scene_target_overlap = len(scene_graph_objects.intersection(target_objects))
            prioritized_overlap = len(prioritized_objects.intersection(target_objects))
            scene_target_total = len(scene_graph_objects.union(target_objects))
            
            base_scene_score = scene_target_overlap / max(scene_target_total, 1)
            priority_bonus = (prioritized_overlap / max(len(target_objects), 1)) * 0.2  # 20%优先权重
            scene_target_score = min(1.0, base_scene_score + priority_bonus)
            
            # VLM-场景图一致性
            vlm_scene_overlap = len(current_objects.intersection(scene_graph_objects))
            vlm_scene_total = len(current_objects.union(scene_graph_objects))
            vlm_scene_consistency = vlm_scene_overlap / max(vlm_scene_total, 1)
            
            # 综合对象匹配分数 - 重点突出VLM-目标匹配
            if vlm_target_overlap > 0:
                # 发现目标时，大幅提升权重
                object_match_score = (vlm_target_score * 0.7 + scene_target_score * 0.2 + vlm_scene_consistency * 0.1)
            else:
                # 未发现目标时，使用平衡权重
                object_match_score = (vlm_target_score * 0.4 + scene_target_score * 0.4 + vlm_scene_consistency * 0.2)
            
            # 应用房间匹配加成
            room_bonus = room_scores.get('room_match_score', 0) * 0.1
            object_match_score = min(1.0, object_match_score + room_bonus)
            
            return {
                'object_match_score': object_match_score,
                'vlm_target_match': vlm_target_score,
                'scene_target_match': scene_target_score,
                'vlm_scene_consistency': vlm_scene_consistency,
                'matched_objects': list(current_objects.intersection(target_objects)),
                'prioritized_matches': list(prioritized_objects.intersection(target_objects)),
                'room_bonus_applied': room_bonus,
                'target_detected': vlm_target_overlap > 0,
                'target_objects_found': list(vlm_target_intersection) if vlm_target_overlap > 0 else []
            }
            
        except Exception as e:
            print(f"   ⚠️ 对象级别匹配计算失败: {e}")
            return {
                'object_match_score': 0.0,
                'vlm_target_match': 0.0,
                'scene_target_match': 0.0,
                'vlm_scene_consistency': 0.0,
                'matched_objects': [],
                'prioritized_matches': [],
                'target_detected': False,
                'target_objects_found': []
            }
    
    def _calculate_hierarchical_spatial_match(self, hierarchical_graph: Dict, goal_subgraph: Dict) -> Dict:
        """
        计算分层空间关系匹配度
        """
        try:
            # 房间层级空间关系
            room_spatial_score = self._evaluate_room_spatial_relationships(
                hierarchical_graph.get('room_edges', []), goal_subgraph
            )
            
            # 对象层级空间关系
            object_spatial_score = self._evaluate_object_spatial_relationships(
                hierarchical_graph.get('object_edges', []), 
                hierarchical_graph.get('room_object_edges', []),
                goal_subgraph
            )
            
            # 跨层级关系匹配
            cross_level_score = self._evaluate_cross_level_relationships(
                hierarchical_graph, goal_subgraph
            )
            
            # 综合空间关系分数 (房间40% + 对象40% + 跨层级20%)
            spatial_match_score = (
                room_spatial_score * 0.4 + 
                object_spatial_score * 0.4 + 
                cross_level_score * 0.2
            )
            
            return {
                'spatial_match_score': min(1.0, spatial_match_score),
                'room_spatial_score': room_spatial_score,
                'object_spatial_score': object_spatial_score,
                'cross_level_score': cross_level_score
            }
            
        except Exception as e:
            print(f"   ⚠️ 分层空间关系计算失败: {e}")
            return {
                'spatial_match_score': 0.0,
                'room_spatial_score': 0.0,
                'object_spatial_score': 0.0,
                'cross_level_score': 0.0
            }
    
    def _calculate_semantic_consistency(self, hierarchical_graph: Dict, goal_subgraph: Dict, vlm_objects: Dict) -> Dict:
        """
        计算语义一致性分数
        """
        try:
            # VLM与场景图语义一致性
            vlm_scene_consistency = self._evaluate_vlm_scene_semantic_alignment(
                vlm_objects, hierarchical_graph
            )
            
            # 场景图语义丰富度
            semantic_richness = self._calculate_hierarchical_semantic_richness(hierarchical_graph)
            
            # 目标导向语义相关性
            goal_relevance = self._evaluate_goal_semantic_relevance(
                hierarchical_graph, goal_subgraph
            )
            
            # 综合语义一致性分数
            consistency_score = (
                vlm_scene_consistency * 0.4 + 
                semantic_richness * 0.3 + 
                goal_relevance * 0.3
            )
            
            return {
                'consistency_score': min(1.0, consistency_score),
                'vlm_scene_consistency': vlm_scene_consistency,
                'semantic_richness': semantic_richness,
                'goal_relevance': goal_relevance
            }
            
        except Exception as e:
            print(f"   ⚠️ 语义一致性计算失败: {e}")
            return {
                'consistency_score': 0.0,
                'vlm_scene_consistency': 0.0,
                'semantic_richness': 0.0,
                'goal_relevance': 0.0
            }
    
    def _evaluate_room_connectivity(self, room_nodes: Dict) -> float:
        """评估房间连接性得分"""
        try:
            if not room_nodes or len(room_nodes) <= 1:
                return 1.0
            
            # 计算房间间的连接度
            total_rooms = len(room_nodes)
            connected_pairs = 0
            
            room_list = list(room_nodes.keys())
            for i in range(len(room_list)):
                for j in range(i + 1, len(room_list)):
                    room1_data = room_nodes[room_list[i]]
                    room2_data = room_nodes[room_list[j]]
                    
                    # 检查房间是否相邻
                    if self._check_room_connection(room1_data, room2_data):
                        connected_pairs += 1
            
            max_connections = total_rooms * (total_rooms - 1) / 2
            connectivity_score = connected_pairs / max_connections if max_connections > 0 else 1.0
            
            return min(1.0, connectivity_score)
        except:
            return 0.5
    
    def _evaluate_room_spatial_relationships(self, room_edges: List, goal_subgraph: Dict) -> float:
        """评估房间空间关系匹配度"""
        try:
            if not room_edges:
                return 0.5
            
            # 提取目标子图中的房间关系
            goal_room_relations = self._extract_room_relations_from_goal(goal_subgraph)
            
            if not goal_room_relations:
                return 0.7  # 无明确房间关系要求时给中等分
            
            # 计算匹配的房间关系数量
            matched_relations = 0
            for edge in room_edges:
                edge_relation = f"{edge.get('source', '')}-{edge.get('relation', '')}-{edge.get('target', '')}"
                if any(goal_rel in edge_relation.lower() for goal_rel in goal_room_relations):
                    matched_relations += 1
            
            match_score = matched_relations / max(len(room_edges), len(goal_room_relations))
            return min(1.0, match_score)
        except:
            return 0.5
    
    def _evaluate_object_spatial_relationships(self, object_edges: List, room_object_edges: List, goal_subgraph: Dict) -> float:
        """评估对象空间关系匹配度"""
        try:
            # 提取目标子图中的对象关系
            goal_object_relations = self._extract_object_relations_from_goal(goal_subgraph)
            
            if not goal_object_relations:
                return 0.7
            
            all_object_edges = object_edges + room_object_edges
            if not all_object_edges:
                return 0.3
            
            matched_relations = 0
            for edge in all_object_edges:
                edge_relation = f"{edge.get('source', '')}-{edge.get('relation', '')}-{edge.get('target', '')}"
                if any(goal_rel in edge_relation.lower() for goal_rel in goal_object_relations):
                    matched_relations += 1
            
            match_score = matched_relations / max(len(all_object_edges), len(goal_object_relations))
            return min(1.0, match_score)
        except:
            return 0.5
    
    def _evaluate_cross_level_relationships(self, hierarchical_graph: Dict, goal_subgraph: Dict) -> float:
        """评估跨层级关系匹配度"""
        try:
            room_object_edges = hierarchical_graph.get('room_object_edges', [])
            if not room_object_edges:
                return 0.5
            
            # 分析目标是否涉及特定房间-对象关系
            goal_cross_level_score = 0.0
            goal_nodes = goal_subgraph.get('nodes', [])
            
            for node in goal_nodes:
                if isinstance(node, dict):
                    node_name = node.get('name', '').lower()
                    # 检查是否存在匹配的房间-对象关系
                    for edge in room_object_edges:
                        if (node_name in edge.get('target', '').lower() or 
                            node_name in edge.get('source', '').lower()):
                            goal_cross_level_score += 0.2
            
            return min(1.0, goal_cross_level_score)
        except:
            return 0.5
    
    def _evaluate_vlm_scene_semantic_alignment(self, vlm_objects: Dict, hierarchical_graph: Dict) -> float:
        """评估VLM与场景图的语义对齐度"""
        try:
            # 提取VLM对象
            vlm_object_names = set()
            if isinstance(vlm_objects, dict):
                for direction_data in vlm_objects.values():
                    if isinstance(direction_data, dict):
                        objects = direction_data.get('Objects', direction_data.get('objects', []))
                        for obj in objects:
                            obj_name = obj if isinstance(obj, str) else obj.get('name', str(obj))
                            vlm_object_names.add(obj_name.lower())
            
            # 提取场景图对象
            scene_object_names = set()
            for obj_node in hierarchical_graph.get('object_nodes', []):
                scene_object_names.add(obj_node.get('name', '').lower())
            
            # 计算语义对齐度
            if not vlm_object_names and not scene_object_names:
                return 1.0
            
            overlap = len(vlm_object_names.intersection(scene_object_names))
            union = len(vlm_object_names.union(scene_object_names))
            
            return overlap / max(union, 1)
        except:
            return 0.5
    
    def _calculate_hierarchical_semantic_richness(self, hierarchical_graph: Dict) -> float:
        """计算分层场景图的语义丰富度"""
        try:
            room_count = len(hierarchical_graph.get('room_nodes', {}))
            object_count = len(hierarchical_graph.get('object_nodes', []))
            room_edges = len(hierarchical_graph.get('room_edges', []))
            object_edges = len(hierarchical_graph.get('object_edges', []))
            cross_edges = len(hierarchical_graph.get('room_object_edges', []))
            
            # 基于结构复杂度计算丰富度
            structure_score = min(1.0, (room_count * 0.3 + object_count * 0.1) / 10)
            connectivity_score = min(1.0, (room_edges + object_edges + cross_edges) / 20)
            
            return (structure_score + connectivity_score) / 2
        except:
            return 0.5
    
    def _evaluate_goal_semantic_relevance(self, hierarchical_graph: Dict, goal_subgraph: Dict) -> float:
        """评估场景图对目标的语义相关性"""
        try:
            goal_objects = set()
            goal_nodes = goal_subgraph.get('nodes', [])
            for node in goal_nodes:
                if isinstance(node, dict):
                    obj_name = node.get('name', node.get('id', ''))
                    if obj_name:
                        goal_objects.add(str(obj_name).lower())
            
            if not goal_objects:
                return 0.5
            
            # 检查场景图中相关对象的存在
            relevant_objects = 0
            for obj_node in hierarchical_graph.get('object_nodes', []):
                obj_name = obj_node.get('name', '').lower()
                if obj_name in goal_objects:
                    relevant_objects += 1
                # 检查语义相关性（同类对象）
                elif any(self._are_semantically_related(obj_name, goal_obj) for goal_obj in goal_objects):
                    relevant_objects += 0.5
            
            relevance_score = relevant_objects / len(goal_objects)
            return min(1.0, relevance_score)
        except:
            return 0.5
    
    def _are_semantically_related(self, obj1: str, obj2: str) -> bool:
        """判断两个对象是否语义相关"""
        # 简单的语义相关性判断
        semantic_groups = [
            ['chair', 'table', 'desk', 'sofa', 'couch'],
            ['bed', 'pillow', 'blanket', 'mattress'],
            ['kitchen', 'stove', 'refrigerator', 'sink', 'microwave'],
            ['bathroom', 'toilet', 'shower', 'bathtub', 'sink'],
            ['book', 'shelf', 'bookshelf', 'library'],
        ]
        
        for group in semantic_groups:
            if obj1 in group and obj2 in group:
                return True
        return False
    
    def _extract_room_relations_from_goal(self, goal_subgraph: Dict) -> List[str]:
        """从目标子图中提取房间关系"""
        try:
            relations = []
            edges = goal_subgraph.get('edges', [])
            for edge in edges:
                if isinstance(edge, dict):
                    relation = f"{edge.get('source', '')}-{edge.get('relation', '')}-{edge.get('target', '')}"
                    if any(room_word in relation.lower() for room_word in 
                           ['kitchen', 'bathroom', 'bedroom', 'living', 'dining']):
                        relations.append(relation.lower())
            return relations
        except:
            return []
    
    def _extract_object_relations_from_goal(self, goal_subgraph: Dict) -> List[str]:
        """从目标子图中提取对象关系"""
        try:
            relations = []
            edges = goal_subgraph.get('edges', [])
            for edge in edges:
                if isinstance(edge, dict):
                    relation = f"{edge.get('source', '')}-{edge.get('relation', '')}-{edge.get('target', '')}"
                    relations.append(relation.lower())
            return relations
        except:
            return []
    
    def _fallback_overlap_calculation(self, vlm_objects: Dict, goal_subgraph: Dict) -> Dict:
        """
        回退的简单重叠计算
        """
        return {
            'overlap_score': 0.5,
            'fallback_used': True,
            'vlm_target_match': 0.5,
            'scene_target_match': 0.0,
            'vlm_scene_consistency': 0.0,
            'spatial_relationship_score': 0.0,
            'semantic_richness': 0.0,
            'direction_relevance': 1.0,
            'matched_objects': [],
            'total_objects_current': 0,
            'total_objects_target': 0
        }
    
    def _map_objects_to_spatial_coordinates_with_voxel(self, direction, vlm_predictions, action_result):
        """
        增强版对象空间定位与映射：充分利用VLM深度信息和空间细节
        结合voxel_map、深度估计和空间关系来提高位置准确性
        """
        spatial_objects = []
        
        # 确保vlm_predictions是字典类型
        if not isinstance(vlm_predictions, dict):
            print(f"⚠️ Warning: vlm_predictions is not a dict in _map_objects_to_spatial_coordinates_with_voxel (type: {type(vlm_predictions)})")
            return spatial_objects
        
        # 检查指定方向的预测数据
        direction_key = str(direction)
        if direction_key in vlm_predictions:
            prediction_data = vlm_predictions[direction_key]
            
            if isinstance(prediction_data, dict):
                # 提取详细信息
                objects = prediction_data.get('Objects', prediction_data.get('objects', []))
                object_relationships = prediction_data.get('ObjectRelationships', [])
                room_type = prediction_data.get('Room', prediction_data.get('room_type', 'unknown'))
                score = prediction_data.get('Score', prediction_data.get('score', 0))
                path_info = prediction_data.get('Path', 'unknown')
                
                if isinstance(objects, dict):
                    # 增强版字典格式处理：充分利用位置描述中的深度信息
                    for obj_name, position_info in objects.items():
                        # 解析深度和空间关系信息
                        depth_info = self._extract_depth_from_description(position_info)
                        spatial_relations = self._extract_spatial_relations_for_object(
                            obj_name, object_relationships, position_info
                        )
                        
                        # 结合深度信息的增强坐标估算
                        enhanced_coordinates = self._estimate_object_coordinates_with_voxel(
                            direction, obj_name, position_info, len(objects), action_result,
                            depth_info=depth_info, spatial_relations=spatial_relations
                        )
                        
                        spatial_obj = {
                            'name': obj_name,
                            'direction': direction,
                            'room_type': room_type,
                            'confidence': score,
                            'position_description': position_info,
                            'coordinates': enhanced_coordinates['position'],
                            'depth_info': depth_info,
                            'spatial_relations': spatial_relations,
                            'voxel_validated': enhanced_coordinates['voxel_validated'],
                            'navigability_score': enhanced_coordinates['navigability_score'],
                            'effective_area': enhanced_coordinates['effective_area'],
                            'depth_confidence': enhanced_coordinates.get('depth_confidence', 0.5),
                            'spatial_precision': enhanced_coordinates.get('spatial_precision', 'medium')
                        }
                        spatial_objects.append(spatial_obj)
                        
                elif isinstance(objects, list):
                    # 增强版列表格式处理
                    for i, obj in enumerate(objects):
                        if isinstance(obj, (str, dict)):
                            obj_name = obj if isinstance(obj, str) else obj.get('name', str(obj))
                            
                            # 尝试从关系信息中推断位置
                            inferred_relations = self._extract_spatial_relations_for_object(
                                obj_name, object_relationships, ""
                            )
                            
                            # 结合关系信息的增强坐标估算
                            enhanced_coordinates = self._estimate_object_coordinates_with_voxel(
                                direction, obj_name, None, len(objects), action_result, 
                                fallback_index=i, spatial_relations=inferred_relations
                            )
                            
                            spatial_obj = {
                                'name': obj_name,
                                'direction': direction,
                                'room_type': room_type,
                                'confidence': score,
                                'spatial_index': i,
                                'spatial_relations': inferred_relations,
                                'coordinates': enhanced_coordinates['position'],
                                'voxel_validated': enhanced_coordinates['voxel_validated'],
                                'navigability_score': enhanced_coordinates['navigability_score'],
                                'effective_area': enhanced_coordinates['effective_area'],
                                'depth_confidence': enhanced_coordinates.get('depth_confidence', 0.3),
                                'spatial_precision': enhanced_coordinates.get('spatial_precision', 'low')
                            }
                            spatial_objects.append(spatial_obj)
                
                # 后处理：基于空间关系优化对象位置
                spatial_objects = self._refine_object_positions_by_relationships(spatial_objects, object_relationships)
                            
        print(f"   🗺️ 映射了 {len(spatial_objects)} 个空间对象 (深度增强+关系优化)")
        return spatial_objects
    
    def _extract_depth_from_description(self, position_info: str) -> Dict:
        """从位置描述中提取深度信息"""
        depth_info = {
            'depth_category': 'mid-ground',
            'estimated_distance': 2.5,
            'depth_confidence': 0.5
        }
        
        if 'foreground' in position_info.lower():
            depth_info['depth_category'] = 'foreground'
            depth_info['estimated_distance'] = 1.5
            depth_info['depth_confidence'] = 0.8
        elif 'background' in position_info.lower():
            depth_info['depth_category'] = 'background'
            depth_info['estimated_distance'] = 5.0
            depth_info['depth_confidence'] = 0.7
        elif 'mid-ground' in position_info.lower():
            depth_info['depth_category'] = 'mid-ground'
            depth_info['estimated_distance'] = 3.0
            depth_info['depth_confidence'] = 0.8
            
        return depth_info
    
    def _extract_spatial_relations_for_object(self, obj_name: str, relationships: List[str], position_info: str) -> List[Dict]:
        """提取对象的空间关系信息"""
        spatial_relations = []
        
        # 从ObjectRelationships中提取
        for relation in relationships:
            if obj_name.lower() in relation.lower():
                relation_parts = relation.split()
                for i, part in enumerate(relation_parts):
                    if part.lower() in ['beside', 'on', 'under', 'near', 'behind', 'in front of']:
                        if i > 0 and i < len(relation_parts) - 1:
                            related_obj = relation_parts[i+1] if relation_parts[i-1].lower() == obj_name.lower() else relation_parts[i-1]
                            spatial_relations.append({
                                'relation_type': part.lower(),
                                'related_object': related_obj,
                                'confidence': 0.8
                            })
        
        # 从位置描述中提取空间关系
        position_lower = position_info.lower()
        relation_keywords = ['beside', 'on', 'under', 'near', 'behind']
        for keyword in relation_keywords:
            if keyword in position_lower:
                # 简单提取相关对象
                parts = position_lower.split(keyword)
                if len(parts) > 1:
                    related_obj = parts[1].strip().split()[0]
                    spatial_relations.append({
                        'relation_type': keyword,
                        'related_object': related_obj,
                        'confidence': 0.6
                    })
        
        return spatial_relations
    
    def _refine_object_positions_by_relationships(self, spatial_objects: List[Dict], relationships: List[str]) -> List[Dict]:
        """基于空间关系优化对象位置"""
        # 构建对象名称到索引的映射
        obj_name_to_idx = {obj['name']: i for i, obj in enumerate(spatial_objects)}
        
        # 基于关系调整位置
        for relation in relationships:
            if 'beside' in relation.lower():
                # 处理并排关系，调整x坐标使其更接近
                parts = relation.lower().split('beside')
                if len(parts) == 2:
                    obj1_name = parts[0].strip()
                    obj2_name = parts[1].strip()
                    
                    if obj1_name in obj_name_to_idx and obj2_name in obj_name_to_idx:
                        idx1, idx2 = obj_name_to_idx[obj1_name], obj_name_to_idx[obj2_name]
                        obj1, obj2 = spatial_objects[idx1], spatial_objects[idx2]
                        
                        # 调整x坐标使其更接近
                        avg_x = (obj1['coordinates'][0] + obj2['coordinates'][0]) / 2
                        spatial_objects[idx1]['coordinates'] = (avg_x - 0.3, obj1['coordinates'][1])
                        spatial_objects[idx2]['coordinates'] = (avg_x + 0.3, obj2['coordinates'][1])
            
            elif 'on' in relation.lower():
                # 处理上下关系，调整y坐标
                parts = relation.lower().split(' on ')
                if len(parts) == 2:
                    obj1_name = parts[0].strip()
                    obj2_name = parts[1].strip()
                    
                    if obj1_name in obj_name_to_idx and obj2_name in obj_name_to_idx:
                        idx1, idx2 = obj_name_to_idx[obj1_name], obj_name_to_idx[obj2_name]
                        obj1, obj2 = spatial_objects[idx1], spatial_objects[idx2]
                        
                        # obj1在obj2上方
                        spatial_objects[idx1]['coordinates'] = (obj2['coordinates'][0], obj2['coordinates'][1] + 0.5)
        
        return spatial_objects
        
    def _estimate_object_coordinates_with_voxel(self, direction, obj_name, position_info, total_objects, action_result, fallback_index=0, depth_info=None, spatial_relations=None):
        """
        增强版对象坐标估算：充分利用深度信息和空间关系
        结合voxel_map进行精确定位
        """
        import math
        
        # 基础角度转换
        direction_rad = math.radians(direction)
        
        # 深度信息处理
        if depth_info:
            estimated_distance = depth_info['estimated_distance']
            depth_confidence = depth_info['depth_confidence']
        elif position_info:
            # 从位置描述推断深度
            if 'foreground' in position_info.lower():
                estimated_distance = 1.5
                depth_confidence = 0.8
            elif 'background' in position_info.lower():
                estimated_distance = 5.0
                depth_confidence = 0.7
            elif 'mid-ground' in position_info.lower():
                estimated_distance = 3.0
                depth_confidence = 0.8
            else:
                estimated_distance = 2.5
                depth_confidence = 0.5
        else:
            estimated_distance = 2.5
            depth_confidence = 0.3
        
        # 水平位置解析（更精确）
        horizontal_offset = 0.0
        spatial_precision = 'medium'
        
        if position_info:
            position_lower = position_info.lower()
            if 'far-left' in position_lower:
                horizontal_offset = -2.0
                spatial_precision = 'high'
            elif 'center-left' in position_lower:
                horizontal_offset = -1.0
                spatial_precision = 'high'
            elif 'center-right' in position_lower:
                horizontal_offset = 1.0
                spatial_precision = 'high'
            elif 'far-right' in position_lower:
                horizontal_offset = 2.0
                spatial_precision = 'high'
            elif 'left' in position_lower:
                horizontal_offset = -1.5
                spatial_precision = 'medium'
            elif 'right' in position_lower:
                horizontal_offset = 1.5
                spatial_precision = 'medium'
            elif 'center' in position_lower:
                horizontal_offset = 0.0
                spatial_precision = 'high'
            else:
                # 根据对象索引分布
                if total_objects > 1:
                    horizontal_offset = -1.5 + (3.0 * fallback_index / (total_objects - 1))
                spatial_precision = 'low'
        else:
            # 使用fallback_index进行均匀分布
            if total_objects > 1:
                horizontal_offset = -1.5 + (3.0 * fallback_index / (total_objects - 1))
            spatial_precision = 'low'
        
        # 计算基础坐标（考虑深度）
        base_x = estimated_distance * math.cos(direction_rad) + horizontal_offset * math.cos(direction_rad + math.pi/2)
        base_y = estimated_distance * math.sin(direction_rad) + horizontal_offset * math.sin(direction_rad + math.pi/2)
        
        # Voxel地图验证和优化
        voxel_validated = False
        navigability_score = 0.5
        effective_area = 1.0
        
        try:
            # 检查panoramic_mask和effective_mask（如果存在）
            angle_str = str(int(direction))
            if hasattr(self, 'panoramic_mask') and angle_str in self.panoramic_mask:
                panoramic_valid = self.panoramic_mask[angle_str]
                if hasattr(self, 'effective_mask') and angle_str in self.effective_mask:
                    effective_valid = self.effective_mask[angle_str]
                    effective_area = np.sum(effective_valid.astype(float)) / effective_valid.size if isinstance(effective_valid, np.ndarray) else 1.0
                    
                # 计算导航性评分
                if isinstance(panoramic_valid, np.ndarray):
                    navigability_score = np.mean(panoramic_valid.astype(float))
                    voxel_validated = True
                    
            # 如果有voxel_map，进一步校正位置
            if hasattr(self, 'voxel_map') and self.voxel_map is not None:
                # 多点采样验证（增强版）
                sample_points = [
                    (base_x, base_y),
                    (base_x + 0.3, base_y),
                    (base_x - 0.3, base_y),
                    (base_x, base_y + 0.3),
                    (base_x, base_y - 0.3)
                ]
                
                valid_points = []
                navigability_scores = []
                
                for px, py in sample_points:
                    if hasattr(self, '_global_to_grid') and hasattr(self, 'agent_state'):
                        agent_pos = getattr(self, 'agent_state', None)
                        if agent_pos and hasattr(agent_pos, 'position'):
                            global_coords = np.array([
                                agent_pos.position[0] + px,
                                agent_pos.position[1],
                                agent_pos.position[2] + py
                            ])
                            
                            grid_coords = self._global_to_grid(global_coords)
                            if (0 <= grid_coords[0] < self.voxel_map.shape[1] and 
                                0 <= grid_coords[1] < self.voxel_map.shape[0]):
                                voxel_color = self.voxel_map[grid_coords[1], grid_coords[0]]
                                if not np.array_equal(voxel_color, [0, 0, 0]):
                                    valid_points.append((px, py))
                                    # 计算可达性评分（基于颜色）
                                    color_score = np.mean(voxel_color) / 255.0 if np.max(voxel_color) > 0 else 0.5
                                    navigability_scores.append(color_score)
                
                if valid_points:
                    # 选择最佳可通行点
                    if navigability_scores:
                        best_idx = navigability_scores.index(max(navigability_scores))
                        base_x, base_y = valid_points[best_idx]
                        navigability_score = max(navigability_scores)
                    else:
                        base_x, base_y = valid_points[0]
                        navigability_score = 0.7
                    
                    voxel_validated = True
                    effective_area = len(valid_points) / len(sample_points)
                    
        except Exception as e:
            print(f"   ⚠️ Voxel校正失败: {e}")
        
        # 基于空间关系的位置微调
        if spatial_relations:
            for relation in spatial_relations:
                if relation['relation_type'] in ['beside', 'near']:
                    relation_confidence = relation.get('confidence', 0.5)
                    if relation_confidence > 0.7:
                        adjustment = 0.2 * relation_confidence
                        base_x += np.random.uniform(-adjustment, adjustment)
                        base_y += np.random.uniform(-adjustment, adjustment)
        
        # 应用深度置信度影响精度
        if depth_confidence > 0.7:
            spatial_precision = 'high' if spatial_precision != 'low' else 'medium'
        elif depth_confidence < 0.4:
            spatial_precision = 'low' if spatial_precision == 'high' else spatial_precision
        
        result = {
            'position': (base_x, base_y),
            'voxel_validated': voxel_validated,
            'navigability_score': navigability_score,
            'effective_area': effective_area,
            'depth_confidence': depth_confidence,
            'spatial_precision': spatial_precision,
            'estimated_distance': estimated_distance,
            'horizontal_offset': horizontal_offset
        }
        
        print(f"      📍 {obj_name}: ({base_x:.2f},{base_y:.2f}) "
              f"深度:{estimated_distance:.1f}m 精度:{spatial_precision} "
              f"voxel验证:{'✓' if voxel_validated else '✗'}")
        
        return result
        
    def _estimate_object_coordinates(self, direction, index, total_objects):
        """
        基于方向和索引估算对象在空间中的相对坐标
        """
        import math
        
        # 基于方向计算基础坐标
        angle_rad = math.radians(direction)
        base_distance = 2.0  # 假设对象距离为2米
        
        base_x = base_distance * math.cos(angle_rad)
        base_y = base_distance * math.sin(angle_rad)
        
        # 基于索引调整坐标（对象在该方向区域内的分布）
        if total_objects > 1:
            spread = 0.5  # 对象分布范围
            offset = (index / max(1, total_objects - 1) - 0.5) * spread
            # 垂直于视线方向的偏移
            perp_x = -spread * math.sin(angle_rad) * offset
            perp_y = spread * math.cos(angle_rad) * offset
            
            return (base_x + perp_x, base_y + perp_y)
        else:
            return (base_x, base_y)
            
    def _build_current_scene_graph(self, spatial_objects, direction):
        """
        增强版分层场景图构建：充分利用深度信息和空间关系
        以room为顶层节点，对象为子节点的高效架构，集成深度感知和关系推理
        """
        # 分层场景图结构（增强版）
        hierarchical_scene_graph = {
            'room_nodes': {},      # 房间节点：{room_type: room_data}
            'object_nodes': [],    # 对象节点列表
            'room_edges': [],      # 房间间边
            'object_edges': [],    # 对象间边
            'room_object_edges': [], # 房间-对象边
            'direction_focus': direction,
            'spatial_regions': {},  # 空间区域聚类（增强版）
            'depth_layers': {},    # 深度层次聚类
            'spatial_relationships': [],  # 显式空间关系
            'hierarchy_type': 'room_centric_enhanced'
        }
        
        # === 第一步：按房间类型和深度层次双重聚类对象 ===
        room_clusters = {}
        depth_clusters = {'foreground': [], 'mid-ground': [], 'background': []}
        
        for i, obj in enumerate(spatial_objects):
            room_type = obj.get('room_type', 'unknown')
            depth_info = obj.get('depth_info', {})
            depth_category = depth_info.get('depth_category', 'mid-ground')
            
            # 房间聚类
            if room_type not in room_clusters:
                room_clusters[room_type] = {
                    'objects': [],
                    'spatial_bounds': {'min_x': float('inf'), 'max_x': float('-inf'),
                                     'min_y': float('inf'), 'max_y': float('-inf')},
                    'depth_distribution': {'foreground': 0, 'mid-ground': 0, 'background': 0},
                    'total_confidence': 0.0,
                    'spatial_precision_score': 0.0,
                    'voxel_validated_count': 0,
                    'object_count': 0
                }
            
            # 更新空间边界和统计信息
            coords = obj['coordinates']
            bounds = room_clusters[room_type]['spatial_bounds']
            bounds['min_x'] = min(bounds['min_x'], coords[0])
            bounds['max_x'] = max(bounds['max_x'], coords[0])
            bounds['min_y'] = min(bounds['min_y'], coords[1])
            bounds['max_y'] = max(bounds['max_y'], coords[1])
            
            room_clusters[room_type]['objects'].append(obj)
            room_clusters[room_type]['depth_distribution'][depth_category] += 1
            room_clusters[room_type]['total_confidence'] += obj.get('confidence', 0.5)
            room_clusters[room_type]['object_count'] += 1
            
            # 空间精度评分累计
            spatial_precision = obj.get('spatial_precision', 'medium')
            precision_score = {'high': 1.0, 'medium': 0.6, 'low': 0.3}.get(spatial_precision, 0.5)
            room_clusters[room_type]['spatial_precision_score'] += precision_score
            
            if obj.get('voxel_validated', False):
                room_clusters[room_type]['voxel_validated_count'] += 1
            
            # 深度聚类
            if depth_category in depth_clusters:
                depth_clusters[depth_category].append(obj)
        
        hierarchical_scene_graph['depth_layers'] = depth_clusters
        
        # === 第二步：创建增强版房间节点 ===
        for room_type, cluster_data in room_clusters.items():
            room_id = f"room_{room_type}_{direction}"
            bounds = cluster_data['spatial_bounds']
            
            # 计算房间中心点和深度分布
            center_x = (bounds['min_x'] + bounds['max_x']) / 2
            center_y = (bounds['min_y'] + bounds['max_y']) / 2
            
            # 计算深度多样性（Shannon熵）
            depth_dist = cluster_data['depth_distribution']
            total_objs = sum(depth_dist.values())
            depth_diversity = 0.0
            if total_objs > 0:
                for count in depth_dist.values():
                    if count > 0:
                        p = count / total_objs
                        depth_diversity -= p * math.log2(p)
            
            room_node = {
                'id': room_id,
                'type': 'room',
                'room_type': room_type,
                'center_coordinates': (center_x, center_y),
                'spatial_bounds': bounds,
                'object_count': cluster_data['object_count'],
                'avg_confidence': cluster_data['total_confidence'] / max(cluster_data['object_count'], 1),
                'contained_objects': [obj['name'] for obj in cluster_data['objects']],
                'spatial_area': max(0.01, (bounds['max_x'] - bounds['min_x']) * (bounds['max_y'] - bounds['min_y'])),
                'depth_distribution': depth_dist,
                'depth_diversity': depth_diversity,
                'spatial_precision_avg': cluster_data['spatial_precision_score'] / max(cluster_data['object_count'], 1),
                'voxel_validation_rate': cluster_data['voxel_validated_count'] / max(cluster_data['object_count'], 1)
            }
            
            hierarchical_scene_graph['room_nodes'][room_type] = room_node
        
        # === 第三步：创建增强版对象节点（包含深度和关系信息） ===
        for i, obj in enumerate(spatial_objects):
            node_id = f"{obj['name']}_{i}"
            if 'spatial_index' in obj:
                node_id = f"{obj['name']}_{obj['spatial_index']}"
            
            object_node = {
                'id': node_id,
                'type': 'object',
                'name': obj['name'],
                'parent_room': obj.get('room_type', 'unknown'),
                'coordinates': obj['coordinates'],
                'confidence': obj.get('confidence', 0.5),
                'depth_info': obj.get('depth_info', {}),
                'spatial_relations': obj.get('spatial_relations', []),
                'spatial_precision': obj.get('spatial_precision', 'medium'),
                'voxel_validated': obj.get('voxel_validated', False),
                'navigability_score': obj.get('navigability_score', 0.5)
            }
            
            # 添加位置描述和空间区域
            if 'position_description' in obj:
                object_node['position_description'] = obj['position_description']
                object_node['spatial_region'] = self._classify_spatial_region(obj['position_description'])
            
            hierarchical_scene_graph['object_nodes'].append(object_node)
            
            # 创建房间-对象边（增强权重）
            edge_weight = obj.get('confidence', 0.5) * obj.get('navigability_score', 0.5)
            room_object_edge = {
                'from': f"room_{obj.get('room_type', 'unknown')}_{direction}",
                'to': node_id,
                'type': 'contains',
                'weight': edge_weight,
                'depth_layer': obj.get('depth_info', {}).get('depth_category', 'mid-ground')
            }
            hierarchical_scene_graph['room_object_edges'].append(room_object_edge)
        
        # === 第四步：基于深度感知的高效对象关系创建 ===
        spatial_regions = {}
        for obj_node in hierarchical_scene_graph['object_nodes']:
            region = obj_node.get('spatial_region', 'center')
            room = obj_node['parent_room']
            depth_layer = obj_node.get('depth_info', {}).get('depth_category', 'mid-ground')
            region_key = f"{room}_{region}_{depth_layer}"
            
            if region_key not in spatial_regions:
                spatial_regions[region_key] = []
            spatial_regions[region_key].append(obj_node)
        
        # 创建基于空间关系的对象边
        for i, obj1 in enumerate(hierarchical_scene_graph['object_nodes']):
            for j, obj2 in enumerate(hierarchical_scene_graph['object_nodes']):
                if i < j:
                    # 检查显式空间关系
                    explicit_relation = self._check_explicit_spatial_relation(obj1, obj2, spatial_objects)
                    
                    if explicit_relation:
                        object_edge = {
                            'from': obj1['id'],
                            'to': obj2['id'],
                            'type': explicit_relation['type'],
                            'weight': explicit_relation['confidence'],
                            'distance': self._calculate_spatial_distance(obj1['coordinates'], obj2['coordinates']),
                            'depth_compatible': self._check_depth_compatibility(obj1, obj2)
                        }
                        hierarchical_scene_graph['object_edges'].append(object_edge)
                        hierarchical_scene_graph['spatial_relationships'].append(explicit_relation)
                    
                    elif self._should_create_proximity_edge(obj1, obj2):
                        # 基于距离的邻近关系
                        distance = self._calculate_spatial_distance(obj1['coordinates'], obj2['coordinates'])
                        if distance < 2.0:  # 2米内认为邻近
                            proximity_edge = {
                                'from': obj1['id'],
                                'to': obj2['id'],
                                'type': 'near',
                                'weight': max(0.1, 1.0 - distance / 2.0),
                                'distance': distance,
                                'depth_compatible': self._check_depth_compatibility(obj1, obj2)
                            }
                            hierarchical_scene_graph['object_edges'].append(proximity_edge)
        
        hierarchical_scene_graph['spatial_regions'] = spatial_regions
        
        # === 第五步：增强版房间间关系（考虑深度分布） ===
        room_types = list(room_clusters.keys())
        for i, room1 in enumerate(room_types):
            for j, room2 in enumerate(room_types):
                if i < j:
                    room1_node = hierarchical_scene_graph['room_nodes'][room1]
                    room2_node = hierarchical_scene_graph['room_nodes'][room2]
                    
                    # 计算房间距离和连接性
                    room_distance = self._calculate_spatial_distance(
                        room1_node['center_coordinates'],
                        room2_node['center_coordinates']
                    )
                    
                    # 计算深度兼容性
                    depth_compatibility = self._calculate_room_depth_compatibility(room1_node, room2_node)
                    
                    # 如果房间相邻或有连接关系，创建边
                    if room_distance < 4.0 or self._check_room_connection(room1_node, room2_node):
                        edge_weight = (1.0 / (room_distance + 0.1)) * depth_compatibility
                        room_edge = {
                            'from': room1_node['id'],
                            'to': room2_node['id'],
                            'type': 'adjacent',
                            'weight': edge_weight,
                            'distance': room_distance,
                            'depth_compatibility': depth_compatibility
                        }
                        hierarchical_scene_graph['room_edges'].append(room_edge)
        
        total_nodes = len(hierarchical_scene_graph['room_nodes']) + len(hierarchical_scene_graph['object_nodes'])
        total_edges = (len(hierarchical_scene_graph['room_edges']) + 
                      len(hierarchical_scene_graph['object_edges']) + 
                      len(hierarchical_scene_graph['room_object_edges']))
        
        print(f"   🏠 增强版分层场景图: {len(hierarchical_scene_graph['room_nodes'])} 房间, "
              f"{len(hierarchical_scene_graph['object_nodes'])} 对象, {total_edges} 关系 "
              f"(深度层次: {len([objs for objs in depth_clusters.values() if objs])}层)")
        
        return hierarchical_scene_graph
    
    def _check_explicit_spatial_relation(self, obj1: Dict, obj2: Dict, spatial_objects: List[Dict]) -> Dict:
        """检查两个对象间的显式空间关系"""
        obj1_relations = obj1.get('spatial_relations', [])
        obj2_relations = obj2.get('spatial_relations', [])
        
        # 检查obj1是否与obj2有关系
        for relation in obj1_relations:
            if relation.get('related_object', '').lower() in obj2['name'].lower():
                return {
                    'type': relation.get('relation_type', 'near'),
                    'confidence': relation.get('confidence', 0.7),
                    'explicit': True
                }
        
        # 检查obj2是否与obj1有关系
        for relation in obj2_relations:
            if relation.get('related_object', '').lower() in obj1['name'].lower():
                return {
                    'type': relation.get('relation_type', 'near'),
                    'confidence': relation.get('confidence', 0.7),
                    'explicit': True
                }
        
        return None
    
    def _check_depth_compatibility(self, obj1: Dict, obj2: Dict) -> float:
        """检查两个对象的深度兼容性"""
        depth1 = obj1.get('depth_info', {}).get('depth_category', 'mid-ground')
        depth2 = obj2.get('depth_info', {}).get('depth_category', 'mid-ground')
        
        if depth1 == depth2:
            return 1.0  # 同一深度层
        elif (depth1 in ['foreground', 'mid-ground'] and depth2 in ['foreground', 'mid-ground']) or \
             (depth1 in ['mid-ground', 'background'] and depth2 in ['mid-ground', 'background']):
            return 0.8  # 相邻深度层
        else:
            return 0.5  # 不同深度层
    
    def _should_create_proximity_edge(self, obj1: Dict, obj2: Dict) -> bool:
        """判断是否应该创建邻近边"""
        # 基于置信度和voxel验证状态
        conf1 = obj1.get('confidence', 0.5)
        conf2 = obj2.get('confidence', 0.5)
        voxel1 = obj1.get('voxel_validated', False)
        voxel2 = obj2.get('voxel_validated', False)
        
        # 如果两个对象都有较高置信度或都经过voxel验证，更倾向于创建边
        if (conf1 > 0.7 and conf2 > 0.7) or (voxel1 and voxel2):
            return True
        
        # 同一房间的对象更容易有关系
        if obj1.get('parent_room') == obj2.get('parent_room'):
            return True
            
        return False
    
    def _calculate_room_depth_compatibility(self, room1: Dict, room2: Dict) -> float:
        """计算两个房间的深度分布兼容性"""
        dist1 = room1.get('depth_distribution', {})
        dist2 = room2.get('depth_distribution', {})
        
        # 计算深度分布的相似性（余弦相似度）
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.5
        
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for depth_cat in ['foreground', 'mid-ground', 'background']:
            v1 = dist1.get(depth_cat, 0) / total1
            v2 = dist2.get(depth_cat, 0) / total2
            dot_product += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
        
        return 0.5
        
    def _calculate_spatial_distance(self, coord1, coord2):
        """计算两个坐标点之间的欧几里得距离"""
        import math
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
    def _check_semantic_relationship(self, obj1_name, obj2_name):
        """检查两个对象之间的语义关系"""
        # 定义一些常见的语义关系组
        semantic_groups = [
            ['chair', 'table', 'desk'],
            ['bed', 'pillow', 'blanket', 'nightstand'],
            ['stove', 'oven', 'refrigerator', 'microwave'],
            ['toilet', 'sink', 'bathtub', 'shower'],
            ['sofa', 'couch', 'coffee table', 'tv', 'television'],
        ]
        
        obj1_lower = obj1_name.lower()
        obj2_lower = obj2_name.lower()
        
        for group in semantic_groups:
            if obj1_lower in group and obj2_lower in group:
                return True
        return False
        
    def _check_position_relationship(self, obj1, obj2):
        """检查两个对象之间的位置关系"""
        # 检查是否都有位置描述
        if 'position_description' not in obj1 or 'position_description' not in obj2:
            return False
            
        pos1 = obj1['position_description'].lower()
        pos2 = obj2['position_description'].lower()
        
        # 检查是否在相同的位置区域
        same_horizontal = any([
            'left' in pos1 and 'left' in pos2,
            'right' in pos1 and 'right' in pos2,
            'center' in pos1 and 'center' in pos2
        ])
        
        same_depth = any([
            'foreground' in pos1 and 'foreground' in pos2,
            'background' in pos1 and 'background' in pos2,
            'front' in pos1 and 'front' in pos2,
            'far' in pos1 and 'far' in pos2
        ])
        
        # 如果在相同的水平或深度区域，认为有位置关系
        return same_horizontal or same_depth
    
    def _classify_spatial_region(self, position_description: str) -> str:
        """根据位置描述分类空间区域"""
        if not position_description:
            return 'center'
        
        pos = position_description.lower()
        
        # 水平位置
        horizontal = 'center'
        if 'left' in pos:
            horizontal = 'left'
        elif 'right' in pos:
            horizontal = 'right'
            
        # 深度位置
        depth = 'middle'
        if 'foreground' in pos or 'front' in pos:
            depth = 'front'
        elif 'background' in pos or 'far' in pos:
            depth = 'back'
        
        return f"{horizontal}_{depth}"
    
    def _get_adjacent_regions(self, region_key: str, all_regions: list) -> list:
        """获取相邻空间区域"""
        adjacent = []
        try:
            room, region = region_key.split('_', 1)
            horizontal, depth = region.split('_')
            
            # 相邻水平区域
            adjacent_horizontal = []
            if horizontal == 'left':
                adjacent_horizontal = ['center']
            elif horizontal == 'center':
                adjacent_horizontal = ['left', 'right']
            elif horizontal == 'right':
                adjacent_horizontal = ['center']
            
            # 相邻深度区域
            adjacent_depth = []
            if depth == 'front':
                adjacent_depth = ['middle']
            elif depth == 'middle':
                adjacent_depth = ['front', 'back']
            elif depth == 'back':
                adjacent_depth = ['middle']
            
            # 生成相邻区域组合
            for h in adjacent_horizontal:
                for d in adjacent_depth:
                    adjacent_key = f"{room}_{h}_{d}"
                    if adjacent_key in all_regions:
                        adjacent.append(adjacent_key)
                        
        except ValueError:
            pass  # 无效的region_key格式
        
        return adjacent
    
    def _create_optimized_object_edge(self, obj1: Dict, obj2: Dict, scene_graph: Dict):
        """优化的对象边创建，避免重复计算"""
        # 检查是否已存在边
        existing_edge = any(
            (edge['from'] == obj1['id'] and edge['to'] == obj2['id']) or
            (edge['from'] == obj2['id'] and edge['to'] == obj1['id'])
            for edge in scene_graph['object_edges']
        )
        
        if existing_edge:
            return
        
        # 计算空间距离
        dist = self._calculate_spatial_distance(obj1['coordinates'], obj2['coordinates'])
        
        # 检查语义关系
        semantic_relation = self._check_semantic_relationship(obj1['name'], obj2['name'])
        
        # 检查位置描述关系
        position_relation = self._check_position_relationship(obj1, obj2)
        
        # 如果距离较近、有语义关系或位置关系，创建边
        if dist < 1.5 or semantic_relation or position_relation:
            edge_type = 'spatial' if dist < 1.5 else ('semantic' if semantic_relation else 'positional')
            
            edge = {
                'from': obj1['id'],
                'to': obj2['id'],
                'type': edge_type,
                'weight': 1.0 / (dist + 0.1),
                'distance': dist,
                'same_room': obj1.get('parent_room') == obj2.get('parent_room')
            }
            scene_graph['object_edges'].append(edge)
    
    def _check_room_connection(self, room1_data: Dict, room2_data: Dict) -> bool:
        """
        增强版房间连接检查：结合goal_subgraph的连接信息和空间距离
        
        Args:
            room1_data: 房间1的数据字典，包含name、spatial_region、position等
            room2_data: 房间2的数据字典，包含name、spatial_region、position等
        """
        room1_name = room1_data.get('name', '').lower()
        room2_name = room2_data.get('name', '').lower()
        
        if room1_name == room2_name:
            return False  # 同一房间不算连接
        
        # === 策略1: 利用goal_subgraph的hallway_navigation信息 ===
        if hasattr(self, '_current_goal_subgraph') and self._current_goal_subgraph:
            goal_subgraph = self._current_goal_subgraph
            hallway_nav = goal_subgraph.get('hallway_navigation', {})
            room_connections = hallway_nav.get('room_connections', {})
            
            # 检查goal_subgraph中是否定义了这两个房间的连接
            for conn_id, conn_info in room_connections.items():
                if isinstance(conn_info, dict):
                    from_room = conn_info.get('from', '').lower()
                    to_room = conn_info.get('to', '').lower()
                    
                    # 双向检查连接
                    if ((from_room == room1_name and to_room == room2_name) or
                        (from_room == room2_name and to_room == room1_name)):
                        return True
        
        # === 策略2: 空间区域相邻性检查 ===
        region1 = room1_data.get('spatial_region', 'unknown')
        region2 = room2_data.get('spatial_region', 'unknown')
        
        if region1 != 'unknown' and region2 != 'unknown':
            # 相邻空间区域更容易连接
            adjacent_regions = {
                'front': ['center', 'left', 'right'],
                'back': ['center', 'left', 'right'], 
                'left': ['center', 'front', 'back'],
                'right': ['center', 'front', 'back'],
                'center': ['front', 'back', 'left', 'right']
            }
            
            if region2 in adjacent_regions.get(region1, []):
                return True
        
        # === 策略3: 基于房间类型的逻辑连接 ===
        # 定义常见的房间连接关系
        room_connections = {
            'living_room': ['kitchen', 'dining_room', 'hallway', 'bedroom', 'entrance'],
            'kitchen': ['living_room', 'dining_room', 'hallway'],
            'dining_room': ['kitchen', 'living_room', 'hallway'],
            'bedroom': ['hallway', 'bathroom', 'living_room'],
            'bathroom': ['hallway', 'bedroom'],
            'hallway': ['bedroom', 'bathroom', 'living_room', 'kitchen', 'entrance'],
            'entrance': ['hallway', 'living_room'],
            'office': ['hallway', 'living_room'],
            'study': ['hallway', 'living_room']
        }
        
        # 检查预定义连接关系
        connected_rooms = room_connections.get(room1_name, [])
        if room2_name in connected_rooms:
            return True
        
        # === 策略4: 空间距离启发式 ===
        if 'position' in room1_data and 'position' in room2_data:
            try:
                pos1 = room1_data['position']
                pos2 = room2_data['position']
                
                # 简单的距离检查（如果房间位置很近，认为可能连接）
                if isinstance(pos1, (list, tuple)) and isinstance(pos2, (list, tuple)) and len(pos1) >= 2 and len(pos2) >= 2:
                    distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                    # 如果距离小于某个阈值，认为可能连接
                    if distance < 5.0:  # 可调参数
                        return True
            except (TypeError, IndexError):
                pass
        
        # 默认情况：如果都是主要房间类型，给予一定连接概率
        major_rooms = {'living_room', 'kitchen', 'bedroom', 'bathroom', 'hallway'}
        if room1_name in major_rooms and room2_name in major_rooms:
            return True
        
        return False
        
        
    def _perform_llm_spatial_reasoning(self, all_direction_data, goal, vlm_predictions, exploration_phase=None):
        """
        LLM Comprehensive Spatial Reasoning: Analyze all directions to recommend the best navigation choice
        
        Args:
            all_direction_data: Dictionary with direction keys and their spatial analysis data
            goal: Target object to find
            vlm_predictions: Complete VLM analysis results for all directions
            exploration_phase: Dict with exploration phase info and reasoning strategy
        
        Returns:
            Dict: Contains recommended_direction, confidence_score, and reasoning
        """
        try:
            # Build comprehensive spatial reasoning prompt
            reasoning_prompt = self._build_comprehensive_spatial_reasoning_prompt(
                all_direction_data, goal, vlm_predictions, exploration_phase
            )
            
            # Use LLM for multi-directional reasoning
            if hasattr(self, 'ReasonLLM') and self.ReasonLLM:
                response = self.ReasonLLM.call(reasoning_prompt)
                
                # Parse LLM response to get recommended direction and reasoning
                result = self._extract_llm_direction_recommendation(response)
                print(f"   � LLM recommends direction: {result.get('recommended_direction', 'unknown')} (confidence: {result.get('confidence_score', 0.0):.3f})")
                return result
            else:
                print("   ⚠️ LLM unavailable, using numerical fallback")
                # Fallback: choose direction with highest overlap score
                return self._fallback_direction_selection(all_direction_data)
                
        except Exception as e:
            print(f"   ❌ LLM reasoning error: {e}")
            return self._fallback_direction_selection(all_direction_data)
            
    def _build_comprehensive_spatial_reasoning_prompt(self, all_direction_data, goal, vlm_predictions, exploration_phase=None):
        """Build comprehensive spatial reasoning prompt aligned with VLM style (English)"""
        current_goal = goal.upper()
        
        # Extract goal subgraph information
        goal_subgraph = getattr(self, '_current_goal_subgraph', {})
        target_room_hierarchy = goal_subgraph.get('target_room_hierarchy', {})
        spatial_anchor_objects = goal_subgraph.get('spatial_anchor_objects', {})
        hallway_navigation = goal_subgraph.get('hallway_navigation', {})
        
        primary_room = target_room_hierarchy.get('primary_room', 'unknown')
        secondary_rooms = target_room_hierarchy.get('secondary_rooms', [])
        anchor_names = list(spatial_anchor_objects.keys()) if isinstance(spatial_anchor_objects, dict) else []
        
        # Add exploration phase information
        exploration_context = ""
        if exploration_phase:
            phase_name = exploration_phase.get('phase_name', 'STANDARD')
            reasoning_strategy = exploration_phase.get('reasoning_strategy', '')
            exploration_context = f"\nEXPLORATION PHASE: {phase_name}\nSTRATEGY: {reasoning_strategy}\n"
        
        # Build connections string similar to VLM prompt
        connections_str = []
        room_connections = hallway_navigation.get('room_connections', {})
        if isinstance(room_connections, dict):
            for conn_key, conn_data in room_connections.items():
                if isinstance(conn_data, dict):
                    desc = conn_data.get('description', conn_key)
                    priority = conn_data.get('priority', 'normal')
                    connections_str.append(f"{desc} ({priority} priority)")
                else:
                    connections_str.append(str(conn_data))
        
        # Build wayfinding objects list
        wayfinding = hallway_navigation.get('wayfinding_objects', [])
        if isinstance(wayfinding, set):
            wayfinding = list(wayfinding)
        
        # Build frontier priorities
        exploration_strategy = goal_subgraph.get('exploration_strategy', {})
        frontiers = exploration_strategy.get('frontier_priorities', [])
        if isinstance(frontiers, set):
            frontiers = list(frontiers)
        
        # Build comprehensive directional analysis using both VLM and spatial analysis data
        direction_summaries = []
        expected_directions = ['30', '90', '150', '210', '270', '330']
        
        # Ensure vlm_predictions is a dictionary
        if not isinstance(vlm_predictions, dict):
            print(f"⚠️ Warning: vlm_predictions is not a dict (type: {type(vlm_predictions)}), using empty dict")
            vlm_predictions = {}
        
        for direction in expected_directions:
            direction_str = str(direction)
            
            # Get VLM analysis data with safe access
            vlm_data = vlm_predictions.get(direction_str, {}) if isinstance(vlm_predictions, dict) else {}
            score = vlm_data.get('Score', 0) if isinstance(vlm_data, dict) else 0
            objects = vlm_data.get('Objects', {}) if isinstance(vlm_data, dict) else {}
            room_type = vlm_data.get('Room', 'unknown') if isinstance(vlm_data, dict) else 'unknown'
            path_quality = vlm_data.get('Path', 'unknown') if isinstance(vlm_data, dict) else 'unknown'
            explanation = vlm_data.get('Explanation', 'No analysis') if isinstance(vlm_data, dict) else 'No analysis'
            object_relationships = vlm_data.get('ObjectRelationships', []) if isinstance(vlm_data, dict) else []
            anchors = vlm_data.get('Anchors', []) if isinstance(vlm_data, dict) else []
            
            # Get spatial analysis data from all_direction_data
            spatial_data = all_direction_data.get(direction_str, {})
            base_score = spatial_data.get('base_score', 0.0)
            overlap_score = spatial_data.get('overlap_score', 0.0)
            spatial_objects = spatial_data.get('spatial_objects', [])
            current_graph = spatial_data.get('current_graph', {})
            
            # Extract enhanced spatial information
            num_spatial_objects = len(spatial_objects) if spatial_objects else 0
            room_nodes = len(current_graph.get('room_nodes', [])) if current_graph else 0
            object_edges = len(current_graph.get('object_edges', [])) if current_graph else 0
            spatial_regions = len(current_graph.get('spatial_regions', {})) if current_graph else 0
            
            # Build spatial object summary
            spatial_obj_summary = []
            if spatial_objects:
                for i, obj in enumerate(spatial_objects[:3]):  # Show top 3 spatial objects
                    obj_name = obj.get('name', f'object_{i}')
                    obj_conf = obj.get('confidence', 0.0)
                    obj_depth = obj.get('depth_info', {}).get('depth_category', 'unknown')
                    obj_room = obj.get('parent_room', 'unknown')
                    spatial_obj_summary.append(f"{obj_name} (conf:{obj_conf:.2f}, depth:{obj_depth}, room:{obj_room})")
            
            # Format VLM objects
            vlm_object_list = []
            for obj_name, obj_position in objects.items():
                vlm_object_list.append(f"{obj_name}: {obj_position}")
            
            # Create comprehensive summary integrating both VLM and spatial analysis
            direction_summaries.append(
                f"Direction {direction}° - VLM Analysis (Score: {score}/10, Room: {room_type}, Path: {path_quality}): "
                f"VLM Objects: {'; '.join(vlm_object_list) if vlm_object_list else 'none'}. "
                f"VLM Relationships: {', '.join(object_relationships) if object_relationships else 'none'}. "
                f"VLM Anchors: {', '.join(anchors) if anchors else 'none'}. "
                f"VLM Explanation: {explanation}. "
                f"|| Spatial Analysis (Base Score: {base_score:.2f}, Overlap: {overlap_score:.3f}): "
                f"Mapped Objects: {num_spatial_objects}, Scene Graph: {room_nodes} rooms + {object_edges} edges + {spatial_regions} regions. "
                f"Key Spatial Objects: {'; '.join(spatial_obj_summary) if spatial_obj_summary else 'none'}."
            )
        
        # Analyze overlap score distribution for adaptive reasoning strategy
        overlap_scores = []
        for direction_str in expected_directions:
            spatial_data = all_direction_data.get(direction_str, {})
            overlap_scores.append(spatial_data.get('overlap_score', 0.0))
        
        # Create exploration-phase-specific prompt in VLM style but for LLM reasoning
        phase_name = exploration_phase.get('phase_name', 'STANDARD') if exploration_phase else 'STANDARD'
        reasoning_strategy = exploration_phase.get('reasoning_strategy', '') if exploration_phase else ''
        
        # Build context header based on exploration phase
        if phase_name == 'TARGET_VERIFICATION':
            context_header = (
                f"NAVIGATION TASK: Find {current_goal} - HIGH CONFIDENCE (target likely nearby)\n"
                f"PRIMARY FOCUS: Semantic similarity analysis and target confirmation\n"
                f"Context: {primary_room} room, anchors: {', '.join(anchor_names[:3]) if anchor_names else 'none'}"
            )
        elif phase_name == 'FOCUSED_SEARCH':
            context_header = (
                f"NAVIGATION TASK: Find {current_goal} - FOCUSED SEARCH (moderate alignment found)\n"
                f"PRIMARY FOCUS: Balance semantic relevance with spatial navigation\n"
                f"Context: Target in {primary_room}, also check {', '.join(secondary_rooms[:2]) if secondary_rooms else 'adjacent areas'}, anchors: {', '.join(anchor_names[:3]) if anchor_names else 'none'}"
            )
        else:  # FRONTIER_EXPLORATION
            context_header = (
                f"NAVIGATION TASK: Find {current_goal} - EXPLORATION MODE (low similarity, search new areas)\n"
                f"PRIMARY FOCUS: Spatial navigation and room discovery\n"
                f"Context: Seek {primary_room} room, expected in {', '.join(secondary_rooms[:2]) if secondary_rooms else 'connected areas'}, watch for {', '.join(anchor_names[:3]) if anchor_names else 'related objects'}"
            )
        
        # Build streamlined directional analysis
        direction_summaries = []
        expected_directions = ['30', '90', '150', '210', '270', '330']
        
        for direction in expected_directions:
            direction_str = str(direction)
            
            # Get core data with safe access
            vlm_data = vlm_predictions.get(direction_str, {}) if isinstance(vlm_predictions, dict) else {}
            spatial_data = all_direction_data.get(direction_str, {})
            
            score = vlm_data.get('Score', 0) if isinstance(vlm_data, dict) else 0
            room_type = vlm_data.get('Room', 'unknown') if isinstance(vlm_data, dict) else 'unknown'
            path_quality = vlm_data.get('Path', 'unknown') if isinstance(vlm_data, dict) else 'unknown'
            objects = vlm_data.get('Objects', {}) if isinstance(vlm_data, dict) else {}
            anchors = vlm_data.get('DirectionalAnchors', vlm_data.get('Anchors', [])) if isinstance(vlm_data, dict) else []
            
            base_score = spatial_data.get('base_score', 0.0)
            overlap_score = spatial_data.get('overlap_score', 0.0)
            
            # Create concise object summary
            key_objects = list(objects.keys())[:3] if objects else []
            anchor_summary = ', '.join(anchors[:2]) if anchors else 'none'
            
            # Phase-specific summary format
            if phase_name == 'TARGET_VERIFICATION':
                direction_summaries.append(
                    f"{direction}°: Overlap={overlap_score:.3f}, VLM={score}/10 ({room_type}), Objects: {', '.join(key_objects) if key_objects else 'none'}, Anchors: {anchor_summary}"
                )
            elif phase_name == 'FOCUSED_SEARCH':
                direction_summaries.append(
                    f"{direction}°: Spatial={base_score:.2f}, Semantic={overlap_score:.3f}, VLM={score}/10 ({room_type}, {path_quality}), Key: {', '.join(key_objects[:2]) if key_objects else 'none'}"
                )
            else:  # FRONTIER_EXPLORATION
                direction_summaries.append(
                    f"{direction}°: VLM={score}/10 ({room_type}, {path_quality}), Spatial={base_score:.2f}, Objects: {', '.join(key_objects[:2]) if key_objects else 'none'}, Guides: {anchor_summary}"
                )
        
        # Build phase-specific reasoning instructions
        if phase_name == 'TARGET_VERIFICATION':
            reasoning_instructions = (
                f"STRATEGY: Focus on highest overlap direction (>0.7). Verify target presence.\n"
                f"1. Prioritize directions with overlap > 0.7 (strong semantic match)\n"
                f"2. Confirm VLM room matches expected ({primary_room})\n"
                f"3. Check for expected anchors: {', '.join(anchor_names[:3]) if anchor_names else 'related objects'}\n"
                f"4. Ensure path accessibility (avoid 'blocked', prefer 'clear')"
            )
        elif phase_name == 'FOCUSED_SEARCH':
            reasoning_instructions = (
                f"STRATEGY: Balance semantic relevance (0.4-0.7) with spatial navigation.\n"
                f"1. Compare top 2-3 overlap directions (0.4-0.7 range)\n"
                f"2. Favor VLM rooms matching {primary_room} or {', '.join(secondary_rooms[:2]) if secondary_rooms else 'connected areas'}\n"
                f"3. Consider spatial scores for navigation quality\n"
                f"4. Look for directional anchors pointing toward target areas"
            )
        else:  # FRONTIER_EXPLORATION  
            reasoning_instructions = (
                f"STRATEGY: Prioritize VLM analysis and spatial exploration (<0.3 overlap).\n"
                f"1. Rely on VLM room predictions (seek {primary_room})\n"
                f"2. Favor higher VLM scores (>6/10) with clear paths\n"
                f"3. Look for any anchor objects or related items\n"
                f"4. Apply spatial logic for room connectivity and exploration"
            )
        
        # Create streamlined prompt
        prompt = (
            f"{context_header}\n\n"
            f"DIRECTIONAL ANALYSIS:\n"
            f"{chr(10).join(direction_summaries)}\n\n"
            f"{reasoning_instructions}\n\n"
            f"Analyze each direction and provide:\n"
            f"RECOMMENDED_DIRECTION: [30, 90, 150, 210, 270, or 330]\n"
            f"CONFIDENCE_SCORE: [0.0 to 1.0]\n"
            f"REASONING_SUMMARY: [Concise explanation based on {phase_name} strategy]"
        )
        
        return prompt
        
    def _try_parse_json_response(self, llm_response):
        """
        Try to parse LLM response as JSON/dictionary format
        Returns parsed result dictionary or None if parsing fails
        """
        try:
            import json
            import re
            
            # Clean the response for JSON parsing
            cleaned_response = llm_response.strip()
            
            # Try direct JSON parsing first
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                try:
                    json_data = json.loads(cleaned_response)
                    return self._extract_from_json_structure(json_data)
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON block within the response
            json_patterns = [
                r'\{[^}]*".*?"[^}]*\}',  # Simple JSON pattern
                r'\{.*?\}',              # Broader JSON pattern
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, cleaned_response, re.DOTALL)
                for match in matches:
                    try:
                        json_data = json.loads(match)
                        result = self._extract_from_json_structure(json_data)
                        if result:  # Only return if we successfully extracted data
                            return result
                    except json.JSONDecodeError:
                        continue
            
            # Try to parse dictionary-like format without quotes
            dict_pattern = r'\{\s*([^}]+)\s*\}'
            dict_match = re.search(dict_pattern, cleaned_response, re.DOTALL)
            if dict_match:
                dict_content = dict_match.group(1)
                return self._parse_dict_like_content(dict_content)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ JSON parsing failed: {e}")
            return None
    
    def _extract_from_json_structure(self, json_data):
        """Extract navigation data from parsed JSON structure"""
        result = {
            'recommended_direction': None,
            'confidence_score': 0.5,
            'reasoning_summary': 'Extracted from JSON response',
            'direction_evaluations': {}
        }
        
        try:
            # Try multiple JSON structure patterns
            
            # Pattern 1: Direct keys
            if 'recommended_direction' in json_data:
                result['recommended_direction'] = int(json_data['recommended_direction'])
            elif 'direction' in json_data:
                result['recommended_direction'] = int(json_data['direction'])
            
            if 'confidence_score' in json_data:
                result['confidence_score'] = float(json_data['confidence_score'])
            elif 'confidence' in json_data:
                result['confidence_score'] = float(json_data['confidence'])
            elif 'confidence_level' in json_data:
                result['confidence_score'] = float(json_data['confidence_level'])
                
            if 'reasoning_summary' in json_data:
                result['reasoning_summary'] = str(json_data['reasoning_summary'])
            elif 'reasoning' in json_data:
                result['reasoning_summary'] = str(json_data['reasoning'])
            elif 'rationale' in json_data:
                result['reasoning_summary'] = str(json_data['rationale'])
            
            # Pattern 2: Nested in analysis_result
            if 'analysis_result' in json_data:
                nested = json_data['analysis_result']
                if 'recommended_direction' in nested:
                    result['recommended_direction'] = int(nested['recommended_direction'])
                if 'confidence' in nested:
                    result['confidence_score'] = float(nested['confidence'])
                if 'reasoning' in nested:
                    result['reasoning_summary'] = str(nested['reasoning'])
                if 'direction_evaluations' in nested:
                    result['direction_evaluations'] = nested['direction_evaluations']
            
            # Pattern 3: Nested in spatial_reasoning
            if 'spatial_reasoning' in json_data:
                spatial = json_data['spatial_reasoning']
                if 'primary_recommendation' in spatial:
                    primary = spatial['primary_recommendation']
                    if 'direction' in primary:
                        result['recommended_direction'] = int(primary['direction'])
                    if 'confidence_level' in primary:
                        result['confidence_score'] = float(primary['confidence_level'])
                    if 'rationale' in primary:
                        result['reasoning_summary'] = str(primary['rationale'])
                
                # Extract direction analysis if available
                if 'direction_analysis' in spatial:
                    analysis = spatial['direction_analysis']
                    for dir_str, dir_data in analysis.items():
                        if isinstance(dir_data, dict) and 'note' in dir_data:
                            result['direction_evaluations'][int(dir_str)] = dir_data['note']
            
            # Validate extracted direction
            if result['recommended_direction'] and result['recommended_direction'] not in [30, 90, 150, 210, 270, 330]:
                result['recommended_direction'] = None
                
            # Normalize confidence score
            if result['confidence_score'] > 1.0:
                result['confidence_score'] = result['confidence_score'] / 10.0
            result['confidence_score'] = min(max(result['confidence_score'], 0.0), 1.0)
            
            # Return result only if we got a valid direction
            if result['recommended_direction'] is not None:
                return result
            
        except Exception as e:
            print(f"   ⚠️ JSON structure extraction failed: {e}")
            
        return None
    
    def _parse_dict_like_content(self, dict_content):
        """Parse dictionary-like content without proper JSON formatting"""
        try:
            result = {
                'recommended_direction': None,
                'confidence_score': 0.5,
                'reasoning_summary': 'Extracted from dictionary-like format',
                'direction_evaluations': {}
            }
            
            # Look for key-value patterns
            key_value_patterns = [
                (r'"?recommended_direction"?\s*[:\=]\s*(\d+)', 'recommended_direction'),
                (r'"?direction"?\s*[:\=]\s*(\d+)', 'recommended_direction'),
                (r'"?confidence"?\s*[:\=]\s*([\d.]+)', 'confidence_score'),
                (r'"?reasoning"?\s*[:\=]\s*"?([^",]+)', 'reasoning_summary'),
            ]
            
            for pattern, key in key_value_patterns:
                match = re.search(pattern, dict_content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"')
                    if key == 'recommended_direction':
                        direction = int(value)
                        if direction in [30, 90, 150, 210, 270, 330]:
                            result['recommended_direction'] = direction
                    elif key == 'confidence_score':
                        confidence = float(value)
                        if confidence > 1.0:
                            confidence = confidence / 10.0
                        result['confidence_score'] = min(max(confidence, 0.0), 1.0)
                    elif key == 'reasoning_summary':
                        result['reasoning_summary'] = value
            
            return result if result['recommended_direction'] is not None else None
            
        except Exception as e:
            print(f"   ⚠️ Dictionary-like parsing failed: {e}")
            return None
    
    def _extract_llm_direction_recommendation(self, llm_response):
        """
        Streamlined extraction for simplified prompt format with robust error handling
        """
        result = {
            'recommended_direction': 90,  # Default to forward
            'confidence_score': 0.5,
            'reasoning_summary': 'Default reasoning',
            'direction_evaluations': {}
        }
        
        try:
            # Extract recommended direction - simplified patterns
            direction_patterns = [
                r'RECOMMENDED_DIRECTION:\s*(\d+)',
                r'recommended[_\s]direction[:\s]*(\d+)',
                r'direction[:\s]*(\d+)',
                r'choose[_\s](\d+)°?',
                r'go[_\s](\d+)°?'
            ]
            
            for pattern in direction_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    try:
                        direction = int(match.group(1))
                        if direction in [30, 90, 150, 210, 270, 330]:
                            result['recommended_direction'] = direction
                            break
                    except ValueError:
                        continue
            
            # Extract confidence score with robust error handling
            confidence_patterns = [
                r'CONFIDENCE_SCORE:\s*([\d.%]+)',
                r'confidence[:\s]*([\d.%]+)',
                r'score[:\s]*([\d.%]+)'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    try:
                        confidence_str = match.group(1).strip()
                        # Handle cases like "0.8" or "80%" or malformed strings
                        confidence_str = confidence_str.replace('%', '')
                        
                        # Remove any non-numeric characters except dots and spaces
                        confidence_str = re.sub(r'[^\d.]', '', confidence_str)
                        
                        # Handle empty string or just dots or trailing dots
                        if not confidence_str or confidence_str == '.' or confidence_str == '..' or confidence_str.endswith('.') and len(confidence_str.replace('.', '')) == 0:
                            continue
                            
                        # Handle trailing dots like "0."
                        if confidence_str.endswith('.') and len(confidence_str) > 1:
                            confidence_str = confidence_str[:-1]
                            
                        # Ensure only one decimal point
                        if '.' in confidence_str:
                            parts = confidence_str.split('.')
                            if len(parts) >= 2:
                                # Take first integer part and first decimal part
                                if parts[0] and parts[1]:
                                    confidence_str = f"{parts[0]}.{parts[1]}"
                                elif parts[0]:  # Only integer part
                                    confidence_str = parts[0]
                                else:
                                    continue  # Invalid format
                        
                        if confidence_str and confidence_str != '.':
                            confidence = float(confidence_str)
                            # Normalize if it's in percentage form (>1.0) but handle edge case like 1.5 properly
                            if confidence > 1.0 and confidence <= 100.0:
                                confidence = confidence / 100.0
                            elif confidence > 100.0:
                                confidence = 1.0  # Cap at maximum
                            result['confidence_score'] = max(0.0, min(1.0, confidence))
                            break
                    except (ValueError, IndexError) as e:
                        print(f"   ⚠️ Failed to parse confidence '{match.group(1)}': {e}")
                        continue
            
            # Extract reasoning summary
            reasoning_patterns = [
                r'REASONING_SUMMARY:\s*(.+?)(?:\n|$)',
                r'reasoning[:\s]*(.+?)(?:\n|$)',
                r'summary[:\s]*(.+?)(?:\n|$)',
                r'explanation[:\s]*(.+?)(?:\n|$)'
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    if len(reasoning) > 10:  # Only use if substantial content
                        result['reasoning_summary'] = reasoning[:200]  # Limit length
                        break
            
            print(f"   🎯 Extracted: Direction={result['recommended_direction']}°, Confidence={result['confidence_score']:.3f}")
            return result
            
        except Exception as e:
            print(f"   ❌ LLM extraction error: {e}, using defaults")
            return result
            
    def _fallback_direction_selection(self, all_direction_data):
        """
        Fallback when LLM recommendation fails
        """
        try:
            if not all_direction_data:
                return {
                    'recommended_direction': 90,
                    'confidence_score': 0.1,
                    'reasoning_summary': 'No direction data available',
                    'direction_evaluations': {}
                }
            
            # Select direction with highest combined score
            best_direction = 90
            best_score = 0.0
            
            for direction, data in all_direction_data.items():
                combined_score = (
                    data.get('novelty_score', 0.0) * 0.3 +
                    data.get('exploration_score', 0.0) * 0.3 +
                    data.get('goal_alignment_score', 0.0) * 0.4
                )
                if combined_score > best_score:
                    best_score = combined_score
                    best_direction = direction
            
            return {
                'recommended_direction': best_direction,
                'confidence_score': min(best_score, 0.8),  # Cap fallback confidence
                'reasoning_summary': f'Fallback selection: highest combined score ({best_score:.3f})',
                'direction_evaluations': all_direction_data
            }
            
        except Exception as e:
            return {
                'recommended_direction': 90,
                'confidence_score': 0.1,
                'reasoning_summary': f'Fallback selection failed: {str(e)}',
                'direction_evaluations': {}
            }
        
         
        raise ValueError('Prompt type must be goal, stopping, predicting, planning, or action')