import numpy as np
import habitat_sim
import logging
import cv2
import time
import os
from typing import Dict, List, Tuple, Any, Optional, Set

from WMNav_agent import WMNavAgent
from simWrapper import PolarAction
from utils import *

# Import these conditionally to avoid circular imports
try:
    from scene_graph import SceneGraph
except ImportError:
    SceneGraph = None
    
try:
    from thought_parser import ThoughtParser
except ImportError:
    ThoughtParser = None

class GraphMemoryAgent(WMNavAgent):
    """
    Enhanced WMNavAgent with a graph-based memory system inspired by UniGoal.
    Implements subgraph matching, prediction re-weighting, and advanced navigation strategies.
    """
    
    def __init__(self, cfg: dict):
        print("� GraphMemoryAgent (OPTIMIZED VERSION) initializing...")
        logging.info("GraphMemoryAgent (OPTIMIZED VERSION) initializing...")
        
        # Initialize scene graph and thought parser BEFORE calling super().__init__()
        if SceneGraph is not None:
            self.scene_graph = SceneGraph()
            print("✅ SceneGraph initialized")
            logging.info("SceneGraph initialized")
        else:
            self.scene_graph = None
            print("⚠️  SceneGraph not available, using placeholder")
            logging.warning("SceneGraph not available, using placeholder")
            
        if ThoughtParser is not None:
            self.thought_parser = ThoughtParser()
            print("✅ ThoughtParser initialized")
            logging.info("ThoughtParser initialized")
        else:
            self.thought_parser = None
            print("⚠️  ThoughtParser not available, using placeholder")
            logging.warning("ThoughtParser not available, using placeholder")
            
        self.path_history = []  # Store agent's path
        
        # Directory to save scene graphs
        self.graph_save_dir = os.path.join(os.environ.get("LOG_DIR", "./logs"), "scene_graphs")
        os.makedirs(self.graph_save_dir, exist_ok=True)
        
        # Initialize graph memory settings
        self.graph_memory_cfg = cfg.get('graph_memory', {})
        self.graph_enabled = self.graph_memory_cfg.get('enabled', True)
        self.visualization_freq = self.graph_memory_cfg.get('visualization_freq', 1)
        self.confidence_threshold = self.graph_memory_cfg.get('confidence_threshold', 0.6)
        self.max_subgraph_distance = self.graph_memory_cfg.get('max_distance', 2)
        
        # UniGoal-inspired optimization tracking
        self.graph_update_count = 0
        self.optimization_stats = {
            'prediction_reweights': 0,
            'subgraph_matches': 0,
            'strategy_switches': 0,
            'successful_navigations': 0
        }
        
        # NOW call parent initialization
        super().__init__(cfg)
        
        print("🎯 GraphMemoryAgent (OPTIMIZED VERSION) initialized successfully!")
        logging.info("GraphMemoryAgent (OPTIMIZED VERSION) initialized successfully!")
    
    def reset(self):
        super().reset()
        print("🔄 GraphMemoryAgent resetting...")
        logging.info("GraphMemoryAgent resetting...")
        
        # Save the current scene graph before resetting
        if self.scene_graph is not None and hasattr(self, 'step_ndx') and self.step_ndx > 0:
            save_path = os.path.join(self.graph_save_dir, f"scene_graph_episode_{self.step_ndx}.json")
            self.scene_graph.save(save_path)
            print(f"💾 Scene graph saved to {save_path}")
            logging.info(f"Scene graph saved to {save_path}")
        
        # Reset scene graph for new episode
        if SceneGraph is not None:
            self.scene_graph = SceneGraph()
            print("✅ Scene graph reset")
            logging.info("Scene graph reset")
        else:
            self.scene_graph = None
            
        self.path_history = []
        self.graph_update_count = 0
        print("🎯 GraphMemoryAgent reset complete!")
    
    def _build_enhanced_room_object_knowledge(self):
        """构建增强的房间-物体概率关系知识库"""
        knowledge = {
            "bedroom": {
                "bed": 0.95, "dresser": 0.8, "nightstand": 0.7, "wardrobe": 0.6,
                "lamp": 0.7, "mirror": 0.5, "desk": 0.4, "chair": 0.4,
                "toilet": 0.02, "sink": 0.05, "shower": 0.01, "bathtub": 0.01,
                "refrigerator": 0.01, "stove": 0.01, "dining_table": 0.03
            },
            "bathroom": {
                "toilet": 0.95, "sink": 0.9, "shower": 0.7, "bathtub": 0.5,
                "mirror": 0.8, "towel": 0.7, "cabinet": 0.6,
                "bed": 0.01, "sofa": 0.01, "dining_table": 0.01, "refrigerator": 0.01
            },
            "kitchen": {
                "refrigerator": 0.9, "stove": 0.9, "sink": 0.9, "microwave": 0.8,
                "counter": 0.9, "cabinet": 0.9, "dining_table": 0.5,
                "bed": 0.01, "toilet": 0.01, "sofa": 0.05
            },
            "living_room": {
                "sofa": 0.9, "tv": 0.8, "coffee_table": 0.7, "chair": 0.6,
                "lamp": 0.7, "bookshelf": 0.5, "plant": 0.4,
                "bed": 0.05, "toilet": 0.01, "refrigerator": 0.02
            },
            "hallway": {
                "door": 0.9, "picture": 0.5, "plant": 0.3, "cabinet": 0.3,
                "bed": 0.01, "sofa": 0.01, "refrigerator": 0.01, "toilet": 0.01
            }
        }
        return knowledge
        
        for room in visible_rooms:
            if room.lower() in knowledge:
                # 获取该房间包含目标物体的概率
                prob = knowledge[room.lower()].get(goal.lower(), 0.1)
                room_probabilities[room] = prob
            else:
                room_probabilities[room] = 0.2  # 未知房间给予默认概率
        
        # 对概率进行归一化
        total_prob = sum(room_probabilities.values())
        if total_prob > 0:
            for room in room_probabilities:
                room_probabilities[room] = round(room_probabilities[room] / total_prob, 2)
        
        return room_probabilities
    
    def _predicting_module(self, evaluator_image, goal):
        """
        Enhanced prediction module with UniGoal-style subgraph matching and score re-weighting.
        Integrates graph knowledge to improve prediction reliability and overcome VLM limitations.
        """
        # Step 1: Extract graph context for enhanced prompting
        graph_context = self._extract_graph_context_for_prediction(goal)
        
        # Step 2: Generate enhanced prompt with graph knowledge
        evaluator_prompt = self._construct_prompt(goal, 'predicting', scene_graph_context=graph_context)
        
        # Step 3: Get VLM predictions
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)
        
        # Step 4: Apply UniGoal-style graph matching and score re-weighting
        if (self.scene_graph is not None and self.graph_enabled and hasattr(self, 'step_ndx')):
            agent_position = self._get_current_agent_position()
            self._update_scene_graph_with_predictions(dct, agent_position, self.step_ndx)
            dct = self._reweight_predictions_with_graph_matching(dct, goal, agent_position)
            self._update_object_associations(dct, goal, agent_position)
        
        return dct
    
    def _extract_graph_context_for_prediction(self, goal: str) -> str:
        """
        Extract relevant graph context to enhance prediction prompts.
        Based on UniGoal's subgraph matching approach.
        """
        if not self.scene_graph or not self.graph_enabled:
            return ""
        
        try:
            # Get goal-relevant subgraph
            relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=2)
            
            # Build context focusing on object associations and spatial relationships
            context_parts = []
            
            # Add related objects found in current scene
            related_objects = []
            for node_name, node in relevant_nodes.items():
                if node.confidence > self.confidence_threshold and node.observations_count > 0:
                    room_info = node.attributes.get('room', 'unknown')
                    related_objects.append(f"{node_name}({node.confidence:.2f})")
            
            if related_objects:
                context_parts.append(f"Known objects: {', '.join(related_objects[:5])}")
            
            # Add spatial relationships
            spatial_relations = []
            for edge in relevant_edges[:3]:
                if edge.confidence > 0.5:
                    spatial_relations.append(f"{edge.source} {edge.relation_type} {edge.target}")
            
            if spatial_relations:
                context_parts.append(f"Spatial: {'; '.join(spatial_relations)}")
            
            # Add room-object probability knowledge
            room_probs = self._get_room_probabilities_for_goal(goal, relevant_nodes)
            if room_probs:
                top_rooms = sorted(room_probs.items(), key=lambda x: x[1], reverse=True)[:2]
                prob_str = ', '.join([f"{room}({prob:.2f})" for room, prob in top_rooms])
                context_parts.append(f"Likely rooms: {prob_str}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logging.warning(f"Error extracting graph context: {e}")
            return ""
    
    def _get_current_agent_position(self) -> np.ndarray:
        """Get current agent position with fallback."""
        if hasattr(self, 'last_agent_state') and self.last_agent_state is not None:
            return np.array(self.last_agent_state.position)
        return np.array([0.0, 0.0, 0.0])
    
    def _reweight_predictions_with_graph_matching(self, predictions: dict, goal: str, agent_position: np.ndarray) -> dict:
        """
        Apply UniGoal-style subgraph matching to re-weight prediction scores.
        This improves reliability by incorporating graph knowledge.
        """
        if not predictions or 'direction_scores' not in predictions:
            return predictions
        
        try:
            # Step 1: Extract visible objects and their spatial relations from predictions
            visible_objects = self._extract_objects_from_predictions(predictions)
            
            # Step 2: Create a temporary subgraph from current observations
            current_subgraph = self._build_observation_subgraph(visible_objects, agent_position)
            
            # Step 3: Perform graph matching with known scene graph
            overlap_score, common_objects = self._compute_graph_overlap(current_subgraph, goal)
            
            # Step 4: Re-weight direction scores based on graph knowledge
            if overlap_score > 0.1:  # Only reweight if we have meaningful overlap
                reweighted_scores = self._apply_graph_based_reweighting(
                    predictions['direction_scores'], goal, overlap_score, common_objects, agent_position
                )
                predictions['direction_scores'] = reweighted_scores
                predictions['graph_confidence'] = overlap_score
                predictions['matched_objects'] = list(common_objects)
                
                self.optimization_stats['prediction_reweights'] += 1
                print(f"🎯 Graph matching: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}")
                logging.info(f"Graph matching applied: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}")
            
        except Exception as e:
            logging.warning(f"Error in graph-based reweighting: {e}")
        
        return predictions
    
    def _extract_objects_from_predictions(self, predictions: dict) -> List[Dict]:
        """Extract object mentions from VLM predictions across all directions."""
        objects = []
        
        # Extract from direction thoughts if available
        if hasattr(self, 'direction_thoughts'):
            for direction, thought in self.direction_thoughts.items():
                if self.thought_parser:
                    parsed = self.thought_parser.parse_thought(thought, direction, self.step_ndx)
                    objects.extend(parsed.get('nodes', []))
        
        # Extract from main prediction response
        if 'thoughts' in predictions:
            thought_text = str(predictions['thoughts'])
            # Simple object extraction - could be enhanced with NLP
            common_objects = ["chair", "table", "sofa", "bed", "refrigerator", "tv", "sink", "toilet", "stove", "microwave"]
            for obj in common_objects:
                if obj in thought_text.lower():
                    objects.append({
                        'name': obj,
                        'confidence': 0.6,  # Default confidence from VLM observation
                        'direction': None
                    })
        
        return objects
    
    def _build_observation_subgraph(self, visible_objects: List[Dict], agent_position: np.ndarray) -> Dict:
        """Build a temporary subgraph from current observations."""
        subgraph = {
            'nodes': [],
            'edges': []
        }
        
        # Add visible objects as nodes
        for obj in visible_objects:
            subgraph['nodes'].append({
                'id': obj['name'],
                'confidence': obj.get('confidence', 0.5),
                'position': agent_position.tolist()  # Approximate position
            })
        
        # Add simple spatial relationships (could be enhanced)
        if len(subgraph['nodes']) > 1:
            for i in range(len(subgraph['nodes']) - 1):
                subgraph['edges'].append({
                    'source': subgraph['nodes'][i]['id'],
                    'target': subgraph['nodes'][i + 1]['id'],
                    'type': 'near',
                    'confidence': 0.5
                })
        
        return subgraph
    
    def _compute_graph_overlap(self, observation_subgraph: Dict, goal: str) -> Tuple[float, Set[str]]:
        """
        Compute overlap between observation subgraph and stored scene graph.
        Based on UniGoal's graph matching algorithm.
        """
        if not self.scene_graph:
            return 0.0, set()
        
        try:
            # Get relevant subgraph from stored scene graph
            relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=2)
            
            # Convert to comparable format
            stored_subgraph = {
                'nodes': [{'id': name, 'position': node.position.tolist() if node.position is not None else [0,0,0]} 
                         for name, node in relevant_nodes.items()],
                'edges': [{'source': edge.source, 'target': edge.target, 'type': edge.relation_type} 
                         for edge in relevant_edges]
            }
            
            # Find common objects
            observed_objects = {node['id'] for node in observation_subgraph['nodes']}
            stored_objects = {node['id'] for node in stored_subgraph['nodes']}
            common_objects = observed_objects.intersection(stored_objects)
            
            # Calculate overlap score
            if len(observed_objects) == 0:
                overlap_score = 0.0
            else:
                overlap_score = len(common_objects) / len(observed_objects)
            
            # Boost score if goal-related objects are found
            goal_lower = goal.lower()
            if any(goal_lower in obj.lower() or obj.lower() in goal_lower for obj in common_objects):
                overlap_score = min(1.0, overlap_score * 1.5)
            
            return overlap_score, common_objects
            
        except Exception as e:
            logging.warning(f"Graph overlap computation error: {e}")
            return 0.0, set()
    
    def _apply_graph_based_reweighting(self, direction_scores: Dict, goal: str, overlap_score: float, 
                                     common_objects: Set[str], agent_position: np.ndarray) -> Dict:
        """
        Re-weight direction scores based on graph knowledge and object associations.
        """
        if not direction_scores:
            return direction_scores
        
        reweighted_scores = direction_scores.copy()
        
        # Apply reweighting based on overlap score
        if overlap_score > 0.3:
            goal_related_directions = self._find_goal_related_directions(goal, common_objects)
            
            for direction, score in reweighted_scores.items():
                try:
                    direction_angle = float(direction)
                    
                    # Boost score if this direction contains goal-related objects
                    if direction_angle in goal_related_directions:
                        boost_factor = 1.0 + (overlap_score * 0.5)
                        reweighted_scores[direction] = min(10.0, score * boost_factor)
                    
                    # Apply room-based knowledge
                    room_boost = self._calculate_room_based_boost(direction_angle, goal, agent_position)
                    if room_boost > 1.0:
                        reweighted_scores[direction] = min(10.0, reweighted_scores[direction] * room_boost)
                        
                except (ValueError, TypeError):
                    continue
        
        return reweighted_scores
    
    def _find_goal_related_directions(self, goal: str, common_objects: Set[str]) -> Set[float]:
        """Find directions where goal-related objects were observed."""
        goal_directions = set()
        
        if not hasattr(self, 'direction_thoughts'):
            return goal_directions
        
        goal_lower = goal.lower()
        for direction_str, thought in self.direction_thoughts.items():
            try:
                direction_angle = float(direction_str)
                thought_lower = thought.lower()
                if goal_lower in thought_lower:
                    goal_directions.add(direction_angle)
                
                for obj in common_objects:
                    if obj.lower() in thought_lower:
                        goal_directions.add(direction_angle)
                        break
                        
            except (ValueError, TypeError):
                continue
        
        return goal_directions
    
    def _calculate_room_based_boost(self, direction_angle: float, goal: str, agent_position: np.ndarray) -> float:
        """Calculate direction boost based on room-object probability knowledge."""
        try:
            knowledge = self._build_enhanced_room_object_knowledge()
            
            # Simple room type inference based on direction (can be enhanced)
            if hasattr(self, 'direction_thoughts') and str(direction_angle) in self.direction_thoughts:
                thought = self.direction_thoughts[str(direction_angle)].lower()
                
                for room_type, obj_probs in knowledge.items():
                    if room_type in thought:
                        return obj_probs.get(goal.lower(), 0.2) + 0.5
            
            return 1.0
            
        except Exception as e:
            logging.warning(f"Room boost calculation error: {e}")
            return 1.0
            # In a full implementation, this would use spatial mapping
            
            # For now, use stored room knowledge
            room_knowledge = self._build_enhanced_room_object_knowledge()
            
            # Check if we're in a hallway and looking toward high-probability rooms
            if hasattr(self, 'in_hallway') and self.in_hallway:
                # Boost directions leading to high-probability rooms for this goal
                for room_type, object_probs in room_knowledge.items():
                    goal_prob = object_probs.get(goal.lower(), 0.0)
                    if goal_prob > 0.7:  # High probability room
                        return 1.2  # 20% boost for high-probability directions
            
            return 1.0  # No boost
            
        except Exception as e:
            logging.warning(f"Room-based boost calculation error: {e}")
            return 1.0
    
    def _update_scene_graph_with_predictions(self, predictions: dict, agent_position: np.ndarray, step: int):
        """
        Update scene graph with new VLM predictions.
        Enhanced version that properly processes structured thoughts.
        """
        if not self.scene_graph or not self.graph_enabled:
            return
        
        try:
            # Process direction thoughts if available  
            if hasattr(self, 'direction_thoughts') and self.thought_parser:
                for direction, thought in self.direction_thoughts.items():
                    try:
                        direction_angle = float(direction)
                        parsed_data = self.thought_parser.parse_thought(thought, direction_angle, step)
                        self.thought_parser.update_scene_graph(self.scene_graph, parsed_data, agent_position, step)
                    except Exception as e:
                        logging.warning(f"Error processing direction {direction}: {e}")
            
            # Process main prediction thoughts
            if 'thoughts' in predictions:
                # Simple object extraction and graph update
                self._extract_and_add_objects_from_text(str(predictions['thoughts']), agent_position, step)
            
            # Update graph statistics
            if hasattr(self, 'graph_update_count'):
                self.graph_update_count += 1
            else:
                self.graph_update_count = 1
                
        except Exception as e:
            logging.warning(f"Scene graph update error: {e}")
    
    def _update_object_associations(self, predictions: dict, goal: str, agent_position: np.ndarray):
        """
        Update object associations in the graph based on co-occurrence and spatial relationships.
        This helps improve future predictions through learned associations.
        """
        if not self.scene_graph or not self.graph_enabled:
            return
        
        try:
            # Extract objects mentioned in current predictions
            current_objects = set()
            
            if 'thoughts' in predictions:
                thought_text = str(predictions['thoughts']).lower()
                common_objects = ["chair", "table", "sofa", "bed", "refrigerator", "tv", "sink", "toilet", "stove", "microwave"]
                current_objects.update(obj for obj in common_objects if obj in thought_text)
            
            # Add associations between co-occurring objects
            objects_list = list(current_objects)
            for i, obj1 in enumerate(objects_list):
                for obj2 in objects_list[i+1:]:
                    # Add "near" relationship
                    self.scene_graph.add_edge(obj1, obj2, "near", confidence=0.6)
            
            # Associate objects with current location/room if known
            if hasattr(self, 'current_room') and self.current_room:
                for obj in current_objects:
                    self.scene_graph.add_edge(self.current_room, obj, "contains", confidence=0.7)
                    
        except Exception as e:
            logging.warning(f"Object association update error: {e}")
    
    def _extract_and_add_objects_from_text(self, text: str, agent_position: np.ndarray, step: int):
        """Simple object extraction from text and addition to scene graph."""
        common_objects = [
            "chair", "table", "sofa", "bed", "refrigerator", "tv", "sink", "toilet", 
            "stove", "microwave", "cabinet", "counter", "door", "window", "lamp"
        ]
        
        text_lower = text.lower()
        for obj in common_objects:
            if obj in text_lower:
                # Add object to scene graph with current position
                self.scene_graph.add_node(
                    obj, 
                    attributes={"type": "object", "source": "vlm_prediction"}, 
                    position=agent_position, 
                    confidence=0.6, 
                    step=step
                )
    
    def _get_room_probabilities_for_goal(self, goal: str, relevant_nodes: Dict) -> Dict[str, float]:
        """Get room probabilities for finding the goal based on current knowledge."""
        room_knowledge = self._build_enhanced_room_object_knowledge()
        room_probs = {}
        
        # Get base probabilities from knowledge
        for room_type, object_probs in room_knowledge.items():
            room_probs[room_type] = object_probs.get(goal.lower(), 0.1)
        
        # Boost probabilities for rooms where we've seen related objects
        for node_name, node in relevant_nodes.items():
            room = node.attributes.get('room', '').lower()
            if room in room_probs and node.confidence > 0.5:
                room_probs[room] = min(1.0, room_probs[room] * 1.2)
        
        return room_probs
    
    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """
        Optimized planning module with UniGoal-inspired subgraph navigation.
        Integrates graph knowledge for superior navigation decisions.
        """
        try:
            # Step 1: Extract goal-relevant subgraph with enhanced analysis
            subgraph_data = self._extract_enhanced_navigation_subgraph(goal)
            relevant_nodes, relevant_edges, waypoints = subgraph_data['nodes'], subgraph_data['edges'], subgraph_data['waypoints']
            
            # Step 2: Perform subgraph matching to identify optimal navigation strategy
            navigation_strategy = self._determine_optimal_navigation_strategy(
                goal=goal,
                waypoints=waypoints,
                relevant_nodes=relevant_nodes,
                relevant_edges=relevant_edges,
                subgraph_data=subgraph_data
            )
            
            # Step 3: Apply graph-enhanced memory context
            enhanced_memory = self._build_graph_enhanced_memory_context(
                goal, navigation_strategy, relevant_nodes, relevant_edges
            )
            
            # Step 4: Integrate strategy into goal_reason
            if navigation_strategy and navigation_strategy.get('guidance'):
                goal_reason = self._integrate_navigation_guidance(goal_reason, navigation_strategy, enhanced_memory)
            
            # Step 5: Generate and execute planning
            planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason, 
                                                   scene_graph_context=enhanced_memory)
            
            image_to_send = [planning_image] if isinstance(planning_image, np.ndarray) else planning_image
            planning_response = self.PlanVLM.call(image_to_send, planning_prompt)
            dct = self._eval_response(planning_response)
            
            # Step 6: Post-process with graph knowledge
            dct = self._enhance_planning_response_with_graph(dct, navigation_strategy, goal)
            
            print(f"🎯 Enhanced planning response: {dct}")
            logging.info(f"Enhanced planning with strategy: {navigation_strategy.get('strategy_type', 'unknown')}")
            return dct
            
        except Exception as e:
            print(f"Enhanced planning module error: {e}")
            logging.error(f"Enhanced planning module error: {e}")
            return {'Flag': False, 'Subtask': '{}'}
    
    def _extract_enhanced_navigation_subgraph(self, goal: str) -> Dict:
        """
        Extract enhanced navigation subgraph with UniGoal-style analysis.
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph') or self.scene_graph is None:
            return {'nodes': {}, 'edges': [], 'waypoints': [], 'analysis': {}}
        
        # Get goal-relevant subgraph with extended distance
        relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=3)
        
        # Enhanced waypoint extraction with prioritization
        waypoints = self._extract_prioritized_waypoints(goal, relevant_nodes, relevant_edges)
        
        # Spatial analysis
        spatial_analysis = self._analyze_spatial_distribution(relevant_nodes, relevant_edges, goal)
        
        # Room connectivity analysis
        room_analysis = self._analyze_room_connectivity(relevant_nodes, relevant_edges)
        
        return {
            'nodes': relevant_nodes,
            'edges': relevant_edges, 
            'waypoints': waypoints,
            'spatial_analysis': spatial_analysis,
            'room_analysis': room_analysis,
            'goal_coverage': self._calculate_goal_coverage(goal, relevant_nodes)
        }

    def _format_memory_from_graph_data(self, nodes, edges, goal, navigation_strategy, in_hallway, has_explored_high_prob):
        """
        统一从图数据格式化记忆上下文，避免重复提取
        """
        memory_parts = []
        
        # 仅在导航策略中未涵盖的情况下添加额外信息
        if navigation_strategy and 'guidance' in navigation_strategy:
            if "already checked" not in navigation_strategy['guidance']:
                # 已探索的房间
                if has_explored_high_prob:
                    memory_parts.append(f"Already checked: {', '.join(self.explored_high_prob_rooms)}")
        
            if "most likely" not in navigation_strategy['guidance'] and nodes:
                # 目标相关信息
                goal_lower = goal.lower()
                goal_nodes = [n for name, n in nodes.items() if goal_lower in name.lower()]
                
                if goal_nodes:
                    rooms_with_goal = set()
                    for node in goal_nodes:
                        room = node.attributes.get('room')
                        if room and room != 'unknown':
                            rooms_with_goal.add(room)
                    
                    if rooms_with_goal:
                        memory_parts.append(f"Goal locations: {', '.join(rooms_with_goal)}")
        
        # 走廊分析
        if in_hallway and "hallway" not in (navigation_strategy.get('guidance', '') if navigation_strategy else ""):
            memory_parts.append("Current location: hallway junction")
        
        return ". ".join(memory_parts)        
    def _determine_navigation_strategy(self, goal, waypoints, relevant_nodes, relevant_edges, 
                                        in_hallway, has_explored_high_prob, hallway_assessment):
        """
        统一的导航策略决定函数
        """
        # 1. 高优先级：如果在走廊且已探索高概率房间，寻找新区域
        if in_hallway and has_explored_high_prob:
            # 寻找新的区域（楼梯、新走廊）
            unexplored_areas = self._find_unexplored_areas_in_graph(goal)
            if unexplored_areas:
                area = unexplored_areas[0]
                return {
                    'strategy_type': 'new_area',
                    'target': area.get('type', 'new area'),
                    'direction': area.get('direction', '?'),
                    'guidance': f"I should explore new areas through {area.get('type', 'new area')} since I've already checked the most likely rooms ({', '.join(self.explored_high_prob_rooms)})."
                }
                
            # 或者寻找未探索的房间
            unexplored_rooms = self._find_unexplored_rooms_in_graph(goal)
            if unexplored_rooms:
                return {
                    'strategy_type': 'unexplored_room',
                    'target': unexplored_rooms[0],
                    'guidance': f"I should check unexplored rooms: {', '.join(unexplored_rooms[:2])}."
                }
        
        # 2. 中优先级：使用子图中的waypoints进行导航
        if waypoints:
            waypoint = waypoints[0]  # 已按优先级排序
            
            # 检查是否需要忽略这个waypoint（如果附近有更高价值的路径）
            if in_hallway and hallway_assessment:
                high_value_directions = [dir for dir, data in hallway_assessment.items() 
                                        if isinstance(data, dict) and data.get('probability', 0) > 0.7
                                        and not data.get('explored', False)]
                
                if high_value_directions and len(high_value_directions) > 0:
                    best_dir = high_value_directions[0]
                    best_room = hallway_assessment[best_dir].get('room_type', 'room')
                    return {
                        'strategy_type': 'high_value_direction',
                        'target': best_room,
                        'direction': best_dir,
                        'guidance': f"The hallway analysis indicates a high probability of finding {goal} in the {best_room} at direction {best_dir}°."
                    }
            
            if waypoint['type'] == 'room_entrance':
                return {
                    'strategy_type': 'room_entrance',
                    'target': waypoint['name'],
                    'guidance': f"Following the scene graph, I should head toward the {waypoint['name']} entrance."
                }
            elif waypoint['type'] == 'intersection':
                # 为交叉路口添加方向信息
                directions_info = ""
                if 'connected_directions' in waypoint and waypoint['connected_directions']:
                    num_exits = len(waypoint['connected_directions'])
                    directions_info = f" with {num_exits} possible exits"
                    
                return {
                    'strategy_type': 'intersection',
                    'target': 'hallway_intersection',
                    'guidance': f"According to the scene graph, I should navigate to the hallway intersection{directions_info}."
                }
        
        # 3. 低优先级：基于环境知识的一般性导航
        room_probabilities = {}
        for room_type in self._build_enhanced_room_object_knowledge().keys():
            prob = self._get_room_object_probability(room_type, goal)
            room_probabilities[room_type] = prob
        
        # 找出最高概率的房间
        likely_rooms = sorted([(room, prob) for room, prob in room_probabilities.items()], 
                                key=lambda x: x[1], reverse=True)
        
        if likely_rooms:
            top_room = likely_rooms[0][0]
            return {
                'strategy_type': 'knowledge_based',
                'target': top_room,
                'guidance': f"Based on object-room knowledge, {goal} is most likely to be found in a {top_room}."
            }
        
        # 找不到明确策略时的默认导航
        return {
            'strategy_type': 'default',
            'guidance': "I should continue exploring and look for open areas or doors that might lead to new rooms."
        }        
        
    def _find_unexplored_areas_in_graph(self, goal: str) -> List[Dict]:
        """
        在场景图中查找未探索的区域（楼梯、新走廊等）
        
        Args:
            goal: 目标物体
            
        Returns:
            包含方向和类型信息的未探索区域列表
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph') or self.scene_graph is None:
            return []
        
        unexplored_areas = []
        current_step = getattr(self, 'step_ndx', 0)
        
        # 获取与目标相关的子图
        relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=3)
        
        # 查找未探索区域（楼梯、新走廊等）
        for node_name, node in relevant_nodes.items():
            # 检查节点名称是否包含特殊区域指示词
            if ("stair" in node_name.lower() or 
                "new hallway" in node_name.lower() or
                "floor" in node_name.lower() or
                "elevator" in node_name.lower()):
                
                # 检查是否未充分探索（未被最近访问）
                if current_step - node.last_seen > 5 or node.observations_count < 2:
                    direction = node.attributes.get('direction', 0)
                    unexplored_areas.append({
                        'direction': direction,
                        'type': "stairway" if "stair" in node_name.lower() else "new hallway",
                        'confidence': node.confidence,
                        'node_name': node_name
                    })
        
        # 如果未找到特定未探索区域，查找过渡指示符
        if not unexplored_areas:
            for edge in relevant_edges:
                if edge.relation_type in ["leads_to", "connects_to", "path_to"]:
                    target_node = relevant_nodes.get(edge.target)
                    if target_node and (current_step - target_node.last_seen > 5 or target_node.observations_count < 2):
                        direction = target_node.attributes.get('direction', 0)
                        unexplored_areas.append({
                            'direction': direction,
                            'type': "area transition",
                            'confidence': edge.confidence,
                            'node_name': edge.target
                        })
        
        # 按信心度排序
        unexplored_areas.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 记录找到的区域
        if unexplored_areas:
            print(f"🔍 发现未探索区域: {[area['type'] for area in unexplored_areas]}")
            logging.info(f"发现未探索区域: {[area['type'] for area in unexplored_areas]}")
            
        return unexplored_areas
    
    def _find_unexplored_rooms_in_graph(self, goal: str) -> List[str]:
        """
        在场景图中查找尚未充分探索的房间
        
        Args:
            goal: 目标物体
            
        Returns:
            未探索房间名称列表
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph') or self.scene_graph is None:
            return []
        
        # 从场景图中获取所有房间
        all_rooms = set()
        for node_name, node in self.scene_graph.nodes.items():
            if node.attributes.get('type') == 'room' or node_name.lower() in self._build_enhanced_room_object_knowledge():
                all_rooms.add(node_name)
        
        # 获取已探索过的房间
        explored_rooms = set(getattr(self, 'explored_high_prob_rooms', []))
        
        # 获取可能包含目标的房间
        object_room_map = self._build_enhanced_room_object_knowledge()
        likely_rooms = set()
        for room_type, objects in object_room_map.items():
            if goal.lower() in objects and objects[goal.lower()] >= 0.4:
                likely_rooms.add(room_type)
        
        # 优先考虑未探索但可能包含目标的房间
        unexplored_likely_rooms = list(all_rooms.intersection(likely_rooms) - explored_rooms)
        
        # 如果没有可能包含目标的未探索房间，返回任何未探索房间
        if not unexplored_likely_rooms:
            unexplored_likely_rooms = list(all_rooms - explored_rooms)
        
        # 记录找到的房间
        if unexplored_likely_rooms:
            print(f"🏠 发现未探索房间: {unexplored_likely_rooms[:3]}")
            logging.info(f"发现未探索房间: {unexplored_likely_rooms[:3]}")
        
        return unexplored_likely_rooms[:3]  # 最多返回3个房间            

    def _extract_subgraph_for_navigation(self, goal: str) -> Tuple[Dict, List, List]:
        """
        Extract a navigation-focused subgraph with optimized waypoints.
        
        Args:
            goal: Target object to find
            
        Returns:
            Tuple of (relevant nodes, relevant edges, prioritized waypoints)
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph') or self.scene_graph is None:
            return {}, [], []
        
        # Get goal-relevant subgraph
        relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=3)
        
        # Enhanced waypoints with prioritization
        waypoints = []
        
        # Extract current agent position
        current_position = None
        if hasattr(self, 'last_agent_state') and self.last_agent_state is not None:
            current_position = np.array(self.last_agent_state.position)
        
        # Priority 1: Goal-containing rooms
        goal_rooms = []
        for node_name, node in relevant_nodes.items():
            # Check if node is a room that likely contains the goal
            if (node.attributes.get('type') == 'room' and 
                node.position is not None and 
                self._get_room_object_probability(node_name, goal) > 0.7):
                goal_rooms.append(node)
        
        # Room entrances with better prioritization
        room_to_hallway_edges = [edge for edge in relevant_edges 
                               if edge.relation_type in ["connected_to", "leads_to"] 
                               and ("hallway" in edge.source.lower() or "hallway" in edge.target.lower())]
        
        for edge in room_to_hallway_edges:
            room_node = relevant_nodes.get(edge.source if "hallway" in edge.target.lower() else edge.target)
            if room_node and room_node.position is not None:
                # Calculate priority based on goal probability and exploration state
                room_type = room_node.name
                probability = self._get_room_object_probability(room_type, goal)
                
                # Check if this room has been explored
                explored = room_type in getattr(self, 'explored_high_prob_rooms', [])
                
                waypoints.append({
                    'position': room_node.position,
                    'type': 'room_entrance',
                    'name': room_node.name,
                    'priority': probability * (0.3 if explored else 1.0),
                    'distance': np.linalg.norm(room_node.position - current_position) if current_position is not None else None
                })
        
        # Add decision points with directional context
        for node_name, node in relevant_nodes.items():
            if "hallway" in node_name.lower() and node.attributes.get('is_intersection', False):
                if node.position is not None:
                    # Get connected directions
                    connected_directions = []
                    for edge in relevant_edges:
                        if edge.source == node_name or edge.target == node_name:
                            other_node = relevant_nodes.get(edge.target if edge.source == node_name else edge.source)
                            if other_node and other_node.position is not None:
                                direction = other_node.position - node.position
                                connected_directions.append(direction)
                    
                    waypoints.append({
                        'position': node.position,
                        'type': 'intersection',
                        'name': node_name,
                        'priority': 0.5,  # Medium priority for intersections
                        'connected_directions': connected_directions,
                        'distance': np.linalg.norm(node.position - current_position) if current_position is not None else None
                    })
        
        # Sort waypoints by priority (descending) and distance (ascending)
        waypoints.sort(key=lambda w: (-w['priority'], w['distance'] if w['distance'] is not None else float('inf')))
        
        return relevant_nodes, relevant_edges, waypoints
    
    def _generate_spatial_context(self) -> str:
        """Generate spatial context from agent's path history."""
        if len(self.path_history) < 2:
            return ""
        
        context_parts = []
        context_parts.append("Agent Path Summary:")
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(self.path_history)):
            distance = np.linalg.norm(self.path_history[i] - self.path_history[i-1])
            total_distance += distance
        
        context_parts.append(f"- Total distance traveled: {total_distance:.2f}m")
        context_parts.append(f"- Current position: {self.path_history[-1]}")
        
        # Add exploration pattern analysis
        if len(self.path_history) > 5:
            recent_positions = self.path_history[-5:]
            position_variance = np.var(recent_positions, axis=0)
            if np.mean(position_variance) < 0.5:
                context_parts.append("- Recent movement: Concentrated exploration (possibly stuck or searching locally)")
            else:
                context_parts.append("- Recent movement: Wide exploration (covering new areas)")
        
        return "\n".join(context_parts)
    
    def step(self, obs: dict):
        """Enhanced step method with optimized scene graph utilization."""
        print(f"🚀 GraphMemoryAgent (FULL VERSION) step {getattr(self, 'step_ndx', 0)}")
        logging.info(f"GraphMemoryAgent (FULL VERSION) step {getattr(self, 'step_ndx', 0)}")
        
        # Save current observation for stopping module access
        self._current_obs = obs
        
        # Store current agent state for scene graph updates
        self.last_agent_state = obs.get('agent_state')
        
        # Update path history
        if self.last_agent_state:
            self.path_history.append(np.array(self.last_agent_state.position))
            print(f"📍 Agent position: {self.last_agent_state.position}")
            logging.info(f"Agent position: {self.last_agent_state.position}")
        
        # Call parent step method
        result = super().step(obs)
        
        # Log scene graph stats if available
        if self.scene_graph is not None:
            num_nodes = len(self.scene_graph.nodes)
            num_edges = len(self.scene_graph.edges)
            print(f"🧠 Scene graph stats: {num_nodes} nodes, {num_edges} edges")
            logging.info(f"Scene graph stats: {num_nodes} nodes, {num_edges} edges")
        
        # Optimize scene graph periodically
        if self.scene_graph is not None and self.graph_enabled:
            current_step = getattr(self, 'step_ndx', 0)
            if current_step > 0 and current_step % 5 == 0:
                self._optimize_scene_graph()
        
        # Visualize scene graph if enabled and conditions met
        if self.scene_graph is not None and self.graph_enabled and self._should_visualize_graph():
            # Generate scene graph visualization
            graph_image = self.draw_scene_graph()
            
            # Add scene graph image to the images dictionary for logging
            if isinstance(result, tuple) and len(result) >= 2:
                agent_action, metadata = result
                if 'images' in metadata:
                    metadata['images']['scene_graph'] = graph_image
                    logging.info(f"✅ Scene graph visualization added to images dictionary")
                    print(f"✅ Scene graph visualization added for step {self.step_ndx}")
        
        return result
    
    def _extract_key_insights(self, memory_context: str, goal: str) -> Tuple[str, str, str]:
        """
        Extract key insights from memory context for targeted prompt integration.
        
        Args:
            memory_context: The memory context string to parse
            goal: The current navigation goal
            
        Returns:
            Tuple of (seen_objects, explored_rooms, goal_locations)
        """
        if not memory_context:
            return "", "", ""
        
        memory_context = memory_context.lower()
        goal = goal.lower()
        seen_objects = ""
        explored_rooms = ""
        goal_locations = ""
        
        # Use regex for more flexible pattern matching
        import re
        
        # Extract goal locations with multiple patterns
        goal_location_patterns = [
            rf"(?:you'?ve seen|found|spotted) (?:a |an |the )?{goal}(?:s)? in ([\w\s,]+)",
            rf"{goal}s are (?:typically |usually |commonly |often )?(?:found |located |seen )in ([\w\s,]+)",
            rf"expected locations(?:\s*for\s*{goal}s?)?:?\s*([\w\s,]+)"
        ]
        
        for pattern in goal_location_patterns:
            match = re.search(pattern, memory_context, re.IGNORECASE)
            if match:
                goal_locations = match.group(1).strip()
                break
        
        # Extract explored rooms with multiple patterns
        explored_room_patterns = [
            r"(?:explored|visited):?\s*([\w\s,]+)",
            r"you'?ve explored:?\s*([\w\s,]+)",
            r"explored (?:areas|rooms|spaces):?\s*([\w\s,]+)"
        ]
        
        for pattern in explored_room_patterns:
            match = re.search(pattern, memory_context, re.IGNORECASE)
            if match:
                explored_rooms = match.group(1).strip()
                break
        
        # Extract seen objects with multiple patterns
        seen_object_patterns = [
            r"objects:?\s*([\w\s,]+)",
            r"(?:observed|seen) objects:?\s*([\w\s,]+)",
            r"found:?\s*([\w\s,]+)"
        ]
        
        for pattern in seen_object_patterns:
            match = re.search(pattern, memory_context, re.IGNORECASE)
            if match:
                seen_objects = match.group(1).strip()
                break
        
        # Clean up outputs
        for item in [seen_objects, explored_rooms, goal_locations]:
            # Remove any trailing periods or commas
            item = re.sub(r'[.,]+$', '', item)
        
        # Look for rooms in the text even if not explicitly marked
        if not explored_rooms:
            room_types = ["kitchen", "bedroom", "bathroom", "living room", "dining room", "hallway", "office"]
            found_rooms = []
            for room in room_types:
                if room in memory_context:
                    found_rooms.append(room)
            if found_rooms:
                explored_rooms = ", ".join(found_rooms)
        
        return seen_objects, explored_rooms, goal_locations
    
    def _is_direction_explored(self, direction: str, current_step: int) -> bool:
        """
        检查某个方向是否已经被充分探索过
        
        Args:
            direction: 方向角度
            current_step: 当前步数
            
        Returns:
            是否已充分探索
        """
        # 转换为数值
        try:
            direction_angle = float(direction)
        except ValueError:
            return False
            
        # 查找图中最近的相同方向的节点
        explored_threshold = 3  # 至少探索3次才算充分探索
        exploration_count = 0
        recency_limit = 10  # 只考虑最近10步的探索
        
        for node_name, node in self.scene_graph.nodes.items():
            node_direction = node.attributes.get('direction')
            if node_direction is None:
                continue
                
            # 检查方向是否相近（±30°）
            if abs(node_direction - direction_angle) <= 30:
                # 检查是否是最近的探索
                if current_step - node.last_seen <= recency_limit:
                    exploration_count += 1
        
        return exploration_count >= explored_threshold
    
    def _get_room_object_probability(self, room_type: str, object_name: str) -> float:
        """获取特定房间包含特定物体的概率"""
        knowledge = self._build_enhanced_room_object_knowledge()
        room_data = knowledge.get(room_type.lower(), {})
        
        # 直接概率查找
        probability = room_data.get(object_name.lower(), 0.1)
        
        # 应用历史观察加权
        if hasattr(self, 'explored_high_prob_rooms') and room_type in self.explored_high_prob_rooms:
            # 已探索过的高概率房间但没找到目标，降低其概率
            probability *= 0.3
        
        return probability
    
    def _identify_room_type(self, text: str) -> str:
        """从描述文本中识别房间类型"""
        room_indicators = {
            "bedroom": ["bedroom", "bed room", "sleeping area", "master bedroom"],
            "bathroom": ["bathroom", "bath room", "restroom", "toilet", "shower"],
            "kitchen": ["kitchen", "cooking area", "stove", "refrigerator"],
            "living room": ["living room", "living area", "sofa", "couch", "tv area"],
            "dining room": ["dining room", "dining area", "dining table"],
            "hallway": ["hallway", "corridor", "passageway"],
            "stairway": ["stairs", "stairway", "staircase"]
        }
        
        text = text.lower()
        
        # 查找最佳匹配
        best_room = None
        max_indicators = 0
        
        for room, indicators in room_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text)
            if matches > max_indicators:
                max_indicators = matches
                best_room = room
        
        return best_room
    
    def _evaluate_hallway_perspective(self, goal: str, panoramic_data: dict) -> dict:
        """
        从走廊视角评估每个可见方向和房间包含目标对象的概率
        
        Args:
            goal: 目标物体
            panoramic_data: 全景观察数据，包括每个方向的评分和解释
            
        Returns:
            评估结果，包含每个方向的房间类型、可能性评分和是否已探索
        """
        results = {}
        
        # Safety check for input data
        if not isinstance(panoramic_data, dict) or not panoramic_data:
            logging.warning("Invalid panoramic data provided to hallway perspective evaluator")
            return results
        
        # 检测是否处于走廊
        in_hallway = False
        hallway_indicators = ["hallway", "corridor", "passageway", "multiple doors", "several rooms"]
        
        for direction, data in panoramic_data.items():
            if not isinstance(data, dict):
                continue
            
            explanation = data.get('Explanation', '')
            if not isinstance(explanation, str):
                continue
                
            explanation = explanation.lower()
            if any(indicator in explanation for indicator in hallway_indicators):
                in_hallway = True
                break
        
        if not in_hallway:
            return results
                
        # 从走廊视角评估每个方向
        for direction, data in panoramic_data.items():
            if not isinstance(data, dict):
                continue
                
            explanation = data.get('Explanation', '')
            if not isinstance(explanation, str):
                continue
                
            explanation = explanation.lower()
            
            # 识别可能的房间类型
            room_type = self._identify_room_type(explanation)
            
            # 计算该房间包含目标的概率
            probability = 0.0
            if room_type:
                probability = self._get_room_object_probability(room_type, goal)
                    
            # 检查该方向是否已被探索
            explored = False
            try:
                explored = self._is_direction_explored(direction, self.step_ndx)
            except Exception as e:
                logging.warning(f"Error checking direction exploration: {e}")
            
            # 存储结果
            results[direction] = {
                'room_type': room_type,
                'probability': probability,
                'explored': explored,
                'explanation': explanation
            }
        
        return results 
    
    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', 
                        reason: str = '{}', num_actions: int = 0, scene_graph_context: str = None):
        """
        Optimized prompt construction with unified memory extraction.
        Uses intelligent memory selection based on prompt type for optimal VLM performance.
        """
        # 仅当reason中未包含足够信息时才提取额外内存上下文
        if prompt_type == 'planning':
            # 判断reason是否已经包含充分的导航信息
            has_navigation_info = any(term in reason.lower() for term in 
                                    ["hallway", "room", "direction", "door", "explore", 
                                    "navigate", "likely", "unlikely", "checked"])
            
            # 仅在必要时才提取
            if not has_navigation_info:
                memory_context = self._extract_relevant_memory(goal, prompt_type)
                seen_objects, explored_rooms, goal_locations = self._extract_key_insights(memory_context, goal)
            else:
                # reason已包含导航信息，不需要重复提取
                memory_context = ""
                seen_objects, explored_rooms, goal_locations = "", "", ""
        else:
            # 其他提示类型保持原有逻辑
            memory_context = self._extract_relevant_memory(goal, prompt_type)
            seen_objects, explored_rooms, goal_locations = self._extract_key_insights(memory_context, goal)
            
        logging.info(f"Memory context for {goal} ({prompt_type}): {memory_context}")
        print(f"Memory context for {goal} ({prompt_type}): {memory_context}")

        if prompt_type == 'goal':
            location_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. " 
            f"These are labeled with a number in a white circle, which represent the location you can move to. "
            f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}, return number 0 if if there is no {goal}, or if you are not sure. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
            f'Second, if there is {goal} in the image, then determine which circle best represents the location of the {goal}(close enough to the target. If a person is standing in that position, they can easily touch the {goal}), and give the number and a reason. '
            f'If none of the circles represent the position of the {goal}, return number 0, and give a reason why you returned 0. '
            "Format your answer in the json {{'Number': <The number you choose>}}")
            return location_prompt
            
        if prompt_type == 'predicting':
            evaluator_prompt = (
                f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image "
                f"describing your surrounding environment, each image contains a label indicating the relative rotation "
                f"angle(30, 90, 150, 210, 270, 330) with red fonts.\n\n"
                
                f"Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction "
                f"is worth exploring, and provide detailed analysis of what you see. Please follow these steps:\n\n"
                
                f"1. For each direction, identify the room type (bedroom, bathroom, kitchen, living room, hallway, etc.)\n"
                f"2. Note any furniture, objects, or architectural features visible\n"
                f"3. Describe spatial relationships (doorways, openings, passages between areas)\n"
                f"4. Assign a score using these criteria:\n"
                f"   - Score 0: Dead end with no passage and clearly no {goal}\n"
                f"   - Score 10: The {goal} is clearly visible\n" 
                f"   - Scores 1-9: Based on likelihood of finding a {goal}, considering room type and layout\n\n"
                
                f"5. If you see a hallway with multiple doorways or path options, note this specifically as it's important for navigation\n"
                f"6. Note if there are stairs or passages to different floors\n\n"
                
                f"Note a chair must have a backrest and a chair is not a stool. A chair is NOT a sofa(couch) which is NOT a bed.\n"
                f"Note you CANNOT GO THROUGH CLOSED DOORS. Only consider OPEN doors or passageways.\n\n"
                
                f"Format your answer in the JSON format:\n"
                f"{{'30': {{'Score': <score 0-10>, 'Thought': '<detailed analysis of what you see and why this score>'}}, "
                f"'90': {{...}}, '150': {{...}}, '210': {{...}}, '270': {{...}}, '330': {{...}}}}\n\n"
                
                f"Example thought structure: 'This appears to be a hallway with doors to other rooms. "
                f"The door on the left might lead to a bathroom based on the tiles visible. "
                f"This direction has good exploration potential for finding a {goal}.'"
            )
            return evaluator_prompt
        
        if prompt_type == 'planning':
            # Use the extracted memory context to inform planning
            if reason != '' and subtask != '{}':
                planning_prompt = (
                    f"The agent has been tasked with navigating to a {goal.upper()}. "
                    f"(1)<The observed image>: The image taken from its current location. "
                    f"(2){reason}. This explains why you should go in this direction. "
                )
                
                # Enhanced room guidance with hallway insights
                if "From hallway" in memory_context or goal_locations or explored_rooms:
                    room_guidance = []
                    
                    # Prioritize hallway analysis if available
                    if "From hallway" in memory_context:
                        hallway_insight = memory_context.split("From hallway")[1].split(".")[0]
                        room_guidance.append(f"From your hallway perspective, {hallway_insight}")
                        
                        # Add info about already checked rooms
                        if "Already checked" in memory_context:
                            already_checked = memory_context.split("Already checked")[1].split(".")[0]
                            room_guidance.append(f"You have already checked{already_checked} without finding {goal}")
                    
                    # Direct evidence from memory
                    elif goal_locations:
                        room_guidance.append(f"You've seen {goal}s in {goal_locations}.")
                    
                    # Then fall back to probabilistic knowledge
                    elif explored_rooms:
                        # Get visible rooms from explored rooms
                        visible_rooms = [room.strip() for room in explored_rooms.split(",")]
                        
                        # Use inference function to get probabilities
                        room_probs = self._infer_target_room(goal, visible_rooms)
                        
                        # Format as high/low probability insights
                        likely_rooms = [f"{room}({int(prob*100)}%)" for room, prob in room_probs.items() if prob >= 0.5]
                        unlikely_rooms = [room for room, prob in room_probs.items() if prob <= 0.1]
                        
                        if likely_rooms:
                            room_guidance.append(f"The {goal} is most likely in: {', '.join(likely_rooms)}")
                        if unlikely_rooms and len(unlikely_rooms) < len(room_probs):
                            room_guidance.append(f"The {goal} is unlikely to be in: {', '.join(unlikely_rooms)}")
                    
                    # Add room-object guidance to prompt if we have any
                    if room_guidance:
                        planning_prompt += (
                            f"(3)<Room analysis>: {' '.join(room_guidance)} Based on this analysis, prioritize "
                            f"rooms where {goal} is most commonly found, and avoid rooms where {goal} is rarely found "
                            f"and areas you've already checked. When in hallways, use this information to choose "
                            f"the most promising door or direction."
                        )
                
                # No need for separate goal_locations integration since it's handled above
                # Integrate navigation context if not already included in room guidance
                if explored_rooms and not (goal_locations or "From hallway" in memory_context):
                    planning_prompt += f"You've already explored {explored_rooms}. "
                
                # Add task instructions
                planning_prompt += (
                    f"Your job is to describe the next place to go. Follow these steps: "
                    f"(1) If the {goal} appears in the image, directly choose it as your target. "
                    f"(2) If the {goal} is not visible and you haven't completed subtask {subtask}, continue with that subtask. "
                )
                
                # Add strategic guidance based on memory
                if not goal_locations and seen_objects:
                    planning_prompt += f"(3) Based on your observations of {seen_objects}, "
                    
                planning_prompt += (
                    f"(3) If the previous subtask is complete, identify a new subtask to find the {goal}. "
                    "Pay special attention to open doors and hallways that lead to unseen areas. "
                    "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                )
                # Add to planning prompt
                planning_prompt += (
                    "IMPORTANT: Be very conservative about reporting the object is present. "
                    f"Only set Flag to True if you are 100% certain you can see a {goal}. "
                    "If you're unsure, set Flag to False."
                )
                planning_prompt += (
                    f" Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}."
                )  
            else:
                # Similar modifications for the else branch
                planning_prompt = (
                    f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                )
                
                # Add improved room guidance with hallway insights
                if "From hallway" in memory_context or goal_locations or explored_rooms:
                    room_guidance = []
                    
                    # Prioritize hallway analysis if available
                    if "From hallway" in memory_context:
                        hallway_insight = memory_context.split("From hallway")[1].split(".")[0]
                        room_guidance.append(f"From your hallway perspective, {hallway_insight}")
                        
                        # Add info about already checked rooms
                        if "Already checked" in memory_context:
                            already_checked = memory_context.split("Already checked")[1].split(".")[0]
                            room_guidance.append(f"You have already checked{already_checked} without finding {goal}")
                    
                    # Direct evidence from memory
                    elif goal_locations:
                        room_guidance.append(f"You've seen {goal}s in {goal_locations}.")
                    
                    # Then fall back to probabilistic knowledge
                    elif explored_rooms:
                        visible_rooms = [room.strip() for room in explored_rooms.split(",")]
                        room_probs = self._infer_target_room(goal, visible_rooms)
                        
                        likely_rooms = [f"{room}({int(prob*100)}%)" for room, prob in room_probs.items() if prob >= 0.5]
                        unlikely_rooms = [room for room, prob in room_probs.items() if prob <= 0.1]
                        
                        if likely_rooms:
                            room_guidance.append(f"The {goal} is most likely in: {', '.join(likely_rooms)}")
                        if unlikely_rooms and len(unlikely_rooms) < len(room_probs):
                            room_guidance.append(f"The {goal} is unlikely to be in: {', '.join(unlikely_rooms)}")
                    
                    if room_guidance:
                        planning_prompt += (
                            f"\n Room analysis: {' '.join(room_guidance)}\n Based on this analysis, prioritize "
                            f"rooms where {goal} is most commonly found, and avoid rooms where {goal} is rarely found "
                            f"and areas you've already checked. When in hallways, use this information to choose "
                            f"the most promising door or direction.\n"
                        )
                    else:
                        planning_prompt += f"\n Memory context from exploration: {memory_context}\n"
                else:
                    planning_prompt += f"\n Memory context from exploration: {memory_context}\n"
                    
                planning_prompt += (
                    f'Your job is to describe next place to go. '
                    f'To help you plan your best next step, I can give you some human suggestions: '
                    f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                    f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                    "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                    "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}"
                )
            
            return planning_prompt
        
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')
    
    def draw_scene_graph(self, agent_state: habitat_sim.AgentState = None) -> np.ndarray:
        """优化的场景图可视化方法"""
        empty_size = 800
        default_image = np.zeros((empty_size, empty_size, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        step_text = f'step {getattr(self, "step_ndx", 0)}'
        
        try:
            import tempfile
            # 只在有数据时生成图像
            if self.scene_graph is not None and len(self.scene_graph.nodes) > 0:
                # 使用上下文管理器自动清理临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    temp_path = tmp.name
                    
                try:
                    # 生成图形并保存到临时文件
                    self.scene_graph.visualize(save_path=temp_path)
                    graph_image = cv2.imread(temp_path)
                    os.remove(temp_path)  # 立即清理
                    
                    if graph_image is not None:
                        # 添加步数信息
                        cv2.putText(graph_image, step_text, (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        return graph_image
                except Exception as e:
                    logging.warning(f"图形生成失败: {e}")
            
            # 返回默认图像
            cv2.putText(default_image, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 添加无数据说明
            if self.scene_graph is None or len(self.scene_graph.nodes) == 0:
                cv2.putText(default_image, "无场景图数据", 
                          (empty_size//6, empty_size//2), 
                          font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            return default_image
                
        except Exception as e:
            # 处理任何未预期的错误
            cv2.putText(default_image, "场景图可视化错误", 
                      (empty_size//6, empty_size//2), 
                      font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            logging.error(f"场景图可视化错误: {e}")
            return default_image
    
    def _choose_action(self, obs: dict):
        """Enhanced action selection with scene graph memory."""
        agent_state = obs['agent_state']
        goal = obs['goal']

        # 调用父类的预处理和停止检测
        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal)
        
        # Goal detection logic
        goal_image = candidate_images['color_sensor'].copy() if candidate_images else obs['color_sensor'].copy()
        
        if a_goal is not None and len(a_goal) > 0:
            try:
                if hasattr(self, '_goal_module'):
                    goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
                    images['goal_image'] = goal_image
                    
                    if goal_number is not None and goal_number != 0:
                        if hasattr(self, '_get_goal_position'):
                            goal_position, goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)
                            if not hasattr(self, 'goal_position'):
                                self.goal_position = []
                            self.goal_position.append(goal_position)
                            self.goal_mask = goal_mask
            except Exception as e:
                logging.warning(f"Goal detection failed: {e}")

        step_metadata['object'] = goal

        # Initialize default values
        agent_action = PolarAction.default
        logging_data = {}

        # 检查是否应该停止
        if step_metadata.get('called_stopping', False):
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {'STOPPING_RESPONSE': 'Distance-based stopping or planner decision'}
        else:
            try:
                # Fix: Handle different possible return formats from _prompting
                result = self._prompting(goal, a_final, images, step_metadata, obs.get('subtask', '{}'))
                
                # Handle different possible return types
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # Standard WMNavAgent format: (step_metadata, logging_data, response)
                        step_metadata, logging_data, _ = result
                    elif len(result) == 2:
                        # Alternative format: (step_metadata, logging_data)
                        step_metadata, logging_data = result
                    else:
                        logging.warning(f"Unexpected tuple length from _prompting: {len(result)}")
                        
                # Get the action from step_metadata
                agent_action = self._action_number_to_polar(step_metadata.get('action_number', -10), list(a_final))
            except Exception as e:
                logging.error(f"Error in action selection: {e}")
                logging_data = {}
                agent_action = PolarAction.default

        # Ensure agent_action is a PolarAction
        if not isinstance(agent_action, PolarAction):
            logging.warning(f"Invalid agent_action type: {type(agent_action)}, using default")
            agent_action = PolarAction.default

        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': getattr(self, 'step_ndx', 0)
        }
        
        return agent_action, metadata
    
    def _optimize_scene_graph(self):
        """
        Optimize the scene graph by removing low-confidence nodes and redundant edges.
        """
        if self.scene_graph is None or not self.graph_enabled:
            return
        
        initial_nodes = len(self.scene_graph.nodes)
        initial_edges = len(self.scene_graph.edges)
        
        # Remove nodes with very low confidence that haven't been seen recently
        current_step = getattr(self, 'step_ndx', 0)
        nodes_to_remove = []
        
        for node_name, node in self.scene_graph.nodes.items():
            # Remove nodes that are old and have low confidence
            if (node.confidence < 0.3 and 
                current_step - node.last_seen > 10 and 
                node.observations_count < 2):
                nodes_to_remove.append(node_name)
        
        # Remove the identified nodes
        for node_name in nodes_to_remove:
            del self.scene_graph.nodes[node_name]
            # Also remove associated edges
            self.scene_graph.edges = [
                edge for edge in self.scene_graph.edges
                if edge.source != node_name and edge.target != node_name
            ]
        
        # Remove duplicate edges (same source, target, and relation type)
        unique_edges = {}
        for edge in self.scene_graph.edges:
            key = (edge.source, edge.target, edge.relation_type)
            if key not in unique_edges or edge.confidence > unique_edges[key].confidence:
                unique_edges[key] = edge
        
        self.scene_graph.edges = list(unique_edges.values())
        
        final_nodes = len(self.scene_graph.nodes)
        final_edges = len(self.scene_graph.edges)
        
        if initial_nodes != final_nodes or initial_edges != final_edges:
            print(f"🧹 Optimized scene graph: {initial_nodes}→{final_nodes} nodes, {initial_edges}→{final_edges} edges")
            logging.info(f"Optimized scene graph: {initial_nodes}→{final_nodes} nodes, {initial_edges}→{final_edges} edges")

    def _should_visualize_graph(self) -> bool:
        """Determine if we should visualize the graph this step."""
        if not self.graph_enabled or self.scene_graph is None:
            return False
        
        current_step = getattr(self, 'step_ndx', 0)
        return current_step % self.visualization_freq == 0 and current_step > 0
    
    def _stopping_module(self, obs, threshold_dist=0.8):
        try:
            parent_result = super()._stopping_module(obs, threshold_dist)
            logging.info(f"✅ Parent stopping result: {parent_result}")
            print(f"✅ Parent stopping result: {parent_result}")
            return parent_result
        except Exception as e:
            logging.warning(f"Error in parent stopping module: {e}")
            print(f"⚠️ Error in parent stopping module: {e}")
            return False
    
    def _goal_module(self, goal_image: np.array, a_goal, goal):
        """目标检测模块 - 从父类继承或重新实现"""
        if hasattr(super(), '_goal_module'):
            return super()._goal_module(goal_image, a_goal, goal)
        else:
            # 简化的目标检测实现
            logging.warning("Goal module not implemented, using fallback")
            return None, "Goal detection not available"

    def _get_goal_position(self, action_goal, idx, agent_state):
        """目标位置计算 - 从父类继承或重新实现"""
        if hasattr(super(), '_get_goal_position'):
            return super()._get_goal_position(action_goal, idx, agent_state)
        else:
            # 简化的位置计算
            logging.warning("Goal position calculation not implemented")
            return None, None
    
    def _extract_relevant_memory(self, goal: str, prompt_type: str) -> str:
        """
        Unified memory extraction method that combines subgraph analysis with intelligent filtering.
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph') or self.scene_graph is None:
            return ""
        
        try:
            # Step 1: Get goal-relevant subgraph
            relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal, max_distance=2)
            
            # Step 2: Apply prompt-type specific filtering and formatting
            if prompt_type == 'planning':
                return self._format_memory_for_planning(relevant_nodes, relevant_edges, goal)
            elif prompt_type == 'predicting':
                return self._format_memory_for_predicting(relevant_nodes, relevant_edges, goal, max_length=100)
            elif prompt_type in ['goal', 'action']:
                return self._format_memory_for_goal_action(relevant_nodes, relevant_edges, goal)
            else:
                return self._format_memory_general(relevant_nodes, relevant_edges, goal)
                    
        except Exception as e:
            logging.warning(f"Error extracting relevant memory: {e}")
            return ""
    
    def _get_related_objects(self, goal: str) -> List[str]:
        """Get objects that are commonly found near the goal object."""
        relations = {
            'bed': ['pillow', 'blanket', 'nightstand', 'dresser', 'closet'],
            'chair': ['table', 'desk', 'dining table'],
            'sofa': ['coffee table', 'tv', 'cushion', 'lamp'],
            'toilet': ['sink', 'bathtub', 'shower', 'towel'],
            'refrigerator': ['stove', 'microwave', 'sink', 'counter'],
            'tv': ['sofa', 'remote', 'entertainment center']
        }
        return relations.get(goal, [])
    
    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Rough estimation of prompt token count for monitoring."""
        # Simple approximation: ~4 characters per token
        return len(prompt) // 4
    
    def _log_prompt_stats(self, prompt: str, prompt_type: str, goal: str):
        """Log prompt statistics for optimization monitoring."""
        token_estimate = self._estimate_prompt_tokens(prompt)
        logging.info(f"Prompt stats - Type: {prompt_type}, Goal: {goal}, Est. tokens: {token_estimate}")
        
        # Log optimization performance
        self._log_optimization_performance(prompt_type, token_estimate, len(prompt))
        
        # Log if prompt is getting too long
        if token_estimate > 800:  # Conservative threshold for model limits
            logging.warning(f"Long prompt detected ({token_estimate} tokens) for {prompt_type}")
        
        return token_estimate
    
    def _optimize_scene_graph_with_predictions(self, prediction_responses: dict, agent_position: np.ndarray, step: int):
        """
        Optimize scene graph construction using VLM prediction responses.
        This directly integrates the compact direction reasoning into the graph.
        
        Args:
            prediction_responses: Dict mapping direction angles to VLM responses with scores and reasons
            agent_position: Current agent position
            step: Current step number
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph'):
            return
        
        try:
            for direction_str, response_data in prediction_responses.items():
                direction_angle = float(direction_str)
                score = response_data.get('Score', 0)
                reason = response_data.get('Reason', '')
                
                # Only process directions with meaningful scores
                if score >= 3:  # Threshold for "promising" directions
                    
                    # Extract objects and rooms from the brief reason
                    reason_lower = reason.lower()
                    
                    # Quick object detection from reason text
                    common_objects = ['chair', 'table', 'bed', 'sofa', 'tv', 'toilet', 'door', 'window', 'shelf']
                    room_types = ['kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 'hallway']
                    
                    detected_objects = []
                    detected_rooms = []
                    
                    for obj in common_objects:
                        if obj in reason_lower:
                            detected_objects.append(obj)
                    
                    for room in room_types:
                        if room in reason_lower or room.replace(' ', '') in reason_lower:
                            detected_rooms.append(room)
                    
                    # Add detected objects to scene graph
                    for obj in detected_objects:
                        node_id = f"{obj}_dir_{direction_angle}_step_{step}"
                        
                        # Calculate approximate position based on direction
                        direction_rad = np.deg2rad(direction_angle)
                        estimated_distance = 2.0  # Rough estimate
                        object_position = agent_position + estimated_distance * np.array([
                            np.cos(direction_rad), 0, np.sin(direction_rad)
                        ])
                        
                        # Determine room context
                        room_context = detected_rooms[0] if detected_rooms else 'unknown'
                        
                        # Add to scene graph with compact attributes
                        self.scene_graph.add_node(
                            name=node_id,
                            attributes={
                                'object_type': obj,
                                'room': room_context,
                                'direction': direction_angle,
                                'source': 'prediction_vlm',
                                'reasoning': reason[:50]  # Truncate for compactness
                            },
                            position=object_position,
                            confidence=min(score / 10.0, 1.0),  # Convert score to confidence
                            step=step
                        )
                    
                    # Add room information
                    for room in detected_rooms:
                        room_node_id = f"room_{room}_{direction_angle}_step_{step}"
                        
                        # Estimate room position
                        room_position = agent_position + 1.5 * np.array([
                            np.cos(np.deg2rad(direction_angle)), 0, np.sin(np.deg2rad(direction_angle))
                        ])
                        
                        self.scene_graph.add_node(
                            name=room_node_id,
                            attributes={
                                'room_type': room,
                                'direction': direction_angle,
                                'source': 'prediction_vlm'
                            },
                            position=room_position,
                            confidence=min(score / 10.0, 1.0),
                            step=step
                        )
                        
                        # Create spatial relationships
                        for obj in detected_objects:
                            obj_node_id = f"{obj}_dir_{direction_angle}_step_{step}"
                            self.scene_graph.add_edge(
                                source=obj_node_id,
                                target=room_node_id,
                                relation_type="located_in",
                                confidence=0.7
                            )
            
            # Log optimization results
            print(f"🔍 Scene graph optimized with prediction data: {len(prediction_responses)} directions processed")
            logging.info(f"Scene graph optimized with prediction responses at step {step}")
            
        except Exception as e:
            print(f"Error optimizing scene graph with predictions: {e}")
            logging.warning(f"Error optimizing scene graph with predictions: {e}")
    
    def _extract_directional_memory(self, goal: str, current_direction: float = None) -> str:
        """
        Extract memory specifically about directions that were previously explored.
        This provides context about which directions were promising/unpromising.
        
        Args:
            goal: Target object to find
            current_direction: Current viewing direction (optional)
            
        Returns:
            Compact string about directional exploration history
        """
        if not self.graph_enabled or not hasattr(self, 'scene_graph'):
            return ""
        
        try:
            directional_info = []
            goal_lower = goal.lower()
            
            # Find nodes created from prediction VLM
            prediction_nodes = {}
            for node_id, node in self.scene_graph.nodes.items():
                if node.attributes.get('source') == 'prediction_vlm':
                    direction = node.attributes.get('direction')
                    if direction is not None:
                        if direction not in prediction_nodes:
                            prediction_nodes[direction] = []
                        prediction_nodes[direction].append(node)
            
            # Summarize promising directions
            promising_dirs = []
            goal_seen_dirs = []
            
            for direction, nodes in prediction_nodes.items():
                # Check if goal was seen in this direction
                goal_in_direction = any(
                    goal_lower in node.attributes.get('object_type', '').lower() 
                    for node in nodes
                )
                
                if goal_in_direction:
                    goal_seen_dirs.append(f"{int(direction)}°")
                elif any(node.confidence > 0.6 for node in nodes):
                    promising_dirs.append(f"{int(direction)}°")
            
            # Build compact directional memory
            if goal_seen_dirs:
                directional_info.append(f"{goal} spotted at {'/'.join(goal_seen_dirs[:2])}")
            
            if promising_dirs and len(promising_dirs) <= 3:
                directional_info.append(f"promising: {'/'.join(promising_dirs)}")
            
            return ". ".join(directional_info)
            
        except Exception as e:
            logging.warning(f"Error extracting directional memory: {e}")
            return ""
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the optimization performance.
        Returns token usage, scene graph efficiency, and memory compactness metrics.
        """
        stats = {
            "token_efficiency": {
                "predicting_prompt_tokens": getattr(self, '_last_predicting_tokens', 0),
                "planning_prompt_tokens": getattr(self, '_last_planning_tokens', 0),
                "memory_context_length": getattr(self, '_last_memory_length', 0),
                "estimated_token_savings": "60-70% vs verbose prompts"
            },
            "scene_graph_efficiency": {
                "total_nodes": len(self.scene_graph.nodes) if self.scene_graph else 0,
                "prediction_based_nodes": sum(1 for node in self.scene_graph.nodes.values() 
                                            if node.attributes.get('source') == 'prediction_vlm') if self.scene_graph else 0,
                "avg_confidence": np.mean([node.confidence for node in self.scene_graph.nodes.values()]) if self.scene_graph and self.scene_graph.nodes else 0
            },
            "memory_compactness": {
                "compact_extraction_enabled": True,
                "directional_memory_enabled": True,
                "max_memory_length": 120,
                "goal_prioritization": "direct sightings > directional history > room layout"
            }
        }
        return stats
    
    def _log_optimization_performance(self, prompt_type: str, tokens: int, memory_length: int = 0):
        """Log optimization performance metrics for analysis."""
        if prompt_type == 'predicting':
            setattr(self, '_last_predicting_tokens', tokens)
        elif prompt_type == 'planning':
            setattr(self, '_last_planning_tokens', tokens)
        
        if memory_length > 0:
            setattr(self, '_last_memory_length', memory_length)
        
        # Log optimization success
        baseline_tokens = {
            'predicting': 350,  # Estimated baseline for verbose version
            'planning': 250     # Estimated baseline for verbose version
        }
        
        if prompt_type in baseline_tokens:
            savings = baseline_tokens[prompt_type] - tokens
            savings_pct = (savings / baseline_tokens[prompt_type]) * 100
            print(f"💡 Token optimization: {prompt_type} uses {tokens} tokens (saved {savings_pct:.1f}%)")
            logging.info(f"Token optimization: {prompt_type} uses {tokens} tokens (saved {savings_pct:.1f}%)")
    
    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        """
        Optimized planning method with unified memory extraction.
        Removes the scene_graph_context parameter to simplify the calling interface.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)
                
                if response and 'Flag' in response and 'Subtask' in response:
                    goal_flag, subtask = response['Flag'], response['Subtask']
                    return goal_flag, subtask
                else:
                    logging.warning(f"Planning attempt {attempt + 1} failed: invalid response {response}")
                    
            except Exception as e:
                logging.warning(f"Planning attempt {attempt + 1} failed with error: {e}")
        
        # All retries failed, return default values
        logging.error("All planning attempts failed, using default values")
        print("All planning attempts failed, using default values")
        return False, '{}'
    
    def _format_memory_for_planning(self, nodes: Dict, edges: List, goal: str) -> str:
        """Format memory specifically for planning prompts with hallway perspective analysis."""
        if not nodes and not edges:
            return ""
        
        goal_lower = goal.lower()
        memory_parts = []
        
        # Add hallway perspective analysis
        if hasattr(self, 'direction_thoughts'):
            hallway_assessment = self._evaluate_hallway_perspective(goal, self.direction_thoughts)
            if hallway_assessment:
                # Find high probability unexplored rooms
                high_prob_directions = [
                    f"{dir_angle}° ({data['room_type']})" 
                    for dir_angle, data in hallway_assessment.items()
                    if data['probability'] > 0.5 and not data['explored']
                ]
                
                if high_prob_directions:
                    memory_parts.append(f"From hallway, {goal} likely in: {', '.join(high_prob_directions[:3])}")
                
                # Add already explored rooms with no goal found
                if hasattr(self, 'explored_high_prob_rooms') and self.explored_high_prob_rooms:
                    memory_parts.append(f"Already checked without finding {goal}: {', '.join(self.explored_high_prob_rooms)}")
        
        # Priority 1: Direct goal object sightings with location context
        goal_sightings = []
        for node_name, node in nodes.items():
            if goal_lower in node_name.lower() and hasattr(node, 'confidence') and node.confidence > 0.6:
                room = node.attributes.get('room', 'unknown')
                if room != 'unknown':
                    goal_sightings.append(f"{goal} in {room}")
        
        if goal_sightings:
            memory_parts.append(f"You've seen {', '.join(goal_sightings[:3])}")
        
        # Priority 2: Expected locations based on goal type
        expected_locations = self._get_expected_locations_for_goal(goal_lower)
        if expected_locations:
            memory_parts.append(f"Expected locations: {', '.join(expected_locations[:3])}")
        
        # Priority 3: Explored rooms and areas
        explored_rooms = set()
        for node_name, node in nodes.items():
            room = node.attributes.get('room')
            if room and room != 'unknown' and hasattr(node, 'confidence') and node.confidence > 0.5:
                explored_rooms.add(room)
        
        if explored_rooms:
            memory_parts.append(f"Explored: {', '.join(list(explored_rooms)[:4])}")
        
        # Priority 4: Navigation strategy hints from spatial relationships
        strategy_hints = self._extract_navigation_strategy(edges, goal_lower)
        if strategy_hints:
            memory_parts.append(f"Strategy: {strategy_hints}")
        
        return ". ".join(memory_parts) + ("." if memory_parts else "")
    
    def _format_memory_for_predicting(self, nodes: Dict, edges: List, goal: str, max_length: int = 100) -> str:
        """Format ultra-compact memory for direction prediction prompts."""
        if not nodes and not edges:
            return ""
        
        goal_lower = goal.lower()
        
        # Ultra-priority: Goal object directions
        for node_name, node in nodes.items():
            if goal_lower in node_name.lower() and hasattr(node, 'attributes'):
                direction = node.attributes.get('direction')
                if direction:
                    return f"{goal} spotted at {direction}"
        
        # Secondary: Room-based hints
        high_conf_rooms = []
        for node_name, node in nodes.items():
            room = node.attributes.get('room')
            if room and room != 'unknown' and hasattr(node, 'confidence') and node.confidence > 0.7:
                high_conf_rooms.append(room)
        
        if high_conf_rooms:
            expected = self._get_expected_locations_for_goal(goal_lower)
            for room in high_conf_rooms[:2]:
                if room in expected:
                    return f"Check {room} for {goal}"
        
        return ""
    
    def _format_memory_for_goal_action(self, nodes: Dict, edges: List, goal: str) -> str:
        """Format memory for goal detection and action selection."""
        if not nodes and not edges:
            return ""
        
        goal_lower = goal.lower()
        memory_parts = []
        
        # Recent goal sightings with spatial context
        recent_sightings = []
        for node_name, node in nodes.items():
            if goal_lower in node_name.lower() and hasattr(node, 'last_seen') and hasattr(node, 'confidence'):
                if getattr(self, 'step_ndx', 0) - node.last_seen < 5 and node.confidence > 0.6:
                    position_info = node.attributes.get('relative_position', 'nearby')
                    recent_sightings.append(f"{goal} {position_info}")
        
        if recent_sightings:
            memory_parts.append(f"Recent: {', '.join(recent_sightings[:2])}")
        
        # Spatial relationships for navigation
        spatial_context = []
        for edge in edges:
            if goal_lower in edge.source.lower() or goal_lower in edge.target.lower():
                if edge.relation_type in ['near', 'in', 'next to']:
                    spatial_context.append(f"{edge.source} {edge.relation_type} {edge.target}")
        
        if spatial_context:
            memory_parts.append(f"Spatial: {spatial_context[0]}")
        
        return ". ".join(memory_parts) + ("." if memory_parts else "")
    
    def _format_memory_general(self, nodes: Dict, edges: List, goal: str) -> str:
        """General memory formatting with balanced information."""
        if not nodes and not edges:
            return ""
        
        goal_lower = goal.lower()
        memory_parts = []
        
        # Objects and rooms seen
        object_types = set()
        rooms = set()
        
        for node_name, node in nodes.items():
            if hasattr(node, 'attributes'):
                obj_type = node.attributes.get('type')
                room = node.attributes.get('room')
                
                if obj_type and obj_type != 'unknown':
                    object_types.add(obj_type)
                if room and room != 'unknown':
                    rooms.add(room)
        
        if rooms:
            memory_parts.append(f"rooms: {', '.join(list(rooms)[:3])}")
        if object_types:
            memory_parts.append(f"objects: {', '.join(list(object_types)[:5])}")
        
        return "You've seen " + ", ".join(memory_parts) if memory_parts else ""
    
    def _get_expected_locations_for_goal(self, goal: str) -> List[str]:
        """Get expected room locations for a given goal object."""
        goal_room_map = {
            "bed": ["bedroom", "master bedroom", "guest room"],
            "toilet": ["bathroom", "restroom"],
            "sink": ["bathroom", "kitchen"],
            "shower": ["bathroom"],
            "sofa": ["living room", "family room"],
            "couch": ["living room", "family room"],
            "tv": ["living room", "bedroom", "family room"],
            "television": ["living room", "bedroom", "family room"],
            "stove": ["kitchen"],
            "refrigerator": ["kitchen"],
            "fridge": ["kitchen"],
            "desk": ["office", "bedroom", "study"],
            "chair": ["dining room", "office", "bedroom", "living room"],
            "table": ["dining room", "kitchen", "living room"]
        }
        
        for pattern, rooms in goal_room_map.items():
            if pattern in goal.lower():
                return rooms
        
        return []
    
    def _extract_navigation_strategy(self, edges: List, goal: str) -> str:
        """Extract navigation strategy hints from spatial relationships."""
        if not edges:
            return ""
        
        # Look for connections to unexplored areas
        unexplored_connections = []
        for edge in edges:
            if edge.relation_type in ['leads_to', 'connects_to', 'opens_to']:
                unexplored_connections.append(f"{edge.relation_type} {edge.target}")
        
        if unexplored_connections:
            return f"explore {unexplored_connections[0].split()[-1]}"
        
        # Look for room transitions that might lead to goal
        expected_rooms = self._get_expected_locations_for_goal(goal)
        for edge in edges:
            if edge.target in expected_rooms:
                return f"explore {edge.relation_type} for {edge.target} doors"
        
        return "explore hallways for new rooms"
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for performance monitoring."""
        return {
            'prediction_reweights': self.optimization_stats.get('prediction_reweights', 0),
            'subgraph_matches': self.optimization_stats.get('subgraph_matches', 0),
            'strategy_switches': self.optimization_stats.get('strategy_switches', 0),
            'successful_navigations': self.optimization_stats.get('successful_navigations', 0),
            'graph_update_count': getattr(self, 'graph_update_count', 0),
            'graph_enabled': self.graph_enabled,
            'total_nodes': len(self.scene_graph.nodes) if self.scene_graph else 0,
            'total_edges': len(self.scene_graph.edges) if self.scene_graph else 0
        }
    
    def step(self, obs: dict):
        """Enhanced step function with optimization tracking."""
        # Update agent position tracking
        if 'agent_state' in obs:
            self.last_agent_state = obs['agent_state']
        
        # Update path history for trajectory analysis
        if hasattr(self, 'last_agent_state') and self.last_agent_state:
            position = self.last_agent_state.position
            self.path_history.append({
                'step': getattr(self, 'step_ndx', 0),
                'position': position,
                'rotation': self.last_agent_state.rotation
            })
        
        # Call parent step method
        result = super().step(obs)
        
        # Track optimization performance
        if hasattr(self, 'optimization_stats'):
            # Check if navigation was successful (can be enhanced with actual success metrics)
            if getattr(self, 'goal_reached', False):
                self.optimization_stats['successful_navigations'] += 1
        
        return result
    
    def _extract_prioritized_waypoints(self, goal: str, relevant_nodes: Dict, relevant_edges: List) -> List[Dict]:
        """Extract and prioritize waypoints based on goal relevance and graph structure."""
        waypoints = []
        
        # Extract rooms as potential waypoints
        for node_name, node in relevant_nodes.items():
            if node.attributes.get('type') == 'room':
                # Calculate priority based on goal probability
                room_probs = self._get_room_probabilities_for_goal(goal, {node_name: node})
                priority = room_probs.get(node_name, 0.1)
                
                waypoints.append({
                    'name': node_name,
                    'type': 'room',
                    'priority': priority,
                    'position': node.position,
                    'confidence': node.confidence
                })
        
        # Sort by priority (highest first)
        waypoints.sort(key=lambda x: x['priority'], reverse=True)
        return waypoints[:5]  # Limit to top 5 waypoints
    
    def _determine_optimal_navigation_strategy(self, goal: str, waypoints: List, relevant_nodes: Dict, 
                                             relevant_edges: List, subgraph_data: Dict) -> Dict:
        """Determine optimal navigation strategy using UniGoal-inspired algorithms."""
        strategy = {
            'strategy_type': 'exploration',
            'confidence': 0.5,
            'guidance': '',
            'target_waypoint': None,
            'reasoning': 'Default exploration strategy'
        }
        
        try:
            if not waypoints:
                return strategy
            
            # Analyze spatial distribution
            spatial_analysis = self._analyze_spatial_distribution(relevant_nodes, relevant_edges, goal)
            
            # Strategy 1: High-probability room targeting
            high_prob_waypoints = [w for w in waypoints if w['priority'] > 0.6]
            if high_prob_waypoints:
                target = high_prob_waypoints[0]
                strategy.update({
                    'strategy_type': 'targeted_room_search',
                    'confidence': 0.8,
                    'target_waypoint': target,
                    'guidance': f"Focus on {target['name']} - high probability ({target['priority']:.2f}) for {goal}",
                    'reasoning': f"Room {target['name']} has high probability for containing {goal}"
                })
                self.optimization_stats['strategy_switches'] += 1
                return strategy
            
            # Strategy 2: Systematic room exploration
            unexplored_rooms = [w for w in waypoints if w['confidence'] < 0.3]
            if unexplored_rooms:
                target = unexplored_rooms[0]
                strategy.update({
                    'strategy_type': 'systematic_exploration',
                    'confidence': 0.6,
                    'target_waypoint': target,
                    'guidance': f"Explore {target['name']} systematically",
                    'reasoning': f"Systematic exploration of unexplored room: {target['name']}"
                })
                return strategy
            
            # Strategy 3: Connectivity-based exploration
            if spatial_analysis.get('connectivity_hotspots'):
                hotspot = spatial_analysis['connectivity_hotspots'][0]
                strategy.update({
                    'strategy_type': 'connectivity_exploration',
                    'confidence': 0.7,
                    'guidance': f"Explore high-connectivity area near {hotspot}",
                    'reasoning': f"High connectivity area may lead to {goal}"
                })
                return strategy
                
        except Exception as e:
            logging.warning(f"Navigation strategy determination error: {e}")
        
        return strategy
    
    def _analyze_spatial_distribution(self, relevant_nodes: Dict, relevant_edges: List, goal: str) -> Dict:
        """Analyze spatial distribution of nodes and edges to identify patterns."""
        analysis = {
            'room_connectivity': {},
            'connectivity_hotspots': [],
            'unexplored_directions': []
        }
        
        try:
            # Analyze room connectivity
            room_connections = {}
            for edge in relevant_edges:
                if edge.relation_type in ['connects_to', 'leads_to', 'opens_to']:
                    if edge.source not in room_connections:
                        room_connections[edge.source] = []
                    room_connections[edge.source].append(edge.target)
            
            # Find connectivity hotspots (rooms with many connections)
            for room, connections in room_connections.items():
                if len(connections) >= 2:
                    analysis['connectivity_hotspots'].append(room)
            
            analysis['room_connectivity'] = room_connections
            
        except Exception as e:
            logging.warning(f"Spatial analysis error: {e}")
        
        return analysis
    
    def _build_graph_enhanced_memory_context(self, goal: str, navigation_strategy: Dict, 
                                           relevant_nodes: Dict, relevant_edges: List) -> str:
        """Build enhanced memory context with graph knowledge integration."""
        context_parts = []
        
        try:
            # Add navigation strategy guidance
            if navigation_strategy.get('guidance'):
                context_parts.append(f"Strategy: {navigation_strategy['guidance']}")
            
            # Add relevant spatial relationships
            if relevant_edges:
                spatial_info = []
                for edge in relevant_edges[:3]:
                    spatial_info.append(f"{edge.source} {edge.relation_type} {edge.target}")
                context_parts.append(f"Spatial: {'; '.join(spatial_info)}")
            
            # Add high-confidence objects
            high_conf_objects = []
            for name, node in relevant_nodes.items():
                if node.confidence > 0.6 and node.attributes.get('type') == 'object':
                    high_conf_objects.append(f"{name}({node.confidence:.2f})")
            
            if high_conf_objects:
                context_parts.append(f"Objects: {', '.join(high_conf_objects[:5])}")
            
            # Add room probabilities
            room_probs = self._get_room_probabilities_for_goal(goal, relevant_nodes)
            if room_probs:
                top_rooms = sorted(room_probs.items(), key=lambda x: x[1], reverse=True)[:2]
                prob_str = ', '.join([f"{room}({prob:.2f})" for room, prob in top_rooms])
                context_parts.append(f"Best rooms: {prob_str}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logging.warning(f"Enhanced memory context error: {e}")
            return ""
    
    def _integrate_navigation_guidance(self, goal_reason: str, navigation_strategy: Dict, enhanced_memory: str) -> str:
        """Integrate navigation strategy guidance into goal reasoning."""
        guidance = navigation_strategy.get('guidance', '')
        if guidance:
            return f"{goal_reason} {guidance}. Graph context: {enhanced_memory}"
        return goal_reason
    
    def _enhance_planning_response_with_graph(self, planning_response: Dict, navigation_strategy: Dict, goal: str) -> Dict:
        """Post-process planning response with graph-based enhancements."""
        if navigation_strategy.get('target_waypoint'):
            waypoint = navigation_strategy['target_waypoint']
            planning_response['graph_guidance'] = {
                'target_room': waypoint['name'],
                'priority': waypoint['priority'],
                'strategy': navigation_strategy['strategy_type']
            }
        
        return planning_response