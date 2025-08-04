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
import json

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
        print("🚀 GraphMemoryAgent (OPTIMIZED VERSION) initializing...")
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
        
        # Initialize attributes that may be conditionally set
        self.goal_detected_direction = None
        
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
            
        # Reset navigation history and states
        self.path_history = []
        self.graph_update_count = 0
        self.exploration_history = []
        self.goal_detected_direction = None
        self.current_goal = None
        
        print("🎯 GraphMemoryAgent reset complete!")
    
    def _build_enhanced_room_object_knowledge(self):
        """Build more specialized room-object probability knowledge based on UniGoal approach"""
        return {
            "bedroom": {
                "bed": 0.98, "dresser": 0.85, "nightstand": 0.85, "wardrobe": 0.8,
                "lamp": 0.7, "mirror": 0.6, "desk": 0.5, "chair": 0.4,
                "toilet": 0.01, "sink": 0.05, "shower": 0.01, "bathtub": 0.01,
                "refrigerator": 0.01, "stove": 0.01, "dining_table": 0.02,
                "tv": 0.3, "couch": 0.05, "recliner": 0.05
            },
            "bathroom": {
                "toilet": 0.98, "sink": 0.95, "shower": 0.9, "bathtub": 0.8,
                "mirror": 0.9, "towel": 0.8, "cabinet": 0.7,
                "bed": 0.01, "sofa": 0.01, "dining_table": 0.01, "refrigerator": 0.01
            },
            "kitchen": {
                "refrigerator": 0.95, "stove": 0.95, "sink": 0.9, "microwave": 0.9,
                "counter": 0.95, "cabinet": 0.9, "dining_table": 0.6,
                "bed": 0.01, "toilet": 0.01, "sofa": 0.02
            },
            "living_room": {
                "sofa": 0.95, "tv": 0.9, "coffee_table": 0.85, "chair": 0.7,
                "lamp": 0.8, "bookshelf": 0.6, "plant": 0.5, "recliner": 0.8,
                "bed": 0.02, "toilet": 0.01, "refrigerator": 0.01
            },
            "hallway": {
                "door": 0.9, "picture": 0.5, "plant": 0.3, "cabinet": 0.3,
                "bed": 0.01, "sofa": 0.01, "refrigerator": 0.01, "toilet": 0.01
            }
        }
    
    def _determine_exploration_strategy(self, overlap_score: float, common_objects: Set[str], goal: str):
        """
        UniGoal-inspired three-stage exploration strategy based on graph matching score:
        Stage 1: Zero Matching (S < σ1=0.3) - FIND_NEW_AREAS 
        Stage 2: Partial Matching (σ1 ≤ S < σ2=0.7) - FOCUSED_SEARCH
        Stage 3: Perfect Matching (S ≥ σ2=0.7) - DIRECT_NAVIGATION
        """
        goal_detected = goal.lower() in common_objects
        
        # === UniGoal Stage 3: Perfect Matching (S ≥ 0.7) ===
        if overlap_score >= 0.8:
            # Stage 3: Perfect Matching - goal graph is well observed in scene graph
            if goal_detected:
                # Central object of goal graph is matched - verify and navigate
                high_confidence_goal = self._has_high_confidence_goal_prediction(goal)
                if high_confidence_goal:
                    print(f"🎯 STAGE 3 (Perfect Matching): Goal detected with high confidence, score={overlap_score:.3f}")
                    return "DIRECT_NAVIGATION"
                else:
                    print(f"🔍 STAGE 3 (Perfect Matching): Goal detected but needs verification, score={overlap_score:.3f}")
                    return "GOAL_VERIFICATION"  # New strategy for goal verification phase
            else:
                print(f"🔍 STAGE 3 (Perfect Matching): High structural match but goal not detected, score={overlap_score:.3f}")
                return "FOCUSED_SEARCH"
        
        # === UniGoal Stage 2: Partial Matching (0.3 ≤ S < 0.7) ===
        elif overlap_score >= 0.3:
            # Stage 2: Partial matching - some elements of goal graph observed
            # Need to find anchor pairs and infer remaining goal locations
            print(f"🔍 STAGE 2 (Partial Matching): Some goal elements found, score={overlap_score:.3f}")
            
            if len(common_objects) >= 2:
                # Multiple anchor pairs - can perform coordinate projection
                return "COORDINATE_PROJECTION"  # New strategy for spatial inference
            elif len(common_objects) == 1:
                # Single anchor pair - focused search around known object
                return "FOCUSED_SEARCH"
            else:
                # Structural similarity without object matches - explore related areas
                return "EXPLORE_REMAINING"
        
        # === UniGoal Stage 1: Zero Matching (S < 0.3) ===
        else:
            # Stage 1: Zero matching - need to expand explored region
            print(f"🔍 STAGE 1 (Zero Matching): Minimal graph overlap, score={overlap_score:.3f}")
            
            # Check if we have any goal-relevant context at all
            if self._has_goal_relevant_context(goal):
                # Some semantic context exists - explore frontiers with goal guidance
                return "SEMANTIC_FRONTIER_EXPLORATION"  # New strategy for goal-guided frontier exploration
            else:
                # No relevant context - pure exploration
                return "FIND_NEW_AREAS"
    
    def _find_unexplored_directions(self) -> Set[float]:
        """Find directions that haven't been explored recently."""
        unexplored = {30.0, 90.0, 150.0, 210.0, 270.0, 330.0}
        
        # Check recent path history (last 3 steps)
        if hasattr(self, 'path_history') and len(self.path_history) > 2:
            recent_positions = [p['position'] for p in self.path_history[-3:]]
            explored_directions = set()
            
            # Current position
            current_pos = self._get_current_agent_position()
            
            # Find directions we've already moved in
            for pos in recent_positions:
                # Ensure position is numpy array
                if not isinstance(pos, np.ndarray):
                    pos = np.array(pos)
                if np.array_equal(current_pos, pos):
                    continue
                    
                # Calculate direction to this previous position
                dir_vector = pos - current_pos
                angle = np.degrees(np.arctan2(dir_vector[0], dir_vector[2]))
                
                # Convert to nearest of our 6 directions
                nearest_dir = round(angle / 60) * 60 + 30
                if nearest_dir <= 0:
                    nearest_dir += 360
                if nearest_dir > 360:
                    nearest_dir -= 360
                    
                # Add to explored directions
                explored_directions.add(float(nearest_dir))
            
            # Remove recently explored directions
            unexplored -= explored_directions
        
        return unexplored    
    
    def _has_high_confidence_goal_prediction(self, goal: str) -> bool:
        """Check if any current predictions have high confidence (≥8) for directions containing the goal."""
        if not hasattr(self, 'last_predictions') or not self.last_predictions:
            return False
            
        for direction, data in self.last_predictions.items():
            if isinstance(data, dict) and 'Score' in data:
                score = data.get('Score', 0)
                # Convert score to numeric if it's a string
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = 0
                        
                # Check if this direction has high confidence AND mentions the goal
                if score >= 8:
                    thought_text = data.get('thought_text', '').lower()
                    if goal.lower() in thought_text and 'definitely see' in thought_text:
                        return True
        
        return False
    
    
    def _parse_vlm_json_response(self, response_text: str, expected_structure=None):
        """
        Safely extract and parse JSON from VLM responses.
        
        Args:
            response_text: The raw response from the VLM
            expected_structure: Optional dictionary with expected keys and default values
            
        Returns:
            Parsed JSON object or default structure if parsing fails
        """
        import re  # Import re at the beginning of the function
        
        if not response_text or not isinstance(response_text, str):
            return {} if expected_structure is None else expected_structure
        
        # Remove markdown formatting
        response_text = response_text.strip()
        markdown_patterns = [
            (r'```json\s*', ''),
            (r'```\s*$', ''),
            (r'^```\s*', ''),
            (r'`{1,3}', ''),
        ]
        
        for pattern, replacement in markdown_patterns:
            response_text = re.sub(pattern, replacement, response_text, flags=re.MULTILINE)
        
        response_text = response_text.strip()
        
        # Try direct JSON parsing first
        if response_text.startswith('{') and response_text.endswith('}'):
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON with regex
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fix common JSON issues
                json_str = json_str.replace('True', 'true').replace('False', 'false')
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        # Using ast.literal_eval is safer than plain eval()
                        import ast
                        return ast.literal_eval(json_str)
                    except (SyntaxError, ValueError):
                        pass
        
        # Return default structure if all parsing attempts fail
        return {} if expected_structure is None else expected_structure    

    def set_goal(self, goal: str):
        """Set a new navigation goal and decompose it into subgoals."""
        if not hasattr(self, 'current_goal') or self.current_goal != goal:
            self.current_goal = goal
            
            print(f"🎯 New goal set: {goal}")
            
            # Add goal node to scene graph if it doesn't exist
            if self.scene_graph and goal not in self.scene_graph.nodes:
                self.scene_graph.add_node(goal, {"type": "goal"}, confidence=0.5)
                
    def _validate_scene_graph(self, step: int):
        """Periodically validate and correct scene graph using VLM with improved parsing"""
        # Only validate occasionally to save computation
        if step % 5 != 0:
            return
            
        # Extract a small subgraph to validate
        relevant_nodes = list(self.scene_graph.nodes.keys())[:5]  # Limit to 5 nodes for efficiency
        if len(relevant_nodes) < 2:
            return
            
        # Find edges between these nodes
        relevant_edges = []
        for edge in self.scene_graph.edges:
            if edge.source in relevant_nodes and edge.target in relevant_nodes:
                relevant_edges.append(edge)
        
        # Validate using VLM
        node_text = ", ".join(relevant_nodes)
        edge_text = ", ".join([f"{e.source} {e.relation_type} {e.target}" for e in relevant_edges])
        
        prompt = (
            f"Given these objects: {node_text} and relationships: {edge_text}, "
            f"are there any unreasonable relationships? If so, how should they be corrected? "
            f"Return ONLY a valid JSON object with this exact structure (nothing else): "
            f"{{\"corrections\": [{{\"source\": \"object1\", \"target\": \"object2\", "
            f"\"current_relation\": \"relation\", \"corrected_relation\": \"new_relation\"}}]}}. "
            f"If all relationships are reasonable, return {{\"corrections\": []}}."
        )
        
        try:
            # Use VLM to get corrections
            response = self.PlanVLM.call([], prompt)
            
            # Use the robust parser with expected structure
            default_structure = {"corrections": []}
            corrections = self._parse_vlm_json_response(response, expected_structure=default_structure)
            
            # Apply corrections if any
            for correction in corrections.get("corrections", []):
                # Validate each correction has the required fields
                if all(key in correction for key in ["source", "target", "current_relation", "corrected_relation"]):
                    for edge in self.scene_graph.edges:
                        if (edge.source == correction["source"] and 
                            edge.target == correction["target"] and 
                            edge.relation_type == correction["current_relation"]):
                            edge.relation_type = correction["corrected_relation"]
                            logging.info(f"Corrected relationship: {edge.source} {edge.relation_type} {edge.target} -> {correction['corrected_relation']}")
                            break
                else:
                    logging.warning(f"Invalid correction structure: {correction}")
                        
        except Exception as e:
            logging.warning(f"Scene graph validation error: {e}")
            
    def _predicting_module(self, evaluator_image, goal):

        # Initialize goal if needed
        if not hasattr(self, 'current_goal') or self.current_goal != goal:
            self.set_goal(goal)
        
        # Step 1: Generate prediction prompt without graph context  
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        
        # Step 2: Get VLM predictions
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        logging.info(f"VLM response: {evaluator_response}")
        dct = self._eval_response(evaluator_response)
        
        # Step 3: Apply enhanced graph matching and navigation strategy
        if (self.scene_graph is not None and self.graph_enabled and hasattr(self, 'step_ndx')):
            agent_position = self._get_current_agent_position()
            
            # DEBUG: Print VLM predictions before processing
            logging.info(f"\n🔍 DEBUG - Step {self.step_ndx}: VLM Predictions Analysis")
            for direction, data in dct.items():
                if isinstance(data, dict) and 'Thought' in data:
                    thought = data['Thought']
                    score = data.get('Score', 0)
                    print(f"Direction {direction} (Score: {score}): {thought}")
            
            self._update_scene_graph_with_predictions(dct, agent_position, self.step_ndx)
            
            # Extract visible objects and create a temporary subgraph
            visible_objects = self._extract_objects_from_predictions(dct)
            
            # DEBUG: Print extracted objects
            print(f"\n Extracted Objects from Predictions:")
            if visible_objects:
                for obj in visible_objects:
                    print(f"  - {obj['name']} (confidence: {obj['confidence']:.2f}, direction: {obj['direction']})")
            else:
                print("  - No objects extracted")
                
            # DEBUG: Print current scene graph state
            print(f"\nCurrent Scene Graph State:")
            if self.scene_graph and self.scene_graph.nodes:
                for name, node in list(self.scene_graph.nodes.items())[:10]:  # Limit to first 10 for readability
                    print(f"  - {name}: confidence={node.confidence:.2f}, type={node.attributes.get('type', 'unknown')}")
                if len(self.scene_graph.nodes) > 10:
                    print(f"  ... and {len(self.scene_graph.nodes) - 10} more nodes")
            else:
                print("  - Empty scene graph")
            
            current_subgraph = self._build_observation_subgraph(visible_objects, agent_position)
            
            # Calculate overlap and determine exploration strategy
            overlap_score, common_objects = self._compute_graph_overlap(current_subgraph, goal)
            
            # DEBUG: Print overlap analysis
            print(f"\nGraph Overlap Analysis:")
            print(f"  - Overlap score: {overlap_score:.2f}")
            print(f"  - Common objects: {list(common_objects)}")
            
            # Periodically validate scene graph (every 5 steps)
            self._validate_scene_graph(self.step_ndx)
            
            # Determine exploration strategy
            exploration_strategy = self._determine_exploration_strategy(overlap_score, common_objects, goal)
            
            # DEBUG: Print exploration strategy
            print(f"  - Exploration strategy: {exploration_strategy}")
            print("="*80)
            
            # Store last strategy for prompt enhancement
            if not hasattr(self, 'exploration_history'):
                self.exploration_history = []
            self.exploration_history.append(exploration_strategy)
            
            # Multi-Stage Strategy Application ===
            dct = self._apply_unigoal_strategy_reweighting(dct, exploration_strategy, goal, overlap_score, common_objects, agent_position)
            
            # Store metadata for planning module
            dct['exploration_strategy'] = exploration_strategy
            dct['graph_confidence'] = overlap_score
            dct['matched_objects'] = list(common_objects)
            
            # Log results
            print(f"🎯 Graph matching: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}, strategy={exploration_strategy}")
            logging.info(f"Graph matching: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}, strategy={exploration_strategy}")
        
        # Step 5: Store predictions for other modules to access
        self.last_predictions = dct
        
        # Step 6: Check for high-confidence goal detection with explicit debugging
        # BUT only trigger goal detection for appropriate strategies
        goal_detected = False
        highest_score = 0
        detection_debug = []
        exploration_strategy = dct.get('exploration_strategy', 'FIND_NEW_AREAS')
        
        # Only allow goal detection for DIRECT_NAVIGATION strategy
        # Other strategies (FOCUSED_SEARCH, etc.) should not trigger immediate goal detection
        allow_goal_detection = exploration_strategy == "DIRECT_NAVIGATION"
        
        for direction, data in dct.items():
            if isinstance(data, dict):
                score = data.get('Score', 0)
                # Convert score to int if it's a string
                if isinstance(score, str):
                    try:
                        score = int(score)
                    except (ValueError, TypeError):
                        score = 0
                
                detection_debug.append(f"Dir {direction}: score={score} (type: {type(score)})")
                highest_score = max(highest_score, score)
                
                if score >= 9 and allow_goal_detection:
                    goal_detected = True
        
        print(f"🔍 Goal Detection Debug:")
        for debug_info in detection_debug:
            print(f"  {debug_info}")
        print(f"  Highest score: {highest_score}")
        print(f"  Exploration strategy: {exploration_strategy}")
        print(f"  Allow goal detection: {allow_goal_detection}")
        print(f"  Goal detected (score ≥ 9 AND strategy allows): {goal_detected}")
        
        if goal_detected:
            print(f"🎯 GOAL DETECTED in prediction phase with high confidence!")
            self.goal_detected_direction = next(
                (dir for dir, data in dct.items() if isinstance(data, dict) and data.get('Score', 0) >= 9),
                None
            )
        else:
            # Reset goal detection if no high score found OR strategy doesn't allow detection
            self.goal_detected_direction = None
            
        logging.info(f"Goal detected: {goal_detected}, direction: {self.goal_detected_direction}")
        
        return dct
    
    def _get_related_objects_for_goal(self, goal: str) -> List[str]:
        """Get objects that are commonly associated with the goal object"""
        goal_associations = {
            "bed": ["nightstand", "dresser", "lamp", "pillow", "bedroom"],
            "toilet": ["sink", "shower", "bathtub", "bathroom"],
            "sofa": ["coffee table", "tv", "living room", "chair"],
            "chair": ["table", "desk", "dining room"],
            "tv": ["sofa", "entertainment center", "living room"],
            "refrigerator": ["stove", "sink", "kitchen"],
            "table": ["chair", "dining room"]
        }
        
        return goal_associations.get(goal.lower(), [])
    
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
        if not predictions:
            return predictions
        
        # 检查是否有角度键和得分
        has_angles = any(k.isdigit() or k in ['30', '90', '150', '210', '270', '330'] for k in predictions.keys())
        if not has_angles:
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
                predictions = self._apply_graph_based_reweighting(
                    predictions, goal, overlap_score, common_objects, agent_position
                )
                predictions['graph_confidence'] = overlap_score
                predictions['matched_objects'] = list(common_objects)
                
                self.optimization_stats['prediction_reweights'] += 1
                print(f"🎯 Graph matching: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}")
                logging.info(f"Graph matching applied: overlap={overlap_score:.2f}, matched_objects={len(common_objects)}")
            
        except Exception as e:
            logging.warning(f"Error in graph-based reweighting: {e}")
        
        return predictions
    
    def _extract_objects_from_predictions(self, predictions: dict) -> List[Dict]:
        """Extract object mentions from VLM predictions with better negative context handling."""
        objects = []
        
        print("\n🔍 DEBUGGING: Object extraction from predictions")
        print(f"Predictions keys: {list(predictions.keys())}")
        
        # Process standard JSON format - from angle keys with Thought field
        common_objects = ["chair", "table", "sofa", "bed", "refrigerator", "tv", "sink", "toilet", "stove", "microwave"]
        room_types = ["bedroom", "bathroom", "kitchen", "living room", "hallway", "dining room", "office"]
        
        # From angle keys with Thought field
        for angle, data in predictions.items():
            if isinstance(data, dict) and 'Thought' in data:
                thought_text = data['Thought'].lower()
                print(f"\n📐 Angle {angle} - Processing thought text:")
                print(f"   Original: {data['Thought'][:200]}...")
                print(f"   Lowercase: {thought_text[:200]}...")
                
                # Process each sentence separately for better context handling
                sentences = thought_text.split('.')
                print(f"   Split into {len(sentences)} sentences")
                
                for i, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if len(sentence) < 5:  # Skip very short sentences
                        continue
                        
                    print(f"   Sentence {i+1}: '{sentence}'")
                    
                    # Extract objects from this sentence
                    for obj in common_objects + room_types:
                        if obj in sentence:
                            print(f"      Found '{obj}' in sentence")
                            
                            # Check validity against both the sentence AND the full thought for better context
                            is_valid_sentence = self._is_valid_object_mention(obj, sentence)
                            is_valid_full_context = self._is_valid_object_mention(obj, thought_text)
                            
                            # Object is valid only if it's valid in both contexts
                            is_valid = is_valid_sentence and is_valid_full_context
                            
                            print(f"      Valid in sentence: {is_valid_sentence}")
                            print(f"      Valid in full context: {is_valid_full_context}")
                            print(f"      Final validity: {is_valid}")
                            
                            if is_valid:
                                # Convert angle string to float for numerical operations
                                try:
                                    direction_angle = float(angle)
                                except (ValueError, TypeError):
                                    direction_angle = 0.0
                                    
                                obj_data = {
                                    'name': obj,
                                    'confidence': 0.6,
                                    'direction': direction_angle
                                }
                                objects.append(obj_data)
                                print(f"      ✅ ADDED object: {obj_data}")
                            else:
                                print(f"      ❌ REJECTED object: {obj} (negative context)")
        
        print(f"\n📊 Total objects extracted: {len(objects)}")
        for obj in objects:
            print(f"   - {obj['name']} (conf: {obj['confidence']}, dir: {obj['direction']})")
        
        return objects
    
    def _is_valid_object_mention(self, obj_name: str, text: str) -> bool:
        """
        Check if an object mention is valid (not in negative context and not a stop word).
        """
        print(f"        🧪 Validating '{obj_name}' in: '{text}'")
        
        # Skip common stop words
        stop_words = ["there", "possibility", "other", "another", "it", "this", "that"]
        if obj_name in stop_words:
            print(f"        ❌ Rejected: '{obj_name}' is a stop word")
            return False
        
        # First check for explicit negative statements about this specific object
        negative_specific_patterns = [
            f"no {obj_name}", f"not visible", f"no visible {obj_name}",
            f"cannot see {obj_name}", f"not found", f"no sign of {obj_name}",
            f"not in sight", f"no clear {obj_name}", f"doesn't have {obj_name}",
            f"does not have {obj_name}", f"{obj_name} is not", f"{obj_name} are not"
        ]
        
        # Also check for general negative patterns that apply to any object in the sentence
        general_negative_patterns = [
            "no strong indication", "no indication", "no sign of", "no evidence of",
            "but no", "however no", "yet no", "still no", "not visible",
            "cannot see", "not found", "not in sight", "no clear",
            "doesn't appear", "does not appear", "no apparent",
            "less likely", "unlikely", "not likely"
        ]
        
        for pattern in negative_specific_patterns:
            if pattern in text:
                print(f"        ❌ Rejected: Found negative pattern '{pattern}' for this object")
                return False
        
        # Check for general negative patterns that would make any object mention invalid
        for pattern in general_negative_patterns:
            if pattern in text:
                print(f"        ❌ Rejected: Found general negative pattern '{pattern}'")
                return False
        
        # Check for uncertain/speculative language that makes objects predicted rather than observed
        uncertain_patterns = [
            "might be", "could be", "likely", "possibly", "perhaps", 
            "may contain", "may lead to", "potential", "promising",
            "would expect", "typical", "usually", "often",
            "suggests", "indicates", "implies", "appears",
            "seems", "looks like", "resembles", "it is possible",
            "there is a possibility", "there might be", "could have"
        ]
        
        # Check if any uncertain pattern appears in the same sentence as the object
        for pattern in uncertain_patterns:
            if pattern in text:
                print(f"        ❌ Rejected: Found uncertain/predictive pattern '{pattern}'")
                return False
        
        # Different validation rules for different object types
        
        # For room types: Accept descriptive language like "living room area"
        if obj_name in ["bedroom", "bathroom", "kitchen", "living room", "dining room", "hallway", "office"]:
            room_descriptive_patterns = [
                f"{obj_name} area", f"{obj_name} with", f"in the {obj_name}",
                f"this {obj_name}", f"current {obj_name}", f"main {obj_name}"
            ]
            
            has_room_descriptive = any(pattern in text for pattern in room_descriptive_patterns)
            if has_room_descriptive:
                print(f"        ✅ Accepted: Found room descriptive language for '{obj_name}'")
                return True
        
        # For furniture/objects: Require more definite observation language
        definite_observation_patterns = [
            "i see", "i can see", "clearly see", "definitely see",
            "visible", "here is", "here are", "in front", "to the left", 
            "to the right", "behind", "with a", "with the"
        ]
        
        # Special check: "there is a" is only valid if NOT preceded by uncertain language
        if "there is a" in text or "there are" in text:
            # Check if it's in an uncertain context
            uncertain_context_patterns = [
                "it is possible there is", "possibly there is", "might be there is",
                "could be there is", "perhaps there is", "maybe there is"
            ]
            
            is_uncertain_context = any(pattern in text for pattern in uncertain_context_patterns)
            if not is_uncertain_context:
                print(f"        ✅ Accepted: Found definite 'there is/are' not in uncertain context")
                return True
            else:
                print(f"        ❌ Rejected: 'there is/are' found in uncertain context")
                return False
        
        # Check for other definite observation patterns
        has_definite = any(pattern in text for pattern in definite_observation_patterns)
        if has_definite:
            print(f"        ✅ Accepted: Found definite observation language")
            return True
        else:
            print(f"        ❌ Rejected: No definite observation language found")
            return False
    
    def _build_observation_subgraph(self, visible_objects: List[Dict], agent_position: np.ndarray) -> Dict:
        """
        Enhanced subgraph construction with spatial relationship inference for panoramic views.
        Addresses the issue of overlapping objects in adjacent viewing directions.
        """
        subgraph = {
            'nodes': [],
            'edges': []
        }
        
        if not visible_objects:
            return subgraph
        
        # === Node Creation with Spatial Clustering ===
        # Group objects by spatial proximity to handle panoramic overlap
        clustered_objects = self._cluster_panoramic_objects(visible_objects)
        
        # Add clustered objects as nodes
        for cluster_id, obj_cluster in enumerate(clustered_objects):
            # Use the highest confidence object in each cluster as representative
            representative_obj = max(obj_cluster, key=lambda x: x.get('confidence', 0.0))
            
            subgraph['nodes'].append({
                'id': representative_obj['name'],
                'confidence': representative_obj.get('confidence', 0.5),
                'position': agent_position.tolist(),
                'direction': representative_obj.get('direction', 0),
                'cluster_size': len(obj_cluster),
                'cluster_id': cluster_id
            })
        
        # === Enhanced Edge Creation with Spatial Relationships ===
        nodes = subgraph['nodes']
        
        if len(nodes) >= 2:
            # Create edges based on spatial relationships inferred from viewing directions
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1, node2 = nodes[i], nodes[j]
                    
                    # Calculate spatial relationship based on viewing directions
                    dir1 = node1.get('direction', 0)
                    dir2 = node2.get('direction', 0)
                    
                    # Compute angular difference
                    angle_diff = abs(dir1 - dir2)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    # Infer spatial relationship
                    relationship, confidence = self._infer_spatial_relationship(
                        node1, node2, angle_diff, agent_position
                    )
                    
                    if relationship and confidence > 0.3:
                        subgraph['edges'].append({
                            'source': node1['id'],
                            'target': node2['id'],
                            'type': relationship,
                            'confidence': confidence,
                            'angle_diff': angle_diff
                        })
        
        # === Add Room Context Edge ===
        # Infer room context and add room-object relationships
        room_context = self._infer_room_context_from_objects([node['id'] for node in nodes])
        if room_context:
            # Add virtual room node
            room_node = {
                'id': f"{room_context}_room",
                'confidence': 0.8,
                'position': agent_position.tolist(),
                'type': 'room_context'
            }
            subgraph['nodes'].append(room_node)
            
            # Connect all objects to room context
            for node in nodes[:-1]:  # Exclude the room node itself
                subgraph['edges'].append({
                    'source': node['id'],
                    'target': room_node['id'],
                    'type': 'located_in',
                    'confidence': 0.7
                })
        
        print(f"📦 Enhanced Subgraph Built: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
        for edge in subgraph['edges']:
            print(f"    {edge['source']} --{edge['type']}--> {edge['target']} (conf: {edge['confidence']:.2f})")
        
        return subgraph
    
    def _cluster_panoramic_objects(self, visible_objects: List[Dict]) -> List[List[Dict]]:
        """
        Cluster objects that appear in multiple panoramic directions to avoid duplicates.
        """
        if not visible_objects:
            return []
        
        clusters = []
        used_objects = set()
        
        for i, obj1 in enumerate(visible_objects):
            if i in used_objects:
                continue
                
            # Start a new cluster
            cluster = [obj1]
            used_objects.add(i)
            
            # Find similar objects in adjacent directions
            for j, obj2 in enumerate(visible_objects):
                if j in used_objects or j == i:
                    continue
                
                # Check if objects are similar (same name) and in adjacent directions
                if self._are_objects_similar_in_panorama(obj1, obj2):
                    cluster.append(obj2)
                    used_objects.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_objects_similar_in_panorama(self, obj1: Dict, obj2: Dict) -> bool:
        """
        Check if two objects are likely the same object seen in adjacent panoramic directions.
        """
        # Same object name
        if obj1.get('name', '').lower() != obj2.get('name', '').lower():
            return False
        
        # Adjacent viewing directions (within 90 degrees)
        dir1 = obj1.get('direction', 0)
        dir2 = obj2.get('direction', 0)
        angle_diff = abs(dir1 - dir2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        return angle_diff <= 90  # Objects in adjacent 60-degree sectors
    
    def _infer_spatial_relationship(self, node1: Dict, node2: Dict, angle_diff: float, 
                                  agent_position: np.ndarray) -> Tuple[str, float]:
        """
        Infer spatial relationship between two objects based on their viewing directions.
        """
        # Very close directions - likely adjacent objects
        if angle_diff < 60:
            return "near", 0.8
        
        # Opposite directions - likely across room from each other
        elif angle_diff > 150:
            return "across_from", 0.7
        
        # Moderate angle difference - general spatial relationship
        elif 60 <= angle_diff <= 120:
            # Use object types to infer more specific relationships
            obj1_name = node1['id'].lower()
            obj2_name = node2['id'].lower()
            
            # Furniture relationships
            if 'table' in obj1_name and 'chair' in obj2_name:
                return "adjacent_to", 0.9
            elif 'bed' in obj1_name and ('nightstand' in obj2_name or 'dresser' in obj2_name):
                return "next_to", 0.9
            elif 'sofa' in obj1_name and 'table' in obj2_name:
                return "facing", 0.8
            else:
                return "positioned_relative_to", 0.6
        
        return None, 0.0
    
    def _infer_room_context_from_objects(self, object_names: List[str]) -> Optional[str]:
        """
        Infer room type based on observed objects using enhanced heuristics.
        """
        if not object_names:
            return None
        
        object_set = {name.lower() for name in object_names}
        
        # Enhanced room detection rules
        room_indicators = {
            'bedroom': {'bed', 'dresser', 'nightstand', 'pillow', 'blanket'},
            'kitchen': {'refrigerator', 'stove', 'counter', 'sink', 'cabinet'},
            'bathroom': {'toilet', 'shower', 'sink', 'bathtub', 'mirror'},
            'living_room': {'sofa', 'tv', 'coffee_table', 'armchair', 'couch'},
            'dining_room': {'dining_table', 'chair', 'chandelier'},
            'office': {'desk', 'computer', 'chair', 'bookshelf'}
        }
        
        # Score each room type
        room_scores = {}
        for room_type, indicators in room_indicators.items():
            score = len(object_set.intersection(indicators)) / len(indicators)
            if score > 0:
                room_scores[room_type] = score
        
        # Return room with highest score if above threshold
        if room_scores:
            best_room = max(room_scores.items(), key=lambda x: x[1])
            if best_room[1] >= 0.3:  # At least 30% of indicators present
                return best_room[0]
        
        return None
    
    def _compute_graph_overlap(self, observation_subgraph: Dict, goal: str) -> Tuple[float, Set[str]]:
        """
        UniGoal-inspired graph matching with three-stage similarity metrics:
        1. Node matching (object overlap)
        2. Edge matching (spatial relationship overlap) 
        3. Topological matching (structural similarity)
        """
        if not self.scene_graph:
            return 0.0, set()
        
        try:
            # Get relevant subgraph from stored scene graph (scene graph Gt)
            goal_lower = goal.lower()
            relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(goal_lower, max_distance=self.max_subgraph_distance)
            
            # Safety check for observation_subgraph structure
            if not isinstance(observation_subgraph, dict) or 'nodes' not in observation_subgraph:
                return 0.0, set()
            
            # Extract nodes and edges from both graphs
            try:
                observed_objects = {node['id'].lower() for node in observation_subgraph['nodes'] if 'id' in node}
                observed_edges = observation_subgraph.get('edges', [])
            except (KeyError, TypeError):
                observed_objects = set()
                observed_edges = []
                
            stored_objects = {name.lower() for name in relevant_nodes.keys()}
            stored_edges = [edge.relation_type for edge in relevant_edges] if relevant_edges else []
            
            # Ensure we have valid sets to work with
            if not observed_objects:
                return 0.0, set()
                
            # === STAGE 1: Node Matching (SN) ===
            exact_matches = observed_objects.intersection(stored_objects)
            node_similarity = 0.0
            if observed_objects:
                # Bipartite matching score - how well observed nodes match stored nodes
                node_similarity = len(exact_matches) / len(observed_objects)
            
            # === STAGE 2: Edge Matching (SE) ===
            edge_similarity = 0.0
            matched_edges = set()
            if observed_edges and stored_edges:
                # Compare spatial relationships
                for obs_edge in observed_edges:
                    obs_relation = f"{obs_edge.get('source', '')}-{obs_edge.get('type', '')}-{obs_edge.get('target', '')}"
                    for stored_edge in stored_edges:
                        # Simple string matching for spatial relationships
                        if obs_relation.lower() in stored_edge.lower() or stored_edge.lower() in obs_relation.lower():
                            matched_edges.add(obs_relation)
                            break
                if observed_edges:
                    edge_similarity = len(matched_edges) / len(observed_edges)
            
            # === STAGE 3: Topological Matching (ST) ===
            topological_similarity = 0.0
            if len(exact_matches) >= 2:
                # If we have anchor pairs (matched nodes), compute structural similarity
                # Simplified version: ratio of matched subgraph structure
                common_subgraph_size = len(exact_matches)
                total_observed_structure = len(observed_objects) + len(observed_edges)
                if total_observed_structure > 0:
                    topological_similarity = common_subgraph_size / total_observed_structure
            
            # === UniGoal Final Score: S = (SN + SE + ST) / 3 ===
            final_score = (node_similarity + edge_similarity + topological_similarity) / 3.0
            
            # Enhanced with goal-specific knowledge
            room_object_knowledge = self._build_enhanced_room_object_knowledge()
            
            # Goal direct match bonus (stronger signal)
            goal_match_bonus = 0.0
            if goal_lower in observed_objects:
                goal_match_bonus = 0.4  # Strong bonus for direct goal observation
                
            # Room-relevance contextual boost
            room_relevance_score = 0.0
            for obj in observed_objects:
                for room, objects in room_object_knowledge.items():
                    if obj in objects and objects[obj] > 0.7:
                        for obj_name, probability in room_object_knowledge.get(room, {}).items():
                            if obj_name == goal_lower and probability > 0.7:
                                room_relevance_score = 0.2
                                break
                        if room_relevance_score > 0:
                            break
                if room_relevance_score > 0:
                    break
                    
            # Combine all scores with appropriate weighting
            combined_score = min(1.0, final_score + goal_match_bonus + room_relevance_score)
            
            # Debug logging for UniGoal metrics
            print(f"🔍 UniGoal Graph Matching Debug:")
            print(f"  Node Similarity (SN): {node_similarity:.3f} ({len(exact_matches)}/{len(observed_objects)} matches)")
            print(f"  Edge Similarity (SE): {edge_similarity:.3f} ({len(matched_edges)}/{len(observed_edges)} matches)")
            print(f"  Topological Similarity (ST): {topological_similarity:.3f}")
            print(f"  Base UniGoal Score: {final_score:.3f}")
            print(f"  Goal Match Bonus: {goal_match_bonus:.3f}")
            print(f"  Room Relevance: {room_relevance_score:.3f}")
            print(f"  Final Combined Score: {combined_score:.3f}")
            
            return combined_score, exact_matches
                
        except Exception as e:
            logging.warning(f"Graph overlap computation error: {e}")
            return 0.0, set()
    
    def _apply_graph_based_reweighting(self, direction_scores: Dict, goal: str, overlap_score: float, 
                                     common_objects: Set[str], agent_position: np.ndarray) -> Dict:
        """
        Re-weight direction scores based on graph knowledge and object associations.
        Enhanced to protect high scores (8-10) from being reduced.
        """
        if not direction_scores:
            return direction_scores
        
        reweighted_scores = direction_scores.copy()
        
        # First: Extract and protect high scores
        high_score_directions = {}
        for direction, data in reweighted_scores.items():
            if isinstance(data, dict) and 'Score' in data and data['Score'] >= 8:
                # Protect high scores from reweighting
                high_score_directions[direction] = data['Score']
        
        # Apply reweighting based on overlap score (if score is not already high)
        if overlap_score > 0.3:
            goal_related_directions = self._find_goal_related_directions(goal, common_objects)
            
            for direction, data in reweighted_scores.items():
                # Skip non-direction keys like 'graph_confidence'
                if not (direction.isdigit() or direction in ['30', '90', '150', '210', '270', '330']):
                    continue
                    
                # Skip high score directions
                if direction in high_score_directions:
                    continue
                    
                try:
                    direction_angle = float(direction)
                    
                    # Boost score if this direction contains goal-related objects
                    if direction_angle in goal_related_directions:
                        boost_factor = 1.0 + (overlap_score * 0.5)
                        if isinstance(data, dict) and 'Score' in data:
                            data['Score'] = min(8.0, data['Score'] * boost_factor)
                    
                    # Apply room-based knowledge
                    room_boost = self._calculate_room_based_boost(direction_angle, goal, agent_position)
                    if room_boost > 1.0 and isinstance(data, dict) and 'Score' in data:
                        data['Score'] = min(8.0, data['Score'] * room_boost)
                        
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
        
    def _format_direction_thought(self, thought_content, angle, goal=None):
        """Format direction thought with better handling of negative mentions."""
        # First, ensure we don't try to access None.lower()
        if not thought_content:
            return f"Direction {angle}: No useful information"
            
        # Check for room types and goal mentions
        room_indicators = ["bedroom", "bathroom", "kitchen", "living room", "hallway", "dining room", "office"]
        has_room_type = any(room in thought_content.lower() for room in room_indicators)
        
        # Add room type if missing
        if not has_room_type:
            thought_content += ". Room type unclear."
        return thought_content
    
    def _update_scene_graph_with_predictions(self, predictions: dict, agent_position: np.ndarray, step: int):
        """Enhanced scene graph update that preserves spatial relationships"""
        if not self.scene_graph or not self.thought_parser:
            return
        
        try:
            # Store thoughts for later processing
            self.direction_thoughts = {}
            
            # Extract angle-based thought dictionary format returned by _eval_response
            if predictions and all(k.isdigit() or k in ['30', '90', '150', '210', '270', '330'] for k in predictions.keys()):
                for angle, data in predictions.items():
                    if isinstance(data, dict) and 'Thought' in data:
                        thought_content = data['Thought']
                        #formatted_thought = self._format_direction_thought(thought_content, angle, goal=None)
                        self.direction_thoughts[angle] = thought_content

            # Process direction-specific thoughts to update scene graph
            if hasattr(self, 'direction_thoughts') and self.direction_thoughts:
                try:
                    # Process each thought separately
                    for direction_str, thought in self.direction_thoughts.items():
                        try:
                            direction_angle = float(direction_str)
                            parsed_data = self.thought_parser.parse_thought(thought, direction_angle, step)
                            self.thought_parser.update_scene_graph(self.scene_graph, parsed_data, agent_position, step)
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Error processing direction {direction_str}: {e}")
                    
                    # Add path node to track agent's movement
                    self.scene_graph.add_node(
                        f"path_point_{step}",
                        {"type": "path", "step": step},
                        position=agent_position,
                        confidence=1.0,
                        step=step
                    )
                    
                    # Connect to previous path point if it exists
                    if step > 0 and f"path_point_{step-1}" in self.scene_graph.nodes:
                        self.scene_graph.add_edge(
                            f"path_point_{step}",
                            f"path_point_{step-1}",
                            "follows",
                            confidence=1.0
                        )
                except Exception as e:
                    logging.error(f"Error parsing direction thoughts: {e}")
            
            # Update graph metadata
            self.graph_update_count += 1
            
        except Exception as e:
            logging.warning(f"Scene graph update error: {e}")
    
    def _update_object_associations(self, predictions: dict, goal: str, agent_position: np.ndarray):
        """Update object associations and goal-object relationships in the scene graph."""
        if not self.scene_graph:
            return
        
        try:
            # Add goal-object association edges
            goal_node = self.scene_graph.nodes.get(goal.lower())
            if goal_node:
                # Find related objects in current predictions
                visible_objects = self._extract_objects_from_predictions(predictions)
                for obj_data in visible_objects:
                    obj_name = obj_data['name']
                    if obj_name != goal.lower():
                        # Add "near" relationship
                        self.scene_graph.add_edge(goal.lower(), obj_name, "possibly_near", 0.6)
            
        except Exception as e:
            logging.warning(f"Object association update error: {e}")
    
    def _extract_and_add_objects_from_text(self, text: str, agent_position: np.ndarray, step: int):
        """
        Extract only confirmed observed objects from text and add to scene graph.
        Handles negative context carefully to avoid adding non-existent objects.
        """
        text_lower = text.lower()
        
        # Common objects we might detect
        common_objects = [
            "bed", "sofa", "chair", "table", "desk", "tv", "refrigerator",
            "stove", "sink", "toilet", "bathtub", "shower", "nightstand",
            "dresser", "couch", "bookshelf", "cabinet", "wardrobe"
        ]
        
        # Negative context patterns that should block object detection
        negative_patterns = [
            "no visible", "not visible", "no sign of", "no strong indication", 
            "not found", "not present", "not in sight", "no clear", "unlikely", 
            "cannot see", "can't see", "doesn't have", "does not have", 
            "might be", "could be", "possibly", "probably", "appears to be",
            "seems like", "looks like there might be"
        ]
        
        # Confirmation patterns that strongly suggest the object is actually observed
        confirmation_patterns = [
            "clearly visible", "can see a", "there is a", "i can see", 
            "definitely a", "visible in", "prominently displayed", 
            "observed", "present in", "in view"
        ]
        
        for obj in common_objects:
            # Check if object is mentioned
            if obj in text_lower:
                # First check if object is in negative context
                negative_context = False
                for pattern in negative_patterns:
                    if pattern in text_lower and abs(text_lower.find(pattern) - text_lower.find(obj)) < 50:
                        negative_context = True
                        break
                
                # Then check if object is in confirmation context
                confirmed = False
                for pattern in confirmation_patterns:
                    if pattern in text_lower and abs(text_lower.find(pattern) - text_lower.find(obj)) < 30:
                        confirmed = True
                        break
                        
                # Only add objects that are confirmed and not in negative context
                if confirmed and not negative_context:
                    self.scene_graph.add_node(obj, {"type": "object"}, agent_position, 0.8, step)
                elif not negative_context:
                    # Add with lower confidence if mentioned but not strongly confirmed
                    self.scene_graph.add_node(obj, {"type": "object"}, agent_position, 0.4, step)
                else:
                    # Log but don't add objects in negative context
                    logging.debug(f"Skipping object in negative context: {obj}")
    
    def _get_room_probabilities_for_goal(self, goal: str, relevant_nodes: Dict) -> Dict[str, float]:
        """Calculate room probabilities for containing the goal object."""
        knowledge = self._build_enhanced_room_object_knowledge()
        room_probs = {}
        
        for room_type, obj_probs in knowledge.items():
            prob = obj_probs.get(goal.lower(), 0.1)
            if prob > 0.3:  # Only include likely rooms
                room_probs[room_type] = prob
        
        return room_probs
    
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
    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal):
        """Enhanced planning with direct goal detection prioritization"""
        try:
            # Check if any prediction had score≥9 AND strategy allows goal detection
            goal_detected = False
            goal_direction = None
            goal_thought = None
            
            print(f"🔍 Planning Module Goal Detection Debug:")
            print(f"  Has last_predictions: {hasattr(self, 'last_predictions')}")
            
            if hasattr(self, 'last_predictions') and isinstance(self.last_predictions, dict):
                print(f"  Last predictions keys: {list(self.last_predictions.keys())}")
                
                # Check exploration strategy - only allow goal detection for DIRECT_NAVIGATION
                exploration_strategy = self.last_predictions.get('exploration_strategy', 'FIND_NEW_AREAS')
                allow_goal_detection = exploration_strategy == "DIRECT_NAVIGATION"
                print(f"  Exploration strategy: {exploration_strategy}")
                print(f"  Allow goal detection: {allow_goal_detection}")
                
                for direction, data in self.last_predictions.items():
                    if isinstance(data, dict):
                        score = data.get('Score', 0)
                        # Convert score to int if it's a string  
                        if isinstance(score, str):
                            try:
                                score = int(score)
                            except (ValueError, TypeError):
                                score = 0
                        
                        print(f"  Direction {direction}: score={score} (≥9? {score >= 9})")
                        
                        if score >= 9 and allow_goal_detection:  # Both score and strategy must allow
                            goal_detected = True
                            goal_direction = direction
                            goal_thought = data.get('Thought', 'Goal object clearly visible')
                            print(f"  ✅ Goal detected at direction {direction}")
                            break
            
            print(f"  Final goal_detected: {goal_detected}")
                        
            # Generate planning prompt
            planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason)
            planning_response = self.PlanVLM.call([planning_image], planning_prompt)
            
            # Log for debugging
            print(f"Planning prompt: {planning_prompt[:200]}...")
            print(f"Planning raw response: {planning_response}")
            
            # Parse response
            dct = self._eval_response(planning_response, goal)
            
            # Force Flag=True if goal was detected with high confidence
            if goal_detected:
                dct['Flag'] = True
                dct['Subtask'] = f'Go to the {goal}'
                print(f"🎯 GOAL OVERRIDE: Goal detected with high confidence - forcing navigation to {goal}")
            logging.info(f"Planning response: {dct}")
            return dct
            
        except Exception as e:
            print(f"Planning module error: {e}")
            return {'Flag': False, 'Subtask': '{}'}
        
    def step(self, obs: dict):
        """Enhanced step function with optimization tracking."""
        # Update agent position tracking
        if 'agent_state' in obs:
            self.last_agent_state = obs['agent_state']
        
        # Update path history for trajectory analysis
        if hasattr(self, 'last_agent_state') and self.last_agent_state:
            position = self.last_agent_state.position
            # Ensure position is stored as numpy array
            if not isinstance(position, np.ndarray):
                position = np.array(position)
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

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask
    
    def make_curiosity_value(self, pano_images, goal):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles)

        response = self._predicting_module(inference_image, goal)

        explorable_value = {}
        reason = {}

        try:
            # Filter out metadata keys added by graph processing
            # Only process angle keys (should be numeric or angle strings like "30", "90", etc.)
            valid_angle_keys = []
            for key in response.keys():
                # Check if key represents an angle (numeric string or int)
                try:
                    angle_val = float(key)
                    valid_angle_keys.append(key)
                except (ValueError, TypeError):
                    # Skip non-angle keys like 'exploration_strategy', 'graph_confidence', etc.
                    continue
            
            for angle in valid_angle_keys:
                values = response[angle]
                if isinstance(values, dict):
                    explorable_value[angle] = values.get('Score', 0)
                    reason[angle] = values.get('Thought', '')
                else:
                    # Fallback for unexpected data structure
                    explorable_value[angle] = 0
                    reason[angle] = ''
                    
        except Exception as e:
            print(f"⚠️  Error in make_curiosity_value: {e}")
            print(f"Response keys: {list(response.keys()) if response else 'None'}")
            explorable_value, reason = None, None
            
        logging.info(f"Explorable value: {explorable_value}")
        logging.info(f"Reasoning: {reason}")

        return inference_image, explorable_value, reason

    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0):

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
            evaluator_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) If the {goal} is found, assign a score of 10. Make sure the object you see is ACTUALLY a {goal}, not something that looks similar. '
            f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
            "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Thought': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
            "Answer Example: {'30': {'Score': 0, 'Thought': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Thought': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Thought': 'Living room area with a recliner.  Similar to 270, there is possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
            
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
    
    def _apply_unigoal_strategy_reweighting(self, predictions: dict, strategy: str, goal: str, 
                                          overlap_score: float, common_objects: set, 
                                          agent_position: np.ndarray) -> dict:
        """
        Apply UniGoal-inspired multi-stage reweighting strategies based on graph matching results.
        """
        reweighted_predictions = predictions.copy()
        
        print(f"🎯 Applying UniGoal Strategy: {strategy}")
        
        if strategy == "DIRECT_NAVIGATION":
            # Stage 3: Perfect Matching - Navigate directly to goal
            print("  -> Stage 3 reweighting: Boosting goal directions")
            goal_directions = self._find_goal_related_directions(goal, {goal.lower()})
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and float(direction) in goal_directions:
                    if isinstance(data, dict) and 'Score' in data:
                        data['Score'] = 10.0  # Max score for goal detection
                        print(f"    ✅ Boosted direction {direction} to score 10.0")
        
        elif strategy == "GOAL_VERIFICATION":
            # Stage 3 verification phase - moderate boost but verify with visual features
            print("  -> Stage 3 verification: Moderate goal boost with verification")
            goal_directions = self._find_goal_related_directions(goal, {goal.lower()})
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and float(direction) in goal_directions:
                    if isinstance(data, dict) and 'Score' in data:
                        # Moderate boost but below automatic goal detection threshold
                        data['Score'] = min(8.5, data['Score'] * 1.3)
                        print(f"    ⚖️ Verification boost for direction {direction}: {data['Score']}")
        
        elif strategy == "COORDINATE_PROJECTION":
            # Stage 2: Project coordinates based on anchor pairs
            print("  -> Stage 2 coordinate projection: Using spatial relationships")
            projected_directions = self._project_goal_coordinates(common_objects, goal, agent_position)
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and float(direction) in projected_directions:
                    if isinstance(data, dict) and 'Score' in data:
                        # Strong boost for projected goal locations
                        data['Score'] = min(8.0, data['Score'] * 1.6)
                        print(f"    📍 Coordinate projection boost for direction {direction}: {data['Score']}")
        
        elif strategy == "FOCUSED_SEARCH":
            # Stage 2: Partial matching - search around known anchor points
            print("  -> Stage 2 focused search: Searching around anchor objects")
            reweighted_predictions = self._reweight_predictions_with_graph_matching(
                reweighted_predictions, goal, agent_position
            )
            # Additional boost for directions with anchor objects
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and isinstance(data, dict) and 'Score' in data:
                    thought = data.get('Thought', '').lower()
                    # Boost if thought mentions any common objects
                    for common_obj in common_objects:
                        if common_obj.lower() in thought:
                            data['Score'] = min(7.5, data['Score'] * 1.4)
                            print(f"    🔍 Anchor object boost for direction {direction}: {data['Score']}")
                            break
        
        elif strategy == "SEMANTIC_FRONTIER_EXPLORATION":
            # Stage 1: Zero matching with semantic guidance
            print("  -> Stage 1 semantic frontiers: Goal-guided exploration")
            # Apply semantic knowledge to guide exploration
            room_knowledge = self._build_enhanced_room_object_knowledge()
            goal_lower = goal.lower()
            
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and isinstance(data, dict) and 'Score' in data:
                    thought = data.get('Thought', '').lower()
                    
                    # Boost directions leading to rooms likely to contain the goal
                    for room_type, obj_probs in room_knowledge.items():
                        if room_type in thought and goal_lower in obj_probs:
                            goal_prob = obj_probs[goal_lower]
                            if goal_prob > 0.5:
                                boost_factor = 1.0 + (goal_prob * 0.6)
                                data['Score'] = min(6.5, data['Score'] * boost_factor)
                                print(f"    🧭 Semantic guidance boost for direction {direction}: {data['Score']}")
                                break
        
        elif strategy == "FIND_NEW_AREAS":
            # Stage 1: Zero matching - pure exploration
            print("  -> Stage 1 exploration: Finding new areas")
            reweighted_predictions = self._reweight_predictions_with_graph_matching(
                reweighted_predictions, goal, agent_position
            )
            
            # Boost unexplored directions but stay below goal detection threshold
            unexplored_directions = self._find_unexplored_directions()
            for direction, data in reweighted_predictions.items():
                if direction.isdigit() and float(direction) in unexplored_directions:
                    if isinstance(data, dict) and 'Score' in data:
                        # Conservative boost to prevent false goal detection
                        data['Score'] = min(7.0, data['Score'] * 1.3)
                        print(f"    🗺️ Exploration boost for direction {direction}: {data['Score']}")
        
        else:
            # Fallback to standard graph-based reweighting
            print(f"  -> Fallback: Standard graph reweighting for strategy {strategy}")
            reweighted_predictions = self._reweight_predictions_with_graph_matching(
                reweighted_predictions, goal, agent_position
            )
        
        return reweighted_predictions
    
    def _project_goal_coordinates(self, anchor_objects: set, goal: str, agent_position: np.ndarray) -> set:
        """
        Project likely goal coordinates based on anchor objects and spatial relationships.
        Implements UniGoal Stage 2 coordinate projection strategy.
        """
        projected_directions = set()
        
        if not anchor_objects or not self.scene_graph:
            return projected_directions
        
        try:
            # Get spatial relationships for anchor objects
            for anchor_obj in anchor_objects:
                # Find stored spatial relationships involving this anchor
                anchor_node = self.scene_graph.nodes.get(anchor_obj.lower())
                if not anchor_node:
                    continue
                
                # Look for edges that might indicate goal location relative to anchor
                for edge_id, edge_data in self.scene_graph.edges.items():
                    source_node = edge_data.get('source', '')
                    target_node = edge_data.get('target', '')
                    relation_type = edge_data.get('type', '')
                    
                    # Check if this edge involves our anchor and potentially the goal
                    if (source_node == anchor_obj.lower() and goal.lower() in target_node) or \
                       (target_node == anchor_obj.lower() and goal.lower() in source_node):
                        
                        # Infer direction based on spatial relationship
                        inferred_directions = self._spatial_relation_to_directions(
                            relation_type, agent_position
                        )
                        projected_directions.update(inferred_directions)
        
        except Exception as e:
            print(f"⚠️ Error in coordinate projection: {e}")
        
        return projected_directions
    
    def _spatial_relation_to_directions(self, relation_type: str, agent_position: np.ndarray) -> set:
        """
        Convert spatial relationship to likely viewing directions.
        """
        directions = set()
        relation_lower = relation_type.lower()
        
        # Map spatial relationships to directional preferences
        if 'left' in relation_lower:
            directions.update({270.0, 330.0})
        elif 'right' in relation_lower:
            directions.update({90.0, 150.0})
        elif 'front' in relation_lower or 'ahead' in relation_lower:
            directions.update({30.0, 90.0, 330.0})
        elif 'behind' in relation_lower or 'back' in relation_lower:
            directions.update({150.0, 210.0, 270.0})
        elif 'near' in relation_lower or 'adjacent' in relation_lower:
            # Objects are nearby - check all adjacent directions
            directions.update({30.0, 90.0, 150.0, 210.0, 270.0, 330.0})
        elif 'across' in relation_lower:
            # Opposite side - check opposite directions
            directions.update({150.0, 210.0, 270.0})
        
        return directions
    
    def _has_goal_relevant_context(self, goal: str) -> bool:
        """
        Check if we have any goal-relevant semantic context from previous observations.
        """
        if not self.scene_graph or not hasattr(self, 'direction_thoughts'):
            return False
        
        goal_lower = goal.lower()
        
        # Check if goal or related objects mentioned in recent thoughts
        for thought in self.direction_thoughts.values():
            if goal_lower in thought.lower():
                return True
        
        # Check if goal-related objects exist in scene graph
        room_knowledge = self._build_enhanced_room_object_knowledge()
        for room_type, obj_probs in room_knowledge.items():
            if goal_lower in obj_probs and obj_probs[goal_lower] > 0.5:
                # Check if any objects from this room type are in scene graph
                for stored_obj in self.scene_graph.nodes.keys():
                    if stored_obj.lower() in obj_probs:
                        return True
        
        return False