import numpy as np
import habitat_sim
import logging
import cv2
import time
import os
from typing import Dict, List, Tuple, Any, Optional

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
    Stores observations in a structured scene graph rather than raw text.
    """
    
    def __init__(self, cfg: dict):
        print("🔥 GraphMemoryAgent (FULL VERSION) initializing...")
        logging.info("GraphMemoryAgent (FULL VERSION) initializing...")
        
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
        
        # NOW call parent initialization
        super().__init__(cfg)
        
        print("🎯 GraphMemoryAgent (FULL VERSION) initialized successfully!")
        logging.info("GraphMemoryAgent (FULL VERSION) initialized successfully!")
    
    def reset(self):
        super().reset()
        print("🔄 GraphMemoryAgent (FULL VERSION) resetting...")
        logging.info("GraphMemoryAgent (FULL VERSION) resetting...")
        
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
        print("🎯 GraphMemoryAgent (FULL VERSION) reset complete!")
    
    def _predicting_module(self, evaluator_image, goal):
        """
        Enhanced prediction module that efficiently stores structured thoughts as a scene graph.
        Optimized for non-redundant processing and SOTA performance.
        """
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        
        # Parse the response into a dictionary
        dct = self._eval_response(evaluator_response)
        
        # Efficiently update scene graph if available and enabled
        if (self.scene_graph is not None and self.thought_parser is not None and 
            self.graph_enabled and hasattr(self, 'step_ndx')):
            
            # Extract and filter structured thoughts
            high_quality_thoughts = {}
            for angle, values in dct.items():
                if isinstance(values, dict) and 'Thought' in values:
                    thought = values['Thought']
                    # Only process thoughts that contain meaningful content
                    if thought and len(thought.strip()) > 10:  # Avoid empty or trivial thoughts
                        high_quality_thoughts[angle] = thought
            
            # Batch process thoughts for efficiency
            if high_quality_thoughts:
                agent_position = None
                if hasattr(self, 'last_agent_state') and self.last_agent_state is not None:
                    agent_position = np.array(self.last_agent_state.position)
                
                processed_count = 0
                for angle, thought in high_quality_thoughts.items():
                    try:
                        angle_float = float(angle)
                        parsed_data = self.thought_parser.parse_thought(thought, angle_float, self.step_ndx)
                        
                        # Only update scene graph if we extracted meaningful data
                        # 修复：安全检查数据，避免numpy数组真值模糊错误
                        has_nodes = False
                        has_edges = False
                        try:
                            nodes_data = parsed_data.get('nodes', [])
                            edges_data = parsed_data.get('edges', [])
                            
                            # 安全检查nodes和edges
                            if isinstance(nodes_data, (list, tuple)):
                                has_nodes = len(nodes_data) > 0
                            elif hasattr(nodes_data, '__len__'):
                                has_nodes = len(nodes_data) > 0
                            else:
                                has_nodes = bool(nodes_data)
                                
                            if isinstance(edges_data, (list, tuple)):
                                has_edges = len(edges_data) > 0
                            elif hasattr(edges_data, '__len__'):
                                has_edges = len(edges_data) > 0
                            else:
                                has_edges = bool(edges_data)
                        except Exception as e:
                            print(f"⚠️  Error checking parsed_data content: {e}")
                            logging.warning(f"Error checking parsed_data content: {e}")
                            has_nodes = has_edges = False
                        
                        if has_nodes or has_edges:
                            self.thought_parser.update_scene_graph(
                                self.scene_graph, parsed_data, agent_position, self.step_ndx
                            )
                            processed_count += 1
                            
                    except Exception as e:
                        print(f"Warning: Could not parse thought for angle {angle}: {e}")
                        logging.warning(f"Could not parse thought for angle {angle}: {e}")
                
                if processed_count > 0:
                    print(f"🧠 Updated scene graph with {processed_count} high-quality observations")
                    logging.info(f"Updated scene graph with {processed_count} high-quality observations")
        
        return dct
    
    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal, scene_graph_context=None):
        """
        Enhanced planning module that uses provided scene graph context or generates its own.
        """
        print(f"🧠 Planning module called with goal: {goal}")
        logging.info(f"Planning module called with goal: {goal}")
        
        # Use provided scene graph context or generate one
        if scene_graph_context:
            subgraph_text = scene_graph_context
            logging.info(f"Using provided scene graph context: {len(scene_graph_context)} chars")
            logging.info(f"graph_memory_context: {scene_graph_context[:200]}...")  # Preview first 200 chars
        else:
            # Fallback: generate scene graph context if not provided
            subgraph_text = ""
            relevant_nodes_count = 0
            relevant_edges_count = 0
            
            if self.scene_graph is not None and self.graph_enabled:
                try:
                    # Extract goal-relevant subgraph
                    relevant_nodes, relevant_edges = self.scene_graph.get_subgraph_for_goal(
                        goal, max_distance=self.max_subgraph_distance
                    )
                    
                    # 修复：确保 confidence 比较使用标量值，避免数组比较问题
                    filtered_nodes = {}
                    for name, node in relevant_nodes.items():
                        try:
                            # 修复数组比较问题
                            if hasattr(node.confidence, '__len__'):  # 如果是数组
                                confidence_val = float(np.mean(node.confidence))
                            else:
                                confidence_val = float(node.confidence)
                            
                            # 安全进行阈值比较
                            if confidence_val >= self.confidence_threshold:
                                filtered_nodes[name] = node
                        except Exception as e:
                            print(f"Error processing node confidence for {name}: {e}")
                            logging.warning(f"Error processing node confidence for {name}: {e}")
                            # 在出错的情况下，保守地包含节点
                            filtered_nodes[name] = node
                    
                    filtered_edges = []
                    for edge in relevant_edges:
                        try:
                            # 修复数组比较问题
                            if hasattr(edge.confidence, '__len__'):  # 如果是数组
                                confidence_val = float(np.mean(edge.confidence))
                            else:
                                confidence_val = float(edge.confidence)
                            
                            # 安全进行阈值比较
                            if confidence_val >= self.confidence_threshold:
                                filtered_edges.append(edge)
                        except Exception as e:
                            print(f"⚠️  Error processing edge confidence: {e}")
                            logging.warning(f"Error processing edge confidence: {e}")
                            # 在出错的情况下，保守地包含边
                            filtered_edges.append(edge)
                    
                    # Generate comprehensive scene graph context
                    if filtered_nodes:
                        subgraph_text = self._subgraph_to_text(filtered_nodes, filtered_edges)
                        relevant_nodes_count = len(filtered_nodes)
                        relevant_edges_count = len(filtered_edges)
                        
                        print(f"Extracted high-confidence subgraph: {relevant_nodes_count} nodes, {relevant_edges_count} edges")
                        logging.info(f"Extracted high-confidence subgraph: {relevant_nodes_count} nodes, {relevant_edges_count} edges")
                    else:
                        print("No high-confidence scene graph context, using exploration guidance")
                        logging.warning("No high-confidence scene graph context, using exploration guidance")
                        
                except Exception as e:
                    print(f"Error extracting scene graph context: {e}")
                    logging.warning(f"Error extracting scene graph context: {e}")
                    subgraph_text = ""
        
        # Construct enhanced planning prompt
        planning_prompt = self._construct_prompt(
            goal, 'planning', 
            subtask=previous_subtask, 
            reason=goal_reason,
            scene_graph_context=subgraph_text
        )
        
        # ALWAYS add scene graph context for SOTA performance
        try:
            # 安全检查 subgraph_text 是否有效
            has_valid_text = False
            if subgraph_text:
                if isinstance(subgraph_text, str):
                    has_valid_text = len(subgraph_text.strip()) > 0
                elif hasattr(subgraph_text, '__len__'):
                    # Handle case where subgraph_text might be an array
                    has_valid_text = False
                else:
                    has_valid_text = bool(subgraph_text)
            
            if has_valid_text:
                # Context already added in _construct_prompt
                context_len = len(subgraph_text)
                print(f"Scene graph context added to planning prompt ({context_len} chars)")
                print(f"graph_memory_context: {subgraph_text[:200]}...")  # Preview first 200 chars
                logging.info(f"Scene graph context added to planning prompt ({context_len} chars)")
                logging.info(f"graph_memory_context: {subgraph_text[:200]}...")
            else:
                print(f"No high-confidence scene graph context, using exploration guidance")
                logging.warning("No high-confidence scene graph context, using exploration guidance")
        except Exception as e:
            print(f"Error in scene graph text processing: {e}")
            logging.warning(f"Error in scene graph text processing: {e}")
            planning_prompt += f"\n\n=== EXPLORATION GUIDANCE ===\nNo previous observations of {goal}. Focus on systematic exploration of likely locations.\n=== END GUIDANCE ===\n"
        
        # Call the planning VLM with enhanced context
        try:
            # 安全处理 planning_image，避免数组比较问题
            image_to_send = None
            try:
                if planning_image is not None:
                    if isinstance(planning_image, list):
                        if len(planning_image) > 0:
                            if isinstance(planning_image[0], np.ndarray):
                                image_to_send = [planning_image[0]]
                            else:
                                image_to_send = planning_image
                        else:
                            image_to_send = []
                    elif isinstance(planning_image, np.ndarray):
                        image_to_send = [planning_image]
                    else:
                        image_to_send = planning_image
                else:
                    image_to_send = []
            except Exception as e:
                print(f"⚠️  Error processing planning_image: {e}")
                logging.warning(f"Error processing planning_image: {e}")
                image_to_send = []
                
            planning_response = self.PlanVLM.call(image_to_send, planning_prompt)
            
            # 安全处理 planning_response，避免数组真值模糊错误
            try:
                if isinstance(planning_response, str):
                    response_preview = planning_response[:200]
                elif hasattr(planning_response, '__str__'):
                    response_str = str(planning_response)
                    response_preview = response_str[:200]
                else:
                    response_preview = f"<{type(planning_response).__name__} object>"
                print(f"📝 Planning response received: {response_preview}...")
                logging.info(f"Planning response received: {response_preview}...")
            except Exception as e:
                print(f"⚠️  Error processing planning response preview: {e}")
                logging.warning(f"Error processing planning response preview: {e}")
            
            # Process the response - 确保是字符串
            try:
                if not isinstance(planning_response, str):
                    planning_response = str(planning_response)
                planning_response = planning_response.replace('false', 'False').replace('true', 'True')
            except Exception as e:
                print(f"⚠️  Error processing planning response string: {e}")
                logging.warning(f"Error processing planning response string: {e}")
                planning_response = "{}"  # 回退到空字典字符串
            
            dct = self._eval_response(planning_response)
            
            # Validate the response format - 安全检查避免数组比较
            is_valid_dict = False
            has_flag = False
            has_subtask = False
            
            try:
                is_valid_dict = isinstance(dct, dict)
                if is_valid_dict:
                    has_flag = 'Flag' in dct
                    has_subtask = 'Subtask' in dct
            except Exception as e:
                print(f"Error validating response format: {e}")
                logging.warning(f"Error validating response format: {e}")
                
            if not (is_valid_dict and has_flag and has_subtask):
                print(f"Invalid planning response format: {dct}")
                logging.warning(f"Invalid planning response format: {dct}")
                return {'Flag': False, 'Subtask': '{}'}
            
            # Log planning decision for debugging - 安全获取值
            try:
                flag_val = dct.get('Flag', False)
                subtask_val = dct.get('Subtask', '{}')
                print(f"Planning decision: Flag={flag_val}, Subtask={subtask_val}")
                logging.info(f"Planning decision: Flag={flag_val}, Subtask={subtask_val}")
            except Exception as e:
                print(f"Error logging planning decision: {e}")
                logging.warning(f"Error logging planning decision: {e}")
                
            return dct
            
        except Exception as e:
            print(f"Planning module error: {e}")
            logging.error(f"Planning module error: {e}")
            return {'Flag': False, 'Subtask': '{}'}
    
    def _subgraph_to_text(self, nodes, edges) -> str:
        """Convert subgraph nodes and edges to text representation with enhanced context."""
        if not nodes and not edges:
            return ""
        
        text_parts = []
        
        # Add node information with confidence scores
        if nodes:
            text_parts.append("Observed Objects:")
            # Sort nodes by confidence for better planning - fix numpy array comparison
            try:
                sorted_nodes = sorted(nodes.items(), key=lambda x: float(np.mean(x[1].confidence)) if hasattr(x[1].confidence, '__len__') else float(x[1].confidence), reverse=True)
            except:
                sorted_nodes = list(nodes.items())  # Fallback to unsorted if sorting fails
                
            for node_name, node in sorted_nodes:
                attrs = ", ".join([f"{k}: {v}" for k, v in node.attributes.items() if v])
                # Handle confidence display safely
                try:
                    if hasattr(node.confidence, '__len__'):
                        confidence_val = float(np.mean(node.confidence))
                    else:
                        confidence_val = float(node.confidence)
                    confidence_str = f"(confidence: {confidence_val:.2f})" if confidence_val > 0 else ""
                except:
                    confidence_str = ""
                last_seen_str = f"(last seen: step {node.last_seen})" if node.last_seen > 0 else ""
                text_parts.append(f"- {node_name}: {attrs} {confidence_str} {last_seen_str}")
        
        # Add spatial relationships with confidence
        if edges:
            text_parts.append("\nSpatial Relationships:")
            # Sort edges by confidence for better planning - fix numpy array comparison
            try:
                sorted_edges = sorted(edges, key=lambda x: float(np.mean(x.confidence)) if hasattr(x.confidence, '__len__') else float(x.confidence), reverse=True)
            except:
                sorted_edges = list(edges)  # Fallback to unsorted if sorting fails
            for edge in sorted_edges:
                # Handle confidence display safely
                try:
                    if hasattr(edge.confidence, '__len__'):
                        confidence_val = float(np.mean(edge.confidence))
                    else:
                        confidence_val = float(edge.confidence)
                    confidence_str = f"(confidence: {confidence_val:.2f})" if confidence_val > 0 else ""
                except:
                    confidence_str = ""
                text_parts.append(f"- {edge.source} {edge.relation_type} {edge.target} {confidence_str}")
        
        return "\n".join(text_parts)
    
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
        """Enhanced step method that updates agent position for scene graph."""
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
    
    def _construct_prompt(self, goal: str, prompt_type: str, subtask: str = '{}', 
                        reason: str = '{}', num_actions: int = 0, scene_graph_context: str = None):
        """Enhanced prompt construction with scene graph context."""
        return super()._construct_prompt(goal, prompt_type, subtask, reason, num_actions, scene_graph_context)
    
    def draw_scene_graph(self, agent_state: habitat_sim.AgentState = None, zoom: int = 9) -> np.ndarray:
        """Draw the scene graph for visualization."""
        try:
            import tempfile
            # Use a temporary file for graph generation
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
            
            # Generate visualization if we have scene graph data
            if self.scene_graph is not None and len(self.scene_graph.nodes) > 0:
                # Generate the graph visualization
                self.scene_graph.visualize(save_path=temp_path)
                
                # Load the saved image
                graph_image = cv2.imread(temp_path)
                
                # Clean up temp file after loading
                os.remove(temp_path)
                
                # Add step number to the image
                if graph_image is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    step_text = f'step {getattr(self, "step_ndx", 0)}'
                    cv2.putText(graph_image, step_text, (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return graph_image
            
            # Default empty image if no graph or visualization failed
            empty_size = 800
            empty_image = np.zeros((empty_size, empty_size, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Add step number
            step_text = f'step {getattr(self, "step_ndx", 0)}'
            cv2.putText(empty_image, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add message if no graph data
            if self.scene_graph is None or len(self.scene_graph.nodes) == 0:
                cv2.putText(empty_image, "No scene graph data available", 
                          (empty_size//6, empty_size//2), 
                          font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            return empty_image
                
        except Exception as e:
            print(f"Warning: Could not visualize scene graph: {e}")
            logging.warning(f"Could not visualize scene graph: {e}")
            
            # Return empty image with error message
            empty_size = 800
            empty_image = np.zeros((empty_size, empty_size, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Add error message
            cv2.putText(empty_image, "Scene Graph Visualization Error", 
                      (empty_size//6, empty_size//2), 
                      font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            return empty_image
    
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
        This prevents the graph from becoming too large and maintains SOTA performance.
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
        """
        Enhanced stopping module for GraphMemoryAgent that uses planner's decision.
        Overrides the parent's distance-based stopping logic with planner-based decisions.
        """
        print(f"🛑 GraphMemoryAgent._stopping_module called")
        logging.info(f"GraphMemoryAgent._stopping_module called")
        
        # Check if planner indicates we've reached the goal
        goal_flag_from_planner = obs.get('goal_flag_from_planner', False)
        subtask_from_planner = obs.get('subtask_from_planner', {})
        
        # Debug logging
        logging.info(f"🔍 Stopping module check: goal_flag={goal_flag_from_planner}, subtask={subtask_from_planner}")
        print(f"🔍 Stopping module check: goal_flag={goal_flag_from_planner}, subtask={subtask_from_planner}")
        
        # Enhanced stopping condition: 
        # Stop if planner says we found the goal AND provides no further subtask
        if goal_flag_from_planner:
            print(f"🎯 Goal flag is TRUE, checking subtask: {subtask_from_planner}")
            logging.info(f"Goal flag is TRUE, checking subtask: {subtask_from_planner}")
            
            # If subtask is empty dict {} or empty, we should stop
            if not subtask_from_planner or subtask_from_planner == {}:
                logging.info("✅ STOPPING: Goal found and no further subtask specified.")
                print("✅ STOPPING: Goal found and no further subtask specified.")
                return True
            else:
                logging.info(f"❌ NOT STOPPING: Goal found but subtask provided: {subtask_from_planner}")
                print(f"❌ NOT STOPPING: Goal found but subtask provided: {subtask_from_planner}")
                return False
        else:
            print(f"❌ Goal flag is FALSE")
            logging.info(f"Goal flag is FALSE")
        
        # Fallback to parent's distance-based stopping as backup
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