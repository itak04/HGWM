import re
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scene_graph import SceneGraph

class ThoughtParser:
    """
    Parser for converting structured VLM reasoning into scene graph components.
    Extracts objects, spatial relationships, and navigation insights from 
    Embodied-Reasoner style structured thoughts.
    """
    
    def __init__(self):
        # Spatial relation patterns
        self.spatial_relations = [
            "in front of", "behind", "next to", "beside", "on top of", "on", 
            "under", "below", "above", "to the left of", "to the right of",
            "inside", "contains", "near", "at", "against", "along", 
            "attached to", "connected to", "part of", "within"
        ]
        
        # Common room types
        self.room_types = [
            "kitchen", "bedroom", "bathroom", "living room", "dining room",
            "hallway", "office", "study", "laundry room", "closet"
        ]
        
        # Object extraction patterns
        self.object_patterns = [
            r'\b(?:a|an|the)\s+([a-zA-Z\s]+?)\b',  # a chair, the kitchen counter
            r'\b([a-zA-Z\s]+?)\s+(?:is|are)\b',     # table is, chairs are
            r'\b(?:see|saw|seeing|notice|noticing|observed|observing)\s+(?:a|an|the)?\s+([a-zA-Z\s]+?)\b'  # see a sofa, observed the table
        ]
        
        # Confidence keywords
        self.confidence_markers = {
            "clearly": 0.9, "definitely": 0.9, "certainly": 0.9,
            "appears to be": 0.7, "looks like": 0.7, "seems to be": 0.7,
            "possibly": 0.5, "maybe": 0.5, "perhaps": 0.5, 
            "might be": 0.4, "could be": 0.4
        }
    
    def parse_thought(self, thought: str, direction_angle: float, step: int) -> Dict:
        """
        Args:
            thought: The thought text from VLM prediction
            direction_angle: The angle (in degrees) this observation was taken from
            step: Current step in the navigation process
            
        Returns:
            Dictionary with extracted entities and relations
        """
        result = {
            "nodes": [],
            "edges": [],
            "direction": direction_angle,
            "step": step
        }
        
        # For the current format, the entire text is used as observation
        observation = thought
        
        # Extract objects from observation text
        objects = self._extract_objects(observation)
        for obj_name, confidence in objects.items():
            result["nodes"].append({
                "name": obj_name,
                "confidence": confidence,
                "direction": direction_angle,
                "step": step
            })
        
        # Extract room information directly from the observation
        # This is important since the current format often mentions room types
        rooms = self._extract_rooms(observation, list(objects.keys()))
        for room_name, confidence in rooms.items():
            result["nodes"].append({
                "name": room_name,
                "attributes": {"type": "room"},
                "confidence": confidence,
                "direction": direction_angle,
                "step": step
            })
            
            # Connect observed objects to their likely rooms
            for obj_name in objects.keys():
                # Check if object is likely in this room based on correlations
                object_room_map = self._get_object_room_correlations()
                is_likely_in_room = (room_name in object_room_map and 
                                    obj_name in object_room_map[room_name])
                
                # Connect objects to room if they're mentioned together or if it's likely
                if is_likely_in_room:
                    result["edges"].append({
                        "source": obj_name,
                        "target": room_name,
                        "relation_type": "in",
                        "confidence": min(objects[obj_name], confidence) * 0.8
                    })
        
        # Extract simple spatial relationships between objects
        if len(objects) >= 2:
            relationships = self._extract_relationships(observation, objects.keys())
            for rel in relationships:
                result["edges"].append(rel)
                
        # Add negative context detection for more accurate object extraction
        # For example, avoid adding "bed" when text says "no sign of a bed"
        negative_patterns = [
            "no sign of", "not visible", "no visible", 
            "no strong indication", "cannot see", "not found"
        ]
        
        nodes_to_remove = []
        for node in result["nodes"]:
            obj_name = node["name"]
            for pattern in negative_patterns:
                # If negative pattern appears near object name, mark for removal
                if pattern in observation.lower() and obj_name in observation.lower():
                    pattern_pos = observation.lower().find(pattern)
                    obj_pos = observation.lower().find(obj_name)
                    if abs(pattern_pos - obj_pos) < 50:  # Within ~50 chars
                        nodes_to_remove.append(node)
                        break
        
        # Remove nodes with negative context
        for node in nodes_to_remove:
            result["nodes"].remove(node)
            
        return result
    
    def update_scene_graph(self, scene_graph: SceneGraph, parsed_data: Dict, 
                           agent_position: np.ndarray = None, step: int = 0) -> None:
        """
        Update the scene graph with confidence accumulation over time.
        
        Args:
            scene_graph: The scene graph to update
            parsed_data: Parsed data from parse_thought()
            agent_position: Current position of the agent
            step: Current step count
        """
        direction_angle = parsed_data["direction"]
        
        # Track added nodes to apply confidence accumulation
        added_nodes = set()
        
        # First, decay confidence of unobserved nodes
        current_observation_objects = {node["name"] for node in parsed_data["nodes"]}
        for node_name, node in scene_graph.nodes.items():
            # Skip path nodes and room type nodes
            if node_name.startswith("path_point_") or node_name in scene_graph.room_nodes:
                continue
                
            # If object wasn't observed in this step, decay its confidence slightly
            if node_name not in current_observation_objects:
                # More aggressive decay for goal objects that weren't seen
                is_goal = node_name == self.current_goal if hasattr(self, 'current_goal') else False
                decay_factor = 0.15 if is_goal else 0.05
                
                # Apply decay
                node.confidence = max(0.1, node.confidence * (1.0 - decay_factor))
            
        # Add or update nodes
        for node_data in parsed_data["nodes"]:
            node_name = node_data["name"]
            added_nodes.add(node_name)
            
            # Position estimation
            position = None
            if agent_position is not None:
                angle_rad = np.deg2rad(direction_angle)
                position = agent_position + np.array([
                    3 * np.sin(angle_rad),
                    0,  # Same height
                    3 * np.cos(angle_rad)
                ])
            
            # Check if this is a room node and apply confidence boost if necessary
            is_room = node_data.get("attributes", {}).get("type") == "room"
            
            # Get existing node if any
            existing_node = scene_graph.nodes.get(node_name)
            base_confidence = node_data["confidence"]
            
            # Apply cumulative confidence if node exists
            if existing_node and existing_node.observations_count > 0:
                # Confidence boost decreases with observations but increases with consistency
                current_conf = float(np.mean(existing_node.confidence) 
                                    if hasattr(existing_node.confidence, '__len__') 
                                    else existing_node.confidence)
                
                # Room confidence accumulates more with repeated observations
                if is_room:
                    # More aggressive accumulation for rooms
                    boost = (1.0 - current_conf) * 0.2 * min(1.0, base_confidence / current_conf)
                    final_confidence = min(0.98, current_conf + boost)
                else:
                    # Normal accumulation for objects
                    boost = (1.0 - current_conf) * 0.15
                    final_confidence = min(0.95, current_conf + boost)
                
                node_data["confidence"] = max(base_confidence, final_confidence)
            
            scene_graph.add_node(
                name=node_name,
                attributes=node_data.get("attributes", {}),
                position=position,
                confidence=node_data["confidence"],
                step=step
            )
        
        # Add or update edges
        for edge_data in parsed_data["edges"]:
            scene_graph.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                relation_type=edge_data["relation_type"],
                confidence=edge_data["confidence"]
            )
    
    def _extract_section(self, thought: str, section_name: str) -> str:
        """Extract a specific section from the structured thought."""
        pattern = f"<{section_name}>(.*?)</{section_name}>"
        match = re.search(pattern, thought, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _is_in_negative_context(self, text: str, object_name: str) -> bool:
        """
        Improved negative context detection that handles full sentence context.
        
        Args:
            text: The full text to analyze
            object_name: The object name to check for negative context
            
        Returns:
            True if the object appears in a negative context, False otherwise
        """
        negative_patterns = [
            "no sign of", "not visible", "no visible", 
            "no strong indication", "cannot see", "not found",
            "not in sight", "no clear", "unlikely to find",
            "doesn't have", "does not have", "is not", "are not",
            "but no", "unlikely", "less likely"
        ]
        
        text_lower = text.lower()
        obj_lower = object_name.lower()
        
        # Check if object is mentioned
        if obj_lower not in text_lower:
            return False
        
        # Split into sentences for more accurate context analysis
        sentences = text_lower.split('.')
        
        for sentence in sentences:
            # Only check sentences containing our object
            if obj_lower in sentence:
                # Check if any negative pattern appears in the same sentence
                if any(pattern in sentence for pattern in negative_patterns):
                    return True
        
        return False
    
    def _extract_objects(self, text: str) -> Dict[str, float]:
        """Extract object names with improved negative context handling."""
        objects = {}
        
        # Common objects we're interested in
        common_objects = [
            "bed", "sofa", "chair", "table", "desk", "tv", "refrigerator",
            "stove", "sink", "toilet", "bathtub", "shower", "nightstand",
            "dresser", "couch", "bookshelf", "cabinet", "wardrobe"
        ]
        
        # Common room types
        room_types = [
            "bedroom", "bathroom", "kitchen", "living room", "dining room",
            "hallway", "office", "closet"
        ]
        
        # Stop words that should never be extracted as objects
        stop_words = [
            "the", "a", "an", "and", "or", "but", "if", "then", "there",
            "here", "where", "when", "who", "what", "how", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should", 
            "may", "might", "must", "can", "could", "to", "in", "on", 
            "with", "no", "not", "other", "another", "possibility", "indication"
        ]
        
        # First pass: direct object matching with negative context check
        text_lower = text.lower()
        
        # Check for common objects
        for obj in common_objects:
            if obj in text_lower and obj not in stop_words:
                # Skip if in negative context
                if not self._is_in_negative_context(text_lower, obj):
                    objects[obj] = 0.6  # Default confidence
        
        # Check for room types with special handling
        for room in room_types:
            if room in text_lower:
                # Rooms need stricter negative context checking
                if not self._is_in_negative_context(text_lower, room):
                    objects[room] = 0.7  # Default confidence for rooms
        
        return objects
    
    def _clean_object_name(self, name: str) -> str:
        """Clean up an object name by removing articles and common modifiers."""
        name = name.lower().strip()
        
        # Remove articles
        name = re.sub(r'\b(a|an|the)\b', '', name).strip()
        
        # Remove common modifiers
        modifiers = ["large", "small", "big", "little", "tall", "short", "wooden", "metal", "plastic"]
        for modifier in modifiers:
            name = re.sub(fr'\b{modifier}\b', '', name).strip()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text (simplified version)."""
        # This is a simplified version - ideally would use spaCy or similar
        # For now, just look for common object terms
        common_objects = [
            "chair", "table", "sofa", "couch", "bed", "cabinet", "desk",
            "lamp", "door", "window", "wall", "floor", "ceiling", "tv",
            "television", "refrigerator", "stove", "oven", "sink", "toilet",
            "shower", "bathtub", "counter", "shelf", "bookshelf", "dresser"
        ]
        
        found = []
        for obj in common_objects:
            if obj in text.lower():
                found.append(obj)
        
        return found
    
    def _extract_relationships(self, text: str, objects: List[str]) -> List[Dict]:
        """More robust spatial relationship extraction."""
        relationships = []
        text_lower = text.lower()
        sentences = re.split(r'[.\n]', text_lower)  # Split text into sentences

        object_list = list(objects)
        if len(object_list) < 2:
            return []

        for sentence in sentences:
            for relation in self.spatial_relations:
                if f' {relation} ' in sentence:
                    # This sentence contains a spatial relation.
                    # Let's see which objects are in it.
                    objects_in_sentence = []
                    for obj in object_list:
                        if obj in sentence:
                            objects_in_sentence.append(obj)
                    
                    # If we found at least two objects, assume a relationship
                    if len(objects_in_sentence) >= 2:
                        # Create pairwise relationships
                        for i in range(len(objects_in_sentence)):
                            for j in range(i + 1, len(objects_in_sentence)):
                                obj1 = objects_in_sentence[i]
                                obj2 = objects_in_sentence[j]
                                
                                # Avoid adding duplicate relationships
                                existing = next((r for r in relationships if (r['source'] == obj1 and r['target'] == obj2) or (r['source'] == obj2 and r['target'] == obj1)), None)
                                if not existing:
                                    # Determine order based on appearance in sentence
                                    if sentence.find(obj1) < sentence.find(obj2):
                                        relationships.append({
                                            "source": obj1,
                                            "target": obj2,
                                            "relation_type": relation,
                                            "confidence": 0.75 
                                        })
                                    else:
                                        relationships.append({
                                            "source": obj2,
                                            "target": obj1,
                                            "relation_type": relation,
                                            "confidence": 0.75
                                        })
        return relationships

    def _extract_rooms(self, text: str, detected_objects: List[str] = None) -> Dict[str, float]:
        """
        Extract room mentions and confidence levels with enhanced object-room correlations.
        
        Args:
            text: The text to extract rooms from
            detected_objects: List of objects detected in the scene
            
        Returns:
            Dictionary mapping room types to confidence levels
        """
        rooms = {}
        detected_objects = detected_objects or []
        
        # First pass: extract rooms mentioned in text
        for room_type in self.room_types:
            if room_type.lower() in text.lower():
                confidence = 0.7  # Default confidence
                
                # Check for confidence markers in text
                if any(marker in text.lower() for marker in ["definitely", "clearly", "certainly"]):
                    confidence = 0.9
                elif any(marker in text.lower() for marker in ["appears to be", "looks like", "seems to be"]):
                    confidence = 0.7
                elif any(marker in text.lower() for marker in ["possibly", "maybe", "perhaps"]):
                    confidence = 0.5
    
                rooms[room_type] = confidence
        
        # Second pass: boost confidence based on detected objects
        object_room_map = self._get_object_room_correlations()
        
        for obj in detected_objects:
            # Check if this object is a strong indicator for any room
            for room, correlated_objects in object_room_map.items():
                if obj in correlated_objects:
                    # Get the confidence boost for this object-room pair
                    indicator_strength = correlated_objects[obj]
                    
                    # If room already detected, take maximum confidence
                    if room in rooms:
                        rooms[room] = max(rooms[room], indicator_strength)
                    # Otherwise, add room with slightly lower confidence
                    elif indicator_strength >= 0.8:  # Only add high-confidence inferences
                        rooms[room] = indicator_strength * 0.85
        
        return rooms
    
    def _get_object_room_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Get mapping from rooms to objects commonly found in them and their confidence levels.
        """
        return {
            "bedroom": {
                "bed": 0.95, 
                "pillow": 0.8, 
                "dresser": 0.85, 
                "nightstand": 0.9,
                "wardrobe": 0.85,
                "mattress": 0.9,
                "blanket": 0.75
            },
            "bathroom": {
                "toilet": 0.95, 
                "shower": 0.95, 
                "bathtub": 0.95, 
                "sink": 0.8,
                "mirror": 0.7,
                "towel": 0.8
            },
            "kitchen": {
                "stove": 0.95, 
                "refrigerator": 0.95, 
                "oven": 0.95,
                "sink": 0.8, 
                "microwave": 0.9,
                "counter": 0.8,
                "dishwasher": 0.9
            },
            "living room": {
                "sofa": 0.9, 
                "couch": 0.9,
                "tv": 0.8, 
                "coffee table": 0.85,
                "armchair": 0.8
            },
            "dining room": {
                "dining table": 0.95, 
                "dining chair": 0.9, 
                "china cabinet": 0.8
            },
            "hallway": {
                "stairs": 0.8, 
                "corridor": 0.9
            }
        }
    

    def parse_vlm_direction_thoughts(thoughts_dict: Dict[str, str], scene_graph: SceneGraph, 
                                    agent_position: np.ndarray, step: int):
        """
        Process all direction thoughts from the PredictVLM and update the scene graph.
        
        Args:
            thoughts_dict: Dictionary mapping direction angles to thought strings
            scene_graph: SceneGraph object to update
            agent_position: Current agent position
            step: Current step number
        
        Returns:
            Updated scene graph
        """
        parser = ThoughtParser()
        
        for direction, thought in thoughts_dict.items():
            try:
                # Convert direction to numeric angle
                angle = int(direction)
                
                # Parse thought and update scene graph
                parsed_data = parser.parse_thought(thought, angle, step)
                parser.update_scene_graph(scene_graph, parsed_data, agent_position, step)
                
            except Exception as e:
                logging.error(f"Error parsing thought for direction {direction}: {e}")
        
        return scene_graph