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
        Parse a structured thought from the VLM into scene graph components.
        
        Args:
            thought: The structured thought from VLM with <Observation>, <Spatial Reasoning>, etc.
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
        
        # Extract each section
        observation = self._extract_section(thought, "Observation")
        spatial_reasoning = self._extract_section(thought, "Spatial Reasoning")
        task_planning = self._extract_section(thought, "Task Planning")
        
        # If no formal "Observation" section, use the "Thought" content or entire text
        if not observation:
            observation = self._extract_section(thought, "Thought")
            if not observation:
                # Use the entire thought as observation
                observation = thought
        
        # Extract objects from observation
        objects = self._extract_objects(observation)
        for obj_name, confidence in objects.items():
            result["nodes"].append({
                "name": obj_name,
                "confidence": confidence,
                "direction": direction_angle,
                "step": step
            })
        
        # Extract spatial relationships
        if spatial_reasoning:
            relationships = self._extract_relationships(spatial_reasoning, objects.keys())
            for rel in relationships:
                result["edges"].append(rel)
                
            # Extract room information with enhanced object-based confidence
            rooms = self._extract_rooms(spatial_reasoning, list(objects.keys()))
            for room_name, confidence in rooms.items():
                result["nodes"].append({
                    "name": room_name,
                    "attributes": {"type": "room"},
                    "confidence": confidence,
                    "direction": direction_angle,
                    "step": step
                })
                
                # Connect observed objects to the room
                for obj_name in objects.keys():
                    # Check if the object is likely in this room based on correlations
                    object_room_map = self._get_object_room_correlations()
                    is_likely_in_room = (room_name in object_room_map and 
                                        obj_name in object_room_map[room_name])
                    
                    if obj_name in spatial_reasoning.lower() or is_likely_in_room:
                        result["edges"].append({
                            "source": obj_name,
                            "target": room_name,
                            "relation_type": "in",
                            "confidence": min(objects[obj_name], confidence) * 0.8
                        })
        
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
    
    def _extract_objects(self, text: str) -> Dict[str, float]:
        """Extract object names and confidence levels from text."""
        objects = {}
        
        # First pass: extract objects using patterns
        for pattern in self.object_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                obj = match.group(1).strip()
                # Filter out very short objects and stopwords
                if len(obj) > 2 and obj not in ["the", "and", "that", "this", "it", "room"]:
                    # Check for confidence markers
                    confidence = 0.6  # Default confidence
                    for marker, value in self.confidence_markers.items():
                        if marker in text.lower():
                            confidence = value
                            break
                    
                    # Clean up the object name
                    obj = self._clean_object_name(obj)
                    if obj and len(obj) > 1:
                        objects[obj] = confidence
        
        # Second pass: direct noun phrase extraction (simulated here)
        # In a real implementation, this could use spaCy or another NLP library
        noun_phrases = self._extract_noun_phrases(text)
        for phrase in noun_phrases:
            cleaned = self._clean_object_name(phrase)
            if cleaned and len(cleaned) > 1:
                objects[cleaned] = objects.get(cleaned, 0.6)
        
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