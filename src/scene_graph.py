import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
import matplotlib.pyplot as plt
import json
import logging
from collections import defaultdict

class SceneGraphNode:
    """Represents a node in the scene graph, typically an object or location."""
    
    def __init__(self, name: str, attributes: dict = None, position: np.ndarray = None):
        self.name = name
        self.attributes = attributes or {}
        self.position = position
        self.confidence = 0.0
        self.last_seen = 0  # Step when this node was last observed
        self.observations_count = 0
        self.detection_timestamps = []
    
    def update(self, attributes=None, position=None, confidence=None, step=None):
        """Update node properties with new observations."""
        if attributes:
            for k, v in attributes.items():
                self.attributes[k] = v
        
        if position is not None:
            if self.position is None:
                self.position = position
            else:
                # Moving average of position - 确保使用标量运算
                if hasattr(self.position, '__len__') and hasattr(position, '__len__'):
                    self.position = (self.position * self.observations_count + position) / (self.observations_count + 1)
                else:
                    self.position = position
        
        if confidence is not None:
            # 修复：确保 confidence 比较使用标量值
            if hasattr(confidence, '__len__'):  # 如果是数组
                confidence = float(np.mean(confidence))
            if hasattr(self.confidence, '__len__'):  # 如果现有confidence是数组
                current_confidence = float(np.mean(self.confidence))
            else:
                current_confidence = float(self.confidence)
            
            self.confidence = max(current_confidence, float(confidence))
        
        if step is not None:
            self.last_seen = step
            self.observations_count += 1
            self.detection_timestamps.append(step)
    
    def __str__(self):
        return f"Node({self.name})"
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "position": self.position.tolist() if self.position is not None else None,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "observations_count": self.observations_count
        }


class SceneGraphEdge:
    """Represents a relationship between two nodes."""
    
    def __init__(self, source: str, target: str, relation_type: str, confidence: float = 0.0):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.confidence = confidence
        self.observations_count = 0
    
    def update(self, relation_type=None, confidence=None):
        """Update edge properties with new observations."""
        if relation_type:
            self.relation_type = relation_type
            
        if confidence is not None:
            # 修复：确保 confidence 比较使用标量值
            if hasattr(confidence, '__len__'):  # 如果是数组
                confidence = float(np.mean(confidence))
            if hasattr(self.confidence, '__len__'):  # 如果现有confidence是数组
                current_confidence = float(np.mean(self.confidence))
            else:
                current_confidence = float(self.confidence)
                
            self.confidence = max(current_confidence, float(confidence))
        
        self.observations_count += 1
    
    def __str__(self):
        return f"Edge({self.source} --{self.relation_type}--> {self.target})"
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "observations_count": self.observations_count
        }


class SceneGraph:
    """
    A structured representation of the agent's knowledge about the environment.
    Implements a graph-based memory model inspired by UniGoal.
    """
    
    def __init__(self):
        self.nodes: Dict[str, SceneGraphNode] = {}
        self.edges: List[SceneGraphEdge] = []
        self.room_nodes: Set[str] = set()  # Specific nodes representing rooms
        self.object_hierarchy = defaultdict(set)  # For tracking object containment
        
        # Initialize with common room types
        common_rooms = ["kitchen", "bedroom", "bathroom", "living room", "dining room", "hallway"]
        for room in common_rooms:
            self.add_room(room)
    
    def add_node(self, name: str, attributes=None, position=None, confidence=0.0, step=None) -> SceneGraphNode:
        """Add or update a node in the graph."""
        name = name.lower()  # Normalize node names to lowercase
        
        if name in self.nodes:
            self.nodes[name].update(attributes, position, confidence, step)
        else:
            node = SceneGraphNode(name, attributes, position)
            node.update(confidence=confidence, step=step)
            self.nodes[name] = node
            
        return self.nodes[name]
    
    def add_room(self, room_name: str) -> SceneGraphNode:
        """Add a room node to the graph."""
        room_name = room_name.lower()
        node = self.add_node(room_name, {"type": "room"})
        self.room_nodes.add(room_name)
        return node
    
    def add_edge(self, source: str, target: str, relation_type: str, confidence: float = 0.5) -> Optional[SceneGraphEdge]:
        """Add or update an edge between two nodes."""
        source = source.lower()
        target = target.lower()
        
        # Make sure both nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        
        # Check if edge already exists
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                edge.update(relation_type, confidence)
                return edge
        
        # Create new edge
        edge = SceneGraphEdge(source, target, relation_type, confidence)
        self.edges.append(edge)
        
        # If this is a containment relation, update the hierarchy
        if relation_type in ["in", "inside", "contains"]:
            self.object_hierarchy[target].add(source)
        
        return edge
    
    def get_subgraph_for_goal(self, goal: str, max_distance: int = 2) -> Tuple[Dict[str, SceneGraphNode], List[SceneGraphEdge]]:
        """
        Extract a subgraph relevant to the specified goal.
        Returns nodes and edges within max_distance of the goal node.
        """
        goal = goal.lower()
        relevant_nodes = {}
        relevant_edges = []
        visited = set()
        
        # Check if we have the goal node
        if goal not in self.nodes:
            # Find rooms that might contain the goal
            related_rooms = self._find_likely_rooms_for_object(goal)
            if related_rooms:
                for room in related_rooms:
                    self._extract_subgraph(room, relevant_nodes, relevant_edges, visited, max_distance)
            return relevant_nodes, relevant_edges
        
        # Extract subgraph starting from goal node
        self._extract_subgraph(goal, relevant_nodes, relevant_edges, visited, max_distance)

        return relevant_nodes, relevant_edges
    
    def _extract_subgraph(self, start_node: str, nodes_dict: dict, edges_list: list, visited: set, max_distance: int, current_distance: int = 0):
        """Helper for extracting a subgraph within max_distance of start_node."""
        if current_distance > max_distance or start_node in visited:
            return
        
        visited.add(start_node)
        if start_node in self.nodes:
            nodes_dict[start_node] = self.nodes[start_node]
        
        # Find all connected edges
        connected_edges = []
        for edge in self.edges:
            if edge.source == start_node or edge.target == start_node:
                connected_edges.append(edge)
                if edge not in edges_list:
                    edges_list.append(edge)
        
        # Recursively traverse connected nodes
        for edge in connected_edges:
            next_node = edge.target if edge.source == start_node else edge.source
            self._extract_subgraph(next_node, nodes_dict, edges_list, visited, max_distance, current_distance + 1)
    
    def _find_likely_rooms_for_object(self, object_name: str) -> List[str]:
        """
        Find rooms that are likely to contain the given object using enhanced probabilities.
        """
        # Common object-room associations with probabilities
        object_room_map = {
            "bed": {"bedroom": 0.95, "living_room": 0.05},
            "toilet": {"bathroom": 0.98},
            "sink": {"bathroom": 0.8, "kitchen": 0.8},
            "shower": {"bathroom": 0.98},
            "sofa": {"living_room": 0.95, "bedroom": 0.05},
            "tv": {"living_room": 0.85, "bedroom": 0.4},
            "table": {"dining_room": 0.9, "living_room": 0.5, "kitchen": 0.7},
            "stove": {"kitchen": 0.98},
            "refrigerator": {"kitchen": 0.98},
            "desk": {"bedroom": 0.7, "office": 0.9}
        }
        
        results = []
        
        # Check if we have a direct mapping
        for obj_pattern, room_probs in object_room_map.items():
            if obj_pattern in object_name.lower():
                # Sort rooms by probability
                sorted_rooms = sorted(room_probs.items(), key=lambda x: x[1], reverse=True)
                for room, prob in sorted_rooms:
                    if room in self.room_nodes and prob > 0.3:
                        results.append(room)
        
        # Return all rooms if no specific mapping found
        if not results:
            return list(self.room_nodes)
        
        return results
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to a NetworkX graph for visualization and analysis."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for name, node in self.nodes.items():
            attributes = node.attributes.copy()
            attributes["confidence"] = node.confidence
            attributes["observations"] = node.observations_count
            if node.position is not None:
                attributes["position"] = node.position.tolist()
            G.add_node(name, **attributes)
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, 
                       relation=edge.relation_type, 
                       confidence=edge.confidence,
                       observations=edge.observations_count)
        
        return G
    
    def visualize(self, save_path=None, highlight_goal=None):
        """Enhanced visualization with goal highlighting and confidence information"""
        G = self.to_networkx()
        
        # Set node colors and sizes based on type and confidence
        colors = []
        sizes = []
        edge_weights = []
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
        
        # Prepare node styling
        for node in G.nodes():
            if node in self.room_nodes:
                colors.append('lightblue')
                sizes.append(700)  # Larger for rooms
            elif highlight_goal and node == highlight_goal.lower():
                colors.append('red')  # Highlight goal in red
                sizes.append(800)  # Make goal larger
            else:
                colors.append('lightgreen')
                sizes.append(500)
                
            # Add confidence to node labels if available
            if node in self.nodes and self.nodes[node].confidence > 0:
                node_label = f"{node}\n{self.nodes[node].confidence:.2f}"
                G.nodes[node]["label"] = node_label
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.8)
        
        # Draw custom node labels with confidence
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
        
        # Draw edges with confidence-based styling
        for edge in self.edges:
            width = 1.0 + edge.confidence * 2  # Thicker edges for higher confidence
            nx.draw_networkx_edges(G, pos, edgelist=[(edge.source, edge.target)], 
                                  width=width, alpha=0.6, 
                                  edge_color='gray' if edge.confidence < 0.7 else 'blue')
        
        # Add edge labels
        edge_labels = {(e.source, e.target): f"{e.relation_type}\n{e.confidence:.2f}" 
                      for e in self.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add title with stats
        plt.title(f"Scene Graph - {len(self.nodes)} objects, {len(self.edges)} relationships")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            plt.show()
            return None
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "rooms": list(self.room_nodes)
        }
    
    def save(self, filepath):
        """Save the graph to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load a graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        
        # Load nodes
        for name, node_data in data["nodes"].items():
            position = np.array(node_data["position"]) if node_data["position"] else None
            node = SceneGraphNode(name, node_data["attributes"], position)
            node.confidence = node_data["confidence"]
            node.last_seen = node_data["last_seen"]
            node.observations_count = node_data["observations_count"]
            graph.nodes[name] = node
        
        # Load room nodes
        graph.room_nodes = set(data["rooms"])
        
        # Load edges
        for edge_data in data["edges"]:
            edge = SceneGraphEdge(
                edge_data["source"], 
                edge_data["target"], 
                edge_data["relation_type"], 
                edge_data["confidence"]
            )
            edge.observations_count = edge_data["observations_count"]
            graph.edges.append(edge)
            
            # Rebuild hierarchy for containment relations
            if edge.relation_type in ["in", "inside", "contains"]:
                graph.object_hierarchy[edge.target].add(edge.source)
        
        return graph