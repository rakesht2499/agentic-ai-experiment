#!/usr/bin/env python3
"""
Knowledge Graph Query Interface for Quiz Generation and Answer Enhancement
"""

import json
import pickle
from typing import List, Dict, Optional, Set
from pathlib import Path
import networkx as nx

class EducationalKGQuerier:
    """Interface to query the educational knowledge graph"""
    
    def __init__(self, kg_dir: str = "knowledge_graph"):
        self.kg_dir = Path(kg_dir)
        self.graph = None
        self.concepts = {}
        self.load_kg()
    
    def load_kg(self, filename: str = "cbse_class10_science_kg") -> None:
        """Load the knowledge graph"""
        graph_file = self.kg_dir / f"{filename}.pkl"
        concepts_file = self.kg_dir / f"{filename}_concepts.json"
        
        if graph_file.exists():
            with open(graph_file, 'rb') as f:
                self.graph = pickle.load(f)
        
        if concepts_file.exists():
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.concepts = json.load(f)
        
        print(f"✅ Loaded KG: {len(self.concepts)} concepts")
    
    def get_related_concepts(self, concept: str, relation_types: List[str] = None, max_results: int = 10) -> List[Dict]:
        """Get concepts related to the given concept"""
        if not self.graph:
            return []
        
        concept = concept.lower()
        related = []
        
        # Find concept nodes
        concept_nodes = [node for node, data in self.graph.nodes(data=True) 
                        if data.get('concept', '').lower() == concept]
        
        for concept_node in concept_nodes:
            # Get neighbors
            for neighbor in self.graph.neighbors(concept_node):
                edge_data = self.graph.get_edge_data(concept_node, neighbor)
                if edge_data:
                    for edge in edge_data.values():
                        relation_type = edge.get('relation_type', 'related_to')
                        
                        if not relation_types or relation_type in relation_types:
                            neighbor_data = self.graph.nodes[neighbor]
                            related.append({
                                'concept': neighbor_data.get('concept', ''),
                                'type': neighbor_data.get('concept_type', ''),
                                'relation_type': relation_type,
                                'chapter': neighbor_data.get('chapter', ''),
                                'definition': neighbor_data.get('definition', '')
                            })
        
        return related[:max_results]
    
    def get_prerequisites(self, concept: str) -> List[Dict]:
        """Get prerequisite concepts for understanding the given concept"""
        return self.get_related_concepts(concept, relation_types=['prerequisite'])
    
    def get_examples(self, concept: str) -> List[Dict]:
        """Get examples of the given concept"""
        return self.get_related_concepts(concept, relation_types=['example_of'])
    
    def get_applications(self, concept: str) -> List[Dict]:
        """Get applications where the concept is used"""
        return self.get_related_concepts(concept, relation_types=['used_in'])
    
    def get_concept_context(self, concept: str) -> Dict:
        """Get comprehensive context for a concept"""
        return {
            'concept': concept,
            'prerequisites': self.get_prerequisites(concept),
            'related_concepts': self.get_related_concepts(concept, relation_types=['related_to']),
            'examples': self.get_examples(concept),
            'applications': self.get_applications(concept),
            'definition': self._get_concept_definition(concept)
        }
    
    def _get_concept_definition(self, concept: str) -> Optional[str]:
        """Get definition of a concept"""
        concept = concept.lower()
        for concept_data in self.concepts.values():
            if concept_data.get('concept', '').lower() == concept:
                return concept_data.get('definition')
        return None
    
    def get_chapter_concepts(self, chapter: str, concept_type: str = None) -> List[Dict]:
        """Get all concepts from a specific chapter"""
        chapter_concepts = []
        
        for concept_data in self.concepts.values():
            if concept_data.get('chapter', '').lower() == chapter.lower():
                if not concept_type or concept_data.get('type') == concept_type:
                    chapter_concepts.append(concept_data)
        
        return sorted(chapter_concepts, key=lambda x: x.get('frequency', 0), reverse=True)
    
    def suggest_quiz_concepts(self, chapter: str, difficulty: str = 'medium', count: int = 5) -> List[Dict]:
        """Suggest concepts for quiz generation based on chapter and difficulty"""
        chapter_concepts = self.get_chapter_concepts(chapter)
        
        # Simple difficulty heuristic based on concept type and frequency
        difficulty_weights = {
            'easy': {'term': 3, 'example': 2, 'process': 1},
            'medium': {'process': 3, 'term': 2, 'law': 2, 'formula': 1},
            'hard': {'law': 3, 'formula': 3, 'process': 2, 'term': 1}
        }
        
        weights = difficulty_weights.get(difficulty, difficulty_weights['medium'])
        
        # Score concepts
        scored_concepts = []
        for concept in chapter_concepts:
            concept_type = concept.get('type', 'term')
            type_weight = weights.get(concept_type, 1)
            frequency_weight = min(concept.get('frequency', 1), 5)  # Cap frequency weight
            
            score = type_weight * frequency_weight
            scored_concepts.append((score, concept))
        
        # Sort by score and return top concepts
        scored_concepts.sort(key=lambda x: x[0], reverse=True)
        return [concept for score, concept in scored_concepts[:count]]
    
    def find_concept_clusters(self, chapter: str, min_connections: int = 2) -> List[List[str]]:
        """Find clusters of highly connected concepts in a chapter"""
        if not self.graph:
            return []
        
        chapter_concepts = self.get_chapter_concepts(chapter)
        chapter_concept_names = [c['concept'] for c in chapter_concepts]
        
        # Create subgraph for the chapter
        chapter_nodes = [node for node, data in self.graph.nodes(data=True) 
                        if data.get('concept', '').lower() in chapter_concept_names]
        
        if not chapter_nodes:
            return []
        
        subgraph = self.graph.subgraph(chapter_nodes)
        
        # Find connected components
        clusters = []
        for component in nx.weakly_connected_components(subgraph):
            if len(component) >= min_connections:
                concept_names = [self.graph.nodes[node]['concept'] for node in component]
                clusters.append(concept_names)
        
        return clusters
    
    def get_concept_path(self, source_concept: str, target_concept: str) -> List[str]:
        """Find the shortest path between two concepts"""
        if not self.graph:
            return []
        
        source_nodes = [node for node, data in self.graph.nodes(data=True) 
                       if data.get('concept', '').lower() == source_concept.lower()]
        target_nodes = [node for node, data in self.graph.nodes(data=True) 
                       if data.get('concept', '').lower() == target_concept.lower()]
        
        if not source_nodes or not target_nodes:
            return []
        
        try:
            path = nx.shortest_path(self.graph, source_nodes[0], target_nodes[0])
            concept_path = [self.graph.nodes[node]['concept'] for node in path]
            return concept_path
        except nx.NetworkXNoPath:
            return []
    
    def search_concepts(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for concepts that match the query"""
        query_lower = query.lower()
        matching_concepts = []
        
        for concept_data in self.concepts.values():
            concept = concept_data.get('concept', '')
            definition = concept_data.get('definition', '')
            
            # Check if query matches concept name or definition
            if (query_lower in concept.lower() or 
                (definition and query_lower in definition.lower())):
                matching_concepts.append(concept_data)
        
        # Sort by relevance (exact matches first, then partial matches)
        def relevance_score(concept_data):
            concept = concept_data.get('concept', '')
            if query_lower == concept.lower():
                return 3  # Exact match
            elif concept.lower().startswith(query_lower):
                return 2  # Starts with query
            else:
                return 1  # Contains query
        
        matching_concepts.sort(key=relevance_score, reverse=True)
        return matching_concepts[:max_results] 