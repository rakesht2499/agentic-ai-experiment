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
    
    def load_kg(self, filename: str = "educational_kg") -> None:
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