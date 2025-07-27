#!/usr/bin/env python3
"""
Knowledge Graph Enhanced RAG for Better Context Retrieval
"""

from typing import Dict, List
from knowledge_graph.kg_query_interface import EducationalKGQuerier

class KGEnhancedRAG:
    """Enhance RAG with knowledge graph context"""
    
    def __init__(self):
        self.kg_querier = EducationalKGQuerier()
    
    def enhance_query_with_kg(self, query: str, subject: str, class_level: str, chapter: str = None) -> Dict:
        """Enhance a user query with knowledge graph context"""
        
        # Extract key concepts from the query
        query_concepts = self._extract_concepts_from_query(query)
        
        enhanced_context = {
            'original_query': query,
            'expanded_concepts': [],
            'prerequisites': [],
            'related_concepts': [],
            'suggested_examples': []
        }
        
        for concept in query_concepts:
            # Get concept context from KG
            context = self.kg_querier.get_concept_context(concept)
            
            enhanced_context['expanded_concepts'].append(concept)
            enhanced_context['prerequisites'].extend(context['prerequisites'])
            enhanced_context['related_concepts'].extend(context['related_concepts'])
            enhanced_context['suggested_examples'].extend(context['examples'])
        
        return enhanced_context
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Simple concept extraction from query (can be enhanced with NLP)"""
        # For now, simple keyword matching
        # You can enhance this with spaCy or other NLP libraries
        
        query_lower = query.lower()
        found_concepts = []
        
        # Check against known concepts in the KG
        for concept_data in self.kg_querier.concepts.values():
            concept = concept_data.get('concept', '')
            if concept and concept in query_lower:
                found_concepts.append(concept)
        
        return list(set(found_concepts))
    
    def generate_enhanced_rag_prompt(self, original_query: str, subject: str, class_level: str, chapter: str = None) -> str:
        """Generate an enhanced prompt for RAG that includes KG context"""
        
        kg_context = self.enhance_query_with_kg(original_query, subject, class_level, chapter)
        
        enhanced_prompt = f"Original query: {original_query}\n\n"
        
        if kg_context['prerequisites']:
            enhanced_prompt += "Related prerequisite concepts to consider:\n"
            for prereq in kg_context['prerequisites'][:3]:  # Limit to top 3
                enhanced_prompt += f"- {prereq['concept']}: {prereq.get('definition', 'Key prerequisite concept')}\n"
            enhanced_prompt += "\n"
        
        if kg_context['related_concepts']:
            enhanced_prompt += "Related concepts that should be mentioned:\n"
            for related in kg_context['related_concepts'][:3]:
                enhanced_prompt += f"- {related['concept']}\n"
            enhanced_prompt += "\n"
        
        if kg_context['suggested_examples']:
            enhanced_prompt += "Relevant examples to include:\n"
            for example in kg_context['suggested_examples'][:2]:
                enhanced_prompt += f"- {example['concept']}\n"
            enhanced_prompt += "\n"
        
        enhanced_prompt += f"Please provide comprehensive information about: {original_query}"
        
        return enhanced_prompt