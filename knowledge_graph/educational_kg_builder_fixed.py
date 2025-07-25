#!/usr/bin/env python3
"""
Educational Knowledge Graph Builder for Sahayak 2.0 - Fixed Version

Uses rule-based extraction instead of direct LLM calls to avoid ADK framework issues.
"""

import json
import re
import os
import networkx as nx
import pickle
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path

@dataclass
class ConceptNode:
    concept: str
    concept_type: str  # term, process, example, law, formula
    subject: str
    class_level: str
    chapter: str
    definition: Optional[str] = None
    examples: List[str] = None
    frequency: int = 1

@dataclass
class ConceptRelationship:
    source: str
    target: str
    relation_type: str  # prerequisite, example_of, part_of, used_in
    strength: float = 1.0
    context: Optional[str] = None

class EducationalKnowledgeGraph:
    """Rule-based knowledge graph builder for educational content"""
    
    def __init__(self, output_dir: str = "knowledge_graph"):
        self.graph = nx.MultiDiGraph()
        self.concepts = {}
        self.relationships = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.concept_frequency = Counter()
        self.relationship_frequency = Counter()
        self.chapter_concepts = defaultdict(set)
        
        # Educational patterns for concept extraction
        self.setup_extraction_patterns()
    
    def setup_extraction_patterns(self):
        """Setup regex patterns for extracting educational concepts"""
        
        # Definition patterns
        self.definition_patterns = [
            r'([A-Za-z\s]+)\s+is\s+([^.]+)\.',
            r'([A-Za-z\s]+)\s+are\s+([^.]+)\.',
            r'([A-Za-z\s]+)\s+means\s+([^.]+)\.',
            r'The\s+([A-Za-z\s]+)\s+is\s+([^.]+)\.',
        ]
        
        # Process indicators
        self.process_indicators = [
            'process', 'procedure', 'method', 'mechanism', 'activity', 'reaction',
            'digestion', 'respiration', 'photosynthesis', 'neutralization'
        ]
        
        # Example indicators
        self.example_indicators = [
            'example', 'for instance', 'such as', 'like', 'including'
        ]
    
    def extract_concepts_from_chunk(self, chunk_data: Dict) -> Tuple[List[ConceptNode], List[ConceptRelationship]]:
        """Extract concepts using rule-based approach"""
        
        content = chunk_data.get('content', '')
        metadata = chunk_data.get('metadata', {})
        
        if len(content.strip()) < 50:
            return [], []
        
        subject = metadata.get('subject', 'Unknown')
        class_level = metadata.get('class', 'Unknown')
        chapter = metadata.get('chapter', 'Unknown')
        
        # Extract concepts
        concepts = self._extract_concepts_from_text(content, subject, class_level, chapter)
        
        # Extract relationships
        relationships = self._extract_relationships_from_text(content, concepts)
        
        return concepts, relationships
    
    def _extract_concepts_from_text(self, text: str, subject: str, class_level: str, chapter: str) -> List[ConceptNode]:
        """Extract concepts using rule-based patterns"""
        
        concepts = []
        
        # 1. Extract definitions
        definitions = self._extract_definitions(text)
        for term, definition in definitions:
            concepts.append(ConceptNode(
                concept=term.strip().lower(),
                concept_type='term',
                subject=subject,
                class_level=class_level,
                chapter=chapter,
                definition=definition.strip()
            ))
        
        # 2. Extract processes
        processes = self._extract_processes(text)
        for process in processes:
            concepts.append(ConceptNode(
                concept=process.strip().lower(),
                concept_type='process',
                subject=subject,
                class_level=class_level,
                chapter=chapter
            ))
        
        # 3. Extract scientific terms
        scientific_terms = self._extract_scientific_terms(text)
        for term in scientific_terms:
            if len(term) > 2 and term.lower() not in [c.concept for c in concepts]:
                concepts.append(ConceptNode(
                    concept=term.strip().lower(),
                    concept_type='term',
                    subject=subject,
                    class_level=class_level,
                    chapter=chapter
                ))
        
        # 4. Extract formulas
        formulas = self._extract_formulas(text)
        for formula in formulas:
            concepts.append(ConceptNode(
                concept=formula.strip(),
                concept_type='formula',
                subject=subject,
                class_level=class_level,
                chapter=chapter
            ))
        
        return concepts
    
    def _extract_definitions(self, text: str) -> List[Tuple[str, str]]:
        """Extract term definitions from text"""
        definitions = []
        
        for pattern in self.definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                # Filter out very short or common words
                if len(term) > 3 and not term.lower() in ['the', 'this', 'that', 'these', 'those']:
                    definitions.append((term, definition))
        
        return definitions
    
    def _extract_processes(self, text: str) -> List[str]:
        """Extract processes and procedures"""
        processes = []
        text_lower = text.lower()
        
        # Look for process indicators
        for indicator in self.process_indicators:
            if indicator in text_lower:
                # Extract surrounding context
                sentences = text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        # Simple extraction of process name
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if indicator in word.lower():
                                # Look for capitalized words around the indicator
                                context_words = words[max(0, i-3):i+4]
                                process_candidates = [w for w in context_words 
                                                    if w[0].isupper() and len(w) > 3]
                                processes.extend(process_candidates)
        
        return list(set(processes))  # Remove duplicates
    
    def _extract_scientific_terms(self, text: str) -> List[str]:
        """Extract scientific terms using patterns"""
        terms = set()
        
        # Common scientific terms
        scientific_vocab = [
            'acid', 'base', 'salt', 'pH', 'litmus', 'indicator', 'neutralization',
            'photosynthesis', 'chlorophyll', 'stomata', 'respiration', 'glucose',
            'oxygen', 'carbon dioxide', 'enzyme', 'protein', 'carbohydrate',
            'vitamin', 'mineral', 'digestion', 'absorption', 'excretion',
            'kidney', 'liver', 'heart', 'blood', 'plasma', 'hemoglobin',
            'mitochondria', 'cell', 'nucleus', 'cytoplasm', 'membrane',
            'element', 'compound', 'mixture', 'solution', 'solute', 'solvent',
            'metal', 'non-metal', 'ion', 'electron', 'proton', 'neutron',
            'atom', 'molecule', 'reaction', 'oxidation', 'reduction'
        ]
        
        text_lower = text.lower()
        for term in scientific_vocab:
            if term in text_lower:
                terms.add(term)
        
        # Extract capitalized scientific terms
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(capitalized_pattern, text)
        for match in matches:
            if len(match) > 3 and not match in ['Activity', 'Question', 'Exercise', 'Chapter']:
                terms.add(match.lower())
        
        return list(terms)
    
    def _extract_formulas(self, text: str) -> List[str]:
        """Extract chemical formulas and equations"""
        formulas = set()
        
        # Common chemical formulas
        common_formulas = ['H2O', 'CO2', 'O2', 'H2', 'NaCl', 'HCl', 'NaOH', 'CaCO3']
        for formula in common_formulas:
            if formula in text:
                formulas.add(formula)
        
        # Chemical formula pattern
        formula_pattern = r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'
        matches = re.findall(formula_pattern, text)
        
        for match in matches:
            # Filter for likely chemical formulas
            if (len(match) >= 2 and len(match) <= 10 and 
                any(c.isupper() for c in match) and
                any(c.isdigit() for c in match)):
                formulas.add(match)
        
        return list(formulas)
    
    def _extract_relationships_from_text(self, text: str, concepts: List[ConceptNode]) -> List[ConceptRelationship]:
        """Extract relationships between concepts"""
        relationships = []
        
        # Simple relationship extraction based on proximity and patterns
        concept_names = [c.concept for c in concepts]
        
        # Look for "is used in" patterns
        for i, concept1 in enumerate(concept_names):
            for j, concept2 in enumerate(concept_names):
                if i != j:
                    # Check if concepts appear in the same sentence
                    sentences = text.split('.')
                    for sentence in sentences:
                        if concept1 in sentence.lower() and concept2 in sentence.lower():
                            # Determine relationship type based on context
                            if 'example' in sentence.lower():
                                relationships.append(ConceptRelationship(
                                    source=concept1,
                                    target=concept2,
                                    relation_type='example_of'
                                ))
                            elif 'used in' in sentence.lower():
                                relationships.append(ConceptRelationship(
                                    source=concept1,
                                    target=concept2,
                                    relation_type='used_in'
                                ))
                            elif 'part of' in sentence.lower():
                                relationships.append(ConceptRelationship(
                                    source=concept1,
                                    target=concept2,
                                    relation_type='part_of'
                                ))
                            else:
                                relationships.append(ConceptRelationship(
                                    source=concept1,
                                    target=concept2,
                                    relation_type='related_to'
                                ))
        
        return relationships
    
    def build_from_jsonl(self, jsonl_file_path: str) -> None:
        """Build knowledge graph from JSONL textbook data"""
        print(f"📚 Building knowledge graph from {jsonl_file_path}")
        
        processed_chunks = 0
        total_concepts = 0
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    chunk_data = json.loads(line.strip())
                    
                    # Extract concepts and relationships
                    concept_nodes, relationships = self.extract_concepts_from_chunk(chunk_data)
                    
                    # Add to graph
                    for concept_node in concept_nodes:
                        self._add_concept_to_graph(concept_node)
                        total_concepts += 1
                    
                    for relationship in relationships:
                        self._add_relationship_to_graph(relationship)
                    
                    processed_chunks += 1
                    
                    if processed_chunks % 50 == 0:
                        print(f"✅ Processed {processed_chunks} chunks, {len(self.concepts)} unique concepts")
                
                except Exception as e:
                    print(f"⚠️ Error processing line {line_num}: {e}")
        
        print(f"🎯 Completed! {len(self.concepts)} unique concepts, {len(self.relationships)} relationships")
        self._generate_statistics()
    
    def _add_concept_to_graph(self, concept_node: ConceptNode) -> None:
        """Add concept to graph"""
        concept_id = f"{concept_node.subject}_{concept_node.class_level}_{concept_node.concept}"
        
        if concept_id in self.concepts:
            self.concepts[concept_id].frequency += 1
        else:
            self.concepts[concept_id] = concept_node
            self.graph.add_node(concept_id, **concept_node.__dict__)
        
        self.concept_frequency[concept_node.concept] += 1
        self.chapter_concepts[concept_node.chapter].add(concept_node.concept)
    
    def _add_relationship_to_graph(self, relationship: ConceptRelationship) -> None:
        """Add relationship to graph"""
        source_concepts = [cid for cid, concept in self.concepts.items() 
                          if concept.concept == relationship.source]
        target_concepts = [cid for cid, concept in self.concepts.items() 
                          if concept.concept == relationship.target]
        
        for source_id in source_concepts:
            for target_id in target_concepts:
                self.graph.add_edge(
                    source_id, target_id,
                    relation_type=relationship.relation_type,
                    strength=relationship.strength,
                    context=relationship.context
                )
                
        if source_concepts and target_concepts:
            self.relationships.append(relationship)
            self.relationship_frequency[relationship.relation_type] += 1
    
    def _generate_statistics(self) -> None:
        """Generate statistics"""
        stats = {
            "total_concepts": len(self.concepts),
            "total_relationships": len(self.relationships),
            "concept_types": dict(Counter([c.concept_type for c in self.concepts.values()])),
            "relationship_types": dict(self.relationship_frequency),
            "top_concepts": dict(self.concept_frequency.most_common(20)),
            "concepts_per_chapter": {k: len(v) for k, v in self.chapter_concepts.items()}
        }
        
        stats_file = self.output_dir / "kg_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Statistics saved to {stats_file}")
        print(f"📈 Summary: {stats['total_concepts']} concepts, {stats['total_relationships']} relationships")
        print(f"🏷️ Concept types: {stats['concept_types']}")
    
    def save_graph(self, filename: str = "educational_kg") -> None:
        """Save knowledge graph"""
        # Save NetworkX graph
        graph_file = self.output_dir / f"{filename}.pkl"
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Save concepts as JSON
        concepts_file = self.output_dir / f"{filename}_concepts.json"
        concepts_data = {
            cid: {
                'concept': c.concept,
                'type': c.concept_type,
                'subject': c.subject,
                'class_level': c.class_level,
                'chapter': c.chapter,
                'frequency': c.frequency,
                'definition': c.definition
            } for cid, c in self.concepts.items()
        }
        
        with open(concepts_file, 'w', encoding='utf-8') as f:
            json.dump(concepts_data, f, indent=2, ensure_ascii=False)
        
        # Save relationships
        relationships_file = self.output_dir / f"{filename}_relationships.json"
        relationships_data = [
            {
                'source': r.source,
                'target': r.target,
                'relation_type': r.relation_type,
                'strength': r.strength,
                'context': r.context
            } for r in self.relationships
        ]
        
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(relationships_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Knowledge graph saved:")
        print(f"  • Graph: {graph_file}")
        print(f"  • Concepts: {concepts_file}")
        print(f"  • Relationships: {relationships_file}")


def main():
    """Main function to build knowledge graph from JSONL data"""
    
    # Initialize the knowledge graph builder
    kg_builder = EducationalKnowledgeGraph()
    
    # Path to your JSONL file
    jsonl_file = "upload_textbook_to_index/class10_science.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        print("Please ensure the file exists and update the path.")
        return
    
    # Build the knowledge graph
    print("🚀 Starting knowledge graph construction...")
    kg_builder.build_from_jsonl(jsonl_file)
    
    # Save the graph
    kg_builder.save_graph("cbse_class10_science_kg")
    
    print("🎉 Knowledge graph construction completed!")
    print("📂 Files saved in: knowledge_graph/")


if __name__ == "__main__":
    main() 