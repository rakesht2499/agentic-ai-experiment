#!/usr/bin/env python3
"""
Improved Educational Knowledge Graph Builder for Sahayak 2.0

Uses smarter extraction focused on actual educational concepts rather than random words.
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
    """Improved knowledge graph builder focused on educational concepts"""
    
    def __init__(self, output_dir: str = "knowledge_graph"):
        self.graph = nx.MultiDiGraph()
        self.concepts = {}
        self.relationships = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.concept_frequency = Counter()
        self.relationship_frequency = Counter()
        self.chapter_concepts = defaultdict(set)
        
        # Setup improved extraction
        self.setup_educational_extraction()
    
    def setup_educational_extraction(self):
        """Setup focused educational concept extraction"""
        
        # Educational vocabulary for Class 10 Science
        self.scientific_vocabulary = {
            # Chemistry terms
            'acid', 'base', 'salt', 'neutralization', 'pH', 'litmus', 'indicator',
            'acidic', 'basic', 'alkaline', 'neutral', 'ion', 'cation', 'anion',
            'hydrogen', 'hydroxide', 'carbonate', 'bicarbonate', 'chloride',
            'sodium', 'calcium', 'magnesium', 'potassium', 'sulfuric', 'hydrochloric',
            'nitric', 'acetic', 'vinegar', 'citric', 'oxalic', 'tartaric',
            'phenolphthalein', 'methyl orange', 'turmeric', 'universal indicator',
            
            # Biology terms  
            'photosynthesis', 'respiration', 'chlorophyll', 'stomata', 'glucose',
            'oxygen', 'carbon dioxide', 'enzyme', 'protein', 'carbohydrate',
            'digestion', 'absorption', 'excretion', 'nutrition', 'autotrophic',
            'heterotrophic', 'cell', 'tissue', 'organ', 'organism',
            'mitochondria', 'chloroplast', 'nucleus', 'cytoplasm', 'membrane',
            'hemoglobin', 'plasma', 'blood', 'heart', 'kidney', 'liver',
            'transpiration', 'translocation', 'xylem', 'phloem',
            
            # Physics terms
            'light', 'reflection', 'refraction', 'lens', 'mirror', 'prism',
            'spectrum', 'dispersion', 'focal length', 'power', 'dioptre',
            'electricity', 'current', 'voltage', 'resistance', 'circuit',
            'magnetic', 'electromagnet', 'induction', 'generator', 'motor',
            
            # General science
            'element', 'compound', 'mixture', 'solution', 'solute', 'solvent',
            'reaction', 'oxidation', 'reduction', 'combustion', 'catalyst',
            'metal', 'non-metal', 'alloy', 'corrosion', 'formula', 'equation'
        }
        
        # Common words to filter out
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'chapter', 'activity',
            'question', 'exercise', 'reprint', 'figure', 'page'
        }
        
        # Chemical formulas pattern
        self.chemical_formulas = {
            'H2O', 'CO2', 'O2', 'H2', 'N2', 'NH3', 'CH4', 'C2H6', 'C2H4', 'C2H2',
            'NaCl', 'CaCl2', 'MgCl2', 'KCl', 'HCl', 'H2SO4', 'HNO3', 'CH3COOH',
            'NaOH', 'Ca(OH)2', 'Mg(OH)2', 'KOH', 'Na2CO3', 'CaCO3', 'NaHCO3',
            'CaSO4', 'MgSO4', 'FeSO4', 'CuSO4', 'ZnSO4', 'Al2O3', 'Fe2O3'
        }
        
        # Process indicators
        self.process_terms = {
            'photosynthesis', 'respiration', 'digestion', 'absorption', 'excretion',
            'neutralization', 'combustion', 'oxidation', 'reduction', 'fermentation',
            'transpiration', 'translocation', 'reflection', 'refraction', 'dispersion'
        }
        
        # Definition patterns
        self.definition_patterns = [
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+([^.]+)\.',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+are\s+([^.]+)\.',
            r'The\s+([a-z]+(?:\s+[a-z]+)*)\s+is\s+([^.]+)\.',
            r'([A-Z][a-z]+)\s*:\s*([^.]+)\.',
        ]
    
    def extract_concepts_from_chunk(self, chunk_data: Dict) -> Tuple[List[ConceptNode], List[ConceptRelationship]]:
        """Extract educational concepts using improved logic"""
        
        content = chunk_data.get('content', '')
        metadata = chunk_data.get('metadata', {})
        
        if len(content.strip()) < 30:
            return [], []
        
        subject = metadata.get('subject', 'Unknown')
        class_level = metadata.get('class', 'Unknown')
        chapter = metadata.get('chapter', 'Unknown')
        
        # Clean the content
        content = self._clean_content(content)
        
        # Extract different types of concepts
        concepts = []
        
        # 1. Extract scientific vocabulary
        scientific_concepts = self._extract_scientific_vocabulary(content, subject, class_level, chapter)
        concepts.extend(scientific_concepts)
        
        # 2. Extract chemical formulas
        formula_concepts = self._extract_chemical_formulas(content, subject, class_level, chapter)
        concepts.extend(formula_concepts)
        
        # 3. Extract definitions
        definition_concepts = self._extract_definitions(content, subject, class_level, chapter)
        concepts.extend(definition_concepts)
        
        # 4. Extract processes
        process_concepts = self._extract_processes(content, subject, class_level, chapter)
        concepts.extend(process_concepts)
        
        # Extract relationships
        relationships = self._extract_meaningful_relationships(content, concepts)
        
        return concepts, relationships
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content)
        # Remove page numbers, figure references, etc.
        content = re.sub(r'Reprint\s+\d{4}-\d{2}', '', content)
        content = re.sub(r'Fig\.\s*\d+\.\d+', '', content)
        content = re.sub(r'\d+CH\d+', '', content)
        return content.strip()
    
    def _extract_scientific_vocabulary(self, content: str, subject: str, class_level: str, chapter: str) -> List[ConceptNode]:
        """Extract known scientific vocabulary from content"""
        concepts = []
        content_lower = content.lower()
        
        for term in self.scientific_vocabulary:
            # Look for the term in the content
            if re.search(rf'\b{re.escape(term)}\b', content_lower):
                concepts.append(ConceptNode(
                    concept=term,
                    concept_type='term',
                    subject=subject,
                    class_level=class_level,
                    chapter=chapter
                ))
        
        return concepts
    
    def _extract_chemical_formulas(self, content: str, subject: str, class_level: str, chapter: str) -> List[ConceptNode]:
        """Extract chemical formulas"""
        concepts = []
        
        for formula in self.chemical_formulas:
            if formula in content:
                concepts.append(ConceptNode(
                    concept=formula,
                    concept_type='formula',
                    subject=subject,
                    class_level=class_level,
                    chapter=chapter
                ))
        
        return concepts
    
    def _extract_definitions(self, content: str, subject: str, class_level: str, chapter: str) -> List[ConceptNode]:
        """Extract terms with definitions"""
        concepts = []
        
        for pattern in self.definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip().lower()
                definition = match.group(2).strip()
                
                # Filter out non-educational terms
                if (len(term) > 2 and 
                    term not in self.stop_words and
                    len(definition) > 10 and
                    len(definition) < 200):
                    
                    concepts.append(ConceptNode(
                        concept=term,
                        concept_type='term',
                        subject=subject,
                        class_level=class_level,
                        chapter=chapter,
                        definition=definition
                    ))
        
        return concepts
    
    def _extract_processes(self, content: str, subject: str, class_level: str, chapter: str) -> List[ConceptNode]:
        """Extract scientific processes"""
        concepts = []
        content_lower = content.lower()
        
        for process in self.process_terms:
            if re.search(rf'\b{re.escape(process)}\b', content_lower):
                concepts.append(ConceptNode(
                    concept=process,
                    concept_type='process',
                    subject=subject,
                    class_level=class_level,
                    chapter=chapter
                ))
        
        return concepts
    
    def _extract_meaningful_relationships(self, content: str, concepts: List[ConceptNode]) -> List[ConceptRelationship]:
        """Extract meaningful relationships between concepts"""
        relationships = []
        concept_names = [c.concept for c in concepts]
        
        # Simple relationship patterns
        relationship_patterns = [
            (r'(\w+)\s+is\s+used\s+in\s+(\w+)', 'used_in'),
            (r'(\w+)\s+is\s+an?\s+example\s+of\s+(\w+)', 'example_of'),
            (r'(\w+)\s+contains\s+(\w+)', 'contains'),
            (r'(\w+)\s+reacts?\s+with\s+(\w+)', 'reacts_with'),
            (r'(\w+)\s+and\s+(\w+)\s+(?:cancel|neutralize)', 'neutralizes')
        ]
        
        for pattern, relation_type in relationship_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                source = match.group(1).lower()
                target = match.group(2).lower()
                
                if source in concept_names and target in concept_names and source != target:
                    relationships.append(ConceptRelationship(
                        source=source,
                        target=target,
                        relation_type=relation_type
                    ))
        
        return relationships
    
    def build_from_jsonl(self, jsonl_file_path: str) -> None:
        """Build knowledge graph from JSONL textbook data"""
        print(f"📚 Building improved knowledge graph from {jsonl_file_path}")
        
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
        
        # Filter out low-quality concepts
        self._filter_concepts()
        
        print(f"🎯 Completed! {len(self.concepts)} quality concepts, {len(self.relationships)} relationships")
        self._generate_statistics()
    
    def _filter_concepts(self):
        """Filter out low-quality concepts"""
        # Remove concepts that appear only once and aren't in our vocabulary
        concepts_to_remove = []
        
        for concept_id, concept in self.concepts.items():
            if (concept.frequency == 1 and 
                concept.concept not in self.scientific_vocabulary and
                concept.concept not in self.chemical_formulas and
                not concept.definition):
                concepts_to_remove.append(concept_id)
        
        for concept_id in concepts_to_remove:
            del self.concepts[concept_id]
            if concept_id in self.graph:
                self.graph.remove_node(concept_id)
        
        print(f"🧹 Filtered out {len(concepts_to_remove)} low-quality concepts")
    
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
        
        stats_file = self.output_dir / "kg_statistics_improved.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Statistics saved to {stats_file}")
        print(f"📈 Summary: {stats['total_concepts']} concepts, {stats['total_relationships']} relationships")
        print(f"🏷️ Concept types: {stats['concept_types']}")
        
        # Show sample concepts
        print("\n🔍 Sample extracted concepts:")
        for i, (cid, concept) in enumerate(list(self.concepts.items())[:10]):
            definition = concept.definition[:50] + "..." if concept.definition else "No definition"
            print(f"  {i+1}. {concept.concept} ({concept.concept_type}) - {definition}")
    
    def save_graph(self, filename: str = "educational_kg_improved") -> None:
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
        
        print(f"💾 Improved knowledge graph saved:")
        print(f"  • Graph: {graph_file}")
        print(f"  • Concepts: {concepts_file}")
        print(f"  • Relationships: {relationships_file}")


def main():
    """Main function to build improved knowledge graph"""
    
    # Initialize the improved knowledge graph builder
    kg_builder = EducationalKnowledgeGraph()
    
    # Path to your JSONL file
    jsonl_file = "upload_textbook_to_index/class10_science.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"❌ JSONL file not found: {jsonl_file}")
        print("Please ensure the file exists and update the path.")
        return
    
    # Build the knowledge graph
    print("🚀 Starting improved knowledge graph construction...")
    kg_builder.build_from_jsonl(jsonl_file)
    
    # Save the graph
    kg_builder.save_graph("cbse_class10_science_kg_improved")
    
    print("🎉 Improved knowledge graph construction completed!")
    print("📂 Files saved in: knowledge_graph/")


if __name__ == "__main__":
    main() 