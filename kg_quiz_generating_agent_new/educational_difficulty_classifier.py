#!/usr/bin/env python3
"""
Educational Difficulty Classifier for Sahayak 2.0

Implements semantic-based difficulty classification using:
1. Educational concept ontology (semantic families)
2. Prerequisite mapping (learning dependencies) 
3. Bloom's taxonomy cognitive levels
4. Curriculum standards alignment
5. Domain-specific educational rules
"""

from typing import Dict, List, Set, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import re

class BloomLevel(Enum):
    """Bloom's Taxonomy cognitive levels"""
    REMEMBER = 1    # Recall facts and basic concepts
    UNDERSTAND = 2  # Explain ideas or concepts  
    APPLY = 3       # Use information in new situations
    ANALYZE = 4     # Draw connections among ideas
    EVALUATE = 5    # Justify a stand or decision
    CREATE = 6      # Produce new or original work

class DifficultyLevel(Enum):
    """Educational difficulty levels"""
    BEGINNER = 1     # Foundation concepts, definitions
    INTERMEDIATE = 2 # Processes, relationships  
    ADVANCED = 3     # Applications, calculations
    EXPERT = 4       # Analysis, synthesis, evaluation

@dataclass
class ConceptClassification:
    """Classification result for an educational concept"""
    concept: str
    difficulty: DifficultyLevel
    bloom_level: BloomLevel
    semantic_family: str
    prerequisites: List[str]
    cognitive_load: int  # 1-10 scale
    educational_importance: float  # 0-1 scale
    reason: str  # Why this difficulty was assigned

class EducationalDifficultyClassifier:
    """Advanced classifier using semantic relationships and educational hierarchy"""
    
    def __init__(self):
        self.setup_semantic_families()
        self.setup_prerequisite_chains()
        self.setup_bloom_taxonomy_mapping()
        self.setup_domain_rules()
    
    def setup_semantic_families(self):
        """Define semantic concept families for chemistry/biology/physics"""
        
        self.semantic_families = {
            # Chemistry - Acids & Bases Family
            "acid_base_family": {
                "concepts": [
                    "acid", "base", "acidic", "basic", "alkaline", "neutral",
                    "ph", "litmus", "indicator", "phenolphthalein", "methyl orange",
                    "turmeric", "neutralization", "salt", "hydrochloric", "sulfuric"
                ],
                "core_difficulty": DifficultyLevel.BEGINNER,
                "description": "Acid-base chemistry fundamentals"
            },
            
            # Chemistry - Chemical Formulas Family  
            "formula_family": {
                "concepts": [
                    "hcl", "h2so4", "naoh", "koh", "nacl", "caco3", 
                    "h2o", "co2", "o2", "h2", "ch3cooh", "ca(oh)2"
                ],
                "core_difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Chemical formulas and equations"
            },
            
            # Chemistry - Ions & Reactions Family
            "ion_reaction_family": {
                "concepts": [
                    "ion", "cation", "anion", "ionic", "electrolyte",
                    "reaction", "oxidation", "reduction", "combustion",
                    "catalyst", "equilibrium"
                ],
                "core_difficulty": DifficultyLevel.ADVANCED,
                "description": "Ionic chemistry and reactions"
            },
            
            # Biology - Life Processes Family
            "life_processes_family": {
                "concepts": [
                    "photosynthesis", "respiration", "digestion", "excretion",
                    "nutrition", "metabolism", "enzyme", "glucose", "oxygen"
                ],
                "core_difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Fundamental life processes"
            },
            
            # Biology - Cell Structure Family
            "cell_structure_family": {
                "concepts": [
                    "cell", "nucleus", "cytoplasm", "mitochondria", "chloroplast",
                    "membrane", "cell wall", "vacuole", "ribosome"
                ],
                "core_difficulty": DifficultyLevel.BEGINNER,
                "description": "Basic cell biology"
            },
            
            # Biology - Human Body Systems Family
            "body_systems_family": {
                "concepts": [
                    "heart", "blood", "kidney", "liver", "lung", "brain",
                    "hemoglobin", "plasma", "circulation", "filtration"
                ],
                "core_difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Human body systems"
            },
            
            # Physics - Light & Optics Family
            "optics_family": {
                "concepts": [
                    "light", "reflection", "refraction", "lens", "mirror",
                    "focal length", "prism", "spectrum", "dispersion"
                ],
                "core_difficulty": DifficultyLevel.INTERMEDIATE,
                "description": "Light and optical phenomena"
            },
            
            # Physics - Electricity Family
            "electricity_family": {
                "concepts": [
                    "current", "voltage", "resistance", "circuit", "power",
                    "electric", "magnetic", "conductor", "insulator"
                ],
                "core_difficulty": DifficultyLevel.ADVANCED,
                "description": "Electrical phenomena"
            }
        }
    
    def setup_prerequisite_chains(self):
        """Define prerequisite relationships between concepts"""
        
        self.prerequisite_chains = {
            # Acid-Base Chemistry Chain
            "neutralization": ["acid", "base"],
            "ph": ["acid", "base", "neutral"],
            "salt": ["acid", "base", "neutralization"],
            "indicator": ["acid", "base"],
            
            # Chemical Reactions Chain  
            "oxidation": ["reaction", "oxygen"],
            "reduction": ["reaction", "oxidation"],
            "combustion": ["reaction", "oxygen"],
            "catalyst": ["reaction"],
            
            # Biology Process Chain
            "photosynthesis": ["chloroplast", "glucose", "oxygen"],
            "respiration": ["oxygen", "glucose", "mitochondria"],
            "digestion": ["enzyme", "nutrition"],
            
            # Cell Biology Chain
            "mitochondria": ["cell", "energy"],
            "chloroplast": ["cell", "photosynthesis"],
            "nucleus": ["cell"],
            
            # Human Body Chain
            "circulation": ["heart", "blood"],
            "hemoglobin": ["blood", "oxygen"],
            "filtration": ["kidney", "blood"],
            
            # Physics Chain
            "refraction": ["light", "reflection"],
            "focal length": ["lens", "light"],
            "spectrum": ["light", "dispersion"],
            "circuit": ["current", "voltage"],
            "power": ["current", "voltage", "resistance"]
        }
    
    def setup_bloom_taxonomy_mapping(self):
        """Map concept types to Bloom's taxonomy levels"""
        
        self.bloom_mapping = {
            # Remember Level - Basic recall
            BloomLevel.REMEMBER: {
                "indicators": ["definition", "what is", "identify", "list", "name"],
                "concept_types": ["basic_term", "simple_definition"],
                "examples": ["What is an acid?", "Name three acids"]
            },
            
            # Understand Level - Comprehension
            BloomLevel.UNDERSTAND: {
                "indicators": ["explain", "describe", "why", "how", "difference"],
                "concept_types": ["process", "relationship", "property"],
                "examples": ["Why do acids turn litmus red?", "How does photosynthesis work?"]
            },
            
            # Apply Level - Use in new situations
            BloomLevel.APPLY: {
                "indicators": ["calculate", "solve", "use", "apply", "demonstrate"],
                "concept_types": ["formula", "equation", "procedure"],
                "examples": ["Balance the equation", "Calculate pH"]
            },
            
            # Analyze Level - Break down and examine
            BloomLevel.ANALYZE: {
                "indicators": ["compare", "contrast", "analyze", "examine", "investigate"],
                "concept_types": ["comparison", "relationship", "mechanism"],
                "examples": ["Compare acids and bases", "Analyze the reaction"]
            }
        }
    
    def setup_domain_rules(self):
        """Domain-specific rules for chemistry, biology, physics"""
        
        self.domain_rules = {
            "chemistry": {
                "formula_difficulty": DifficultyLevel.INTERMEDIATE,
                "reaction_difficulty": DifficultyLevel.ADVANCED,
                "basic_properties": DifficultyLevel.BEGINNER,
                "calculation_difficulty": DifficultyLevel.EXPERT
            },
            
            "biology": {
                "structure_difficulty": DifficultyLevel.BEGINNER,
                "process_difficulty": DifficultyLevel.INTERMEDIATE,
                "system_difficulty": DifficultyLevel.ADVANCED,
                "molecular_difficulty": DifficultyLevel.EXPERT
            },
            
            "physics": {
                "phenomenon_difficulty": DifficultyLevel.INTERMEDIATE,
                "calculation_difficulty": DifficultyLevel.ADVANCED,
                "theory_difficulty": DifficultyLevel.EXPERT,
                "observation_difficulty": DifficultyLevel.BEGINNER
            }
        }
    
    def classify_concept(self, concept: str, concept_type: str, frequency: int, 
                        definition: Optional[str] = None, chapter: str = "") -> ConceptClassification:
        """Classify a concept using semantic and educational hierarchy"""
        
        concept_lower = concept.lower()
        
        # 1. Determine semantic family
        semantic_family = self._find_semantic_family(concept_lower)
        
        # 2. Calculate base difficulty from semantic family
        base_difficulty = self._get_base_difficulty(concept_lower, semantic_family, concept_type)
        
        # 3. Adjust for prerequisites
        prerequisite_adjustment = self._calculate_prerequisite_difficulty(concept_lower)
        
        # 4. Determine Bloom's taxonomy level
        bloom_level = self._determine_bloom_level(concept_lower, concept_type, definition)
        
        # 5. Calculate cognitive load
        cognitive_load = self._calculate_cognitive_load(
            concept_lower, concept_type, semantic_family, prerequisite_adjustment
        )
        
        # 6. Apply domain-specific rules
        domain_adjusted_difficulty = self._apply_domain_rules(
            base_difficulty, concept_type, semantic_family
        )
        
        # 7. Calculate educational importance
        educational_importance = self._calculate_educational_importance(
            concept_lower, frequency, semantic_family, chapter
        )
        
        # 8. Final difficulty determination
        final_difficulty = self._determine_final_difficulty(
            domain_adjusted_difficulty, prerequisite_adjustment, cognitive_load
        )
        
        # 9. Generate explanation
        reason = self._generate_classification_reason(
            concept_lower, semantic_family, final_difficulty, bloom_level
        )
        
        return ConceptClassification(
            concept=concept,
            difficulty=final_difficulty,
            bloom_level=bloom_level,
            semantic_family=semantic_family,
            prerequisites=self.prerequisite_chains.get(concept_lower, []),
            cognitive_load=cognitive_load,
            educational_importance=educational_importance,
            reason=reason
        )
    
    def _find_semantic_family(self, concept: str) -> str:
        """Find which semantic family a concept belongs to"""
        for family_name, family_data in self.semantic_families.items():
            if concept in family_data["concepts"]:
                return family_name
        return "general_science"
    
    def _get_base_difficulty(self, concept: str, semantic_family: str, concept_type: str) -> DifficultyLevel:
        """Get base difficulty from semantic family and concept type"""
        
        # Chemical formulas are inherently more complex
        if concept_type == "formula" or re.match(r'^[A-Z][a-z]?(\d+)?(\([A-Z][a-z]?\d*\)\d*)?$', concept):
            return DifficultyLevel.INTERMEDIATE
        
        # Get from semantic family
        if semantic_family in self.semantic_families:
            return self.semantic_families[semantic_family]["core_difficulty"]
        
        # Default based on concept type
        if concept_type == "term":
            return DifficultyLevel.BEGINNER
        elif concept_type == "process":
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.BEGINNER
    
    def _calculate_prerequisite_difficulty(self, concept: str) -> int:
        """Calculate difficulty adjustment based on prerequisites"""
        if concept in self.prerequisite_chains:
            # More prerequisites = higher difficulty
            return len(self.prerequisite_chains[concept])
        return 0
    
    def _determine_bloom_level(self, concept: str, concept_type: str, definition: Optional[str]) -> BloomLevel:
        """Determine Bloom's taxonomy level"""
        
        # Formulas typically require application
        if concept_type == "formula":
            return BloomLevel.APPLY
        
        # Processes require understanding
        if concept_type == "process":
            return BloomLevel.UNDERSTAND
        
        # Terms with definitions are usually remember level
        if definition and len(definition) > 10:
            return BloomLevel.REMEMBER
        
        # Complex concepts require understanding
        if concept in self.prerequisite_chains:
            return BloomLevel.UNDERSTAND
        
        return BloomLevel.REMEMBER
    
    def _calculate_cognitive_load(self, concept: str, concept_type: str, 
                                 semantic_family: str, prerequisite_count: int) -> int:
        """Calculate cognitive load on 1-10 scale"""
        
        load = 1  # Base load
        
        # Add for concept type
        if concept_type == "formula":
            load += 3
        elif concept_type == "process":
            load += 2
        
        # Add for semantic family complexity
        complex_families = ["ion_reaction_family", "electricity_family"]
        if semantic_family in complex_families:
            load += 2
        
        # Add for prerequisites
        load += prerequisite_count
        
        # Add for concept length (complex terms)
        if len(concept) > 12:
            load += 1
        
        return min(load, 10)
    
    def _apply_domain_rules(self, base_difficulty: DifficultyLevel, 
                           concept_type: str, semantic_family: str) -> DifficultyLevel:
        """Apply domain-specific adjustment rules"""
        
        # Chemistry rules
        if "acid" in semantic_family or "formula" in semantic_family:
            if concept_type == "formula":
                return DifficultyLevel.INTERMEDIATE
        
        # Biology rules  
        if "life_processes" in semantic_family:
            if concept_type == "process":
                return DifficultyLevel.INTERMEDIATE
        
        # Physics rules
        if "electricity" in semantic_family or "optics" in semantic_family:
            return DifficultyLevel.ADVANCED
        
        return base_difficulty
    
    def _calculate_educational_importance(self, concept: str, frequency: int,
                                        semantic_family: str, chapter: str) -> float:
        """Calculate educational importance (0-1 scale)"""
        
        importance = 0.0
        
        # Base importance from frequency (normalized)
        importance += min(frequency / 100.0, 0.4)  # Max 40% from frequency
        
        # Add for being in core semantic families
        core_families = ["acid_base_family", "life_processes_family", "cell_structure_family"]
        if semantic_family in core_families:
            importance += 0.3
        
        # Add for being a prerequisite
        if concept in self.prerequisite_chains:
            importance += 0.2
        
        # Add for fundamental concepts
        fundamental_concepts = ["acid", "base", "cell", "light", "current", "oxygen", "water"]
        if concept in fundamental_concepts:
            importance += 0.3
        
        return min(importance, 1.0)
    
    def _determine_final_difficulty(self, base_difficulty: DifficultyLevel,
                                  prerequisite_adjustment: int, cognitive_load: int) -> DifficultyLevel:
        """Determine final difficulty level"""
        
        # Start with base difficulty
        difficulty_score = base_difficulty.value
        
        # Adjust for prerequisites (more prerequisites = harder)
        if prerequisite_adjustment >= 3:
            difficulty_score += 1
        elif prerequisite_adjustment >= 1:
            difficulty_score += 0.5
        
        # Adjust for cognitive load
        if cognitive_load >= 8:
            difficulty_score += 1
        elif cognitive_load >= 6:
            difficulty_score += 0.5
        
        # Convert back to enum
        final_score = min(int(round(difficulty_score)), 4)
        return DifficultyLevel(final_score)
    
    def _generate_classification_reason(self, concept: str, semantic_family: str,
                                      difficulty: DifficultyLevel, bloom_level: BloomLevel) -> str:
        """Generate human-readable explanation for classification"""
        
        family_desc = self.semantic_families.get(semantic_family, {}).get("description", semantic_family)
        prerequisites = self.prerequisite_chains.get(concept, [])
        
        reason = f"{difficulty.name.title()} level: {family_desc}"
        
        if prerequisites:
            reason += f". Requires: {', '.join(prerequisites)}"
        
        reason += f". Bloom level: {bloom_level.name.title()}"
        
        return reason
    
    def classify_concepts_for_quiz(self, concepts: List[Dict], 
                                  target_distribution: Dict[str, float] = None,
                                  mode: str = "quiz",
                                  complexity_level: str = "standard") -> Dict[str, List[Dict]]:
        """Classify concepts and distribute them for quiz generation with context awareness"""
        
        # Adjust distribution based on mode and complexity
        if target_distribution is None:
            target_distribution = self._get_contextual_distribution(mode, complexity_level)
        
        classified_concepts = {
            "beginner": [],
            "intermediate": [],
            "advanced": [],
            "expert": []
        }
        
        for concept_data in concepts:
            classification = self.classify_concept(
                concept=concept_data.get("concept", ""),
                concept_type=concept_data.get("type", "term"),
                frequency=concept_data.get("frequency", 0),
                definition=concept_data.get("definition"),
                chapter=concept_data.get("chapter", "")
            )
            
            # Adjust classification based on complexity level
            adjusted_classification = self._adjust_for_complexity(classification, complexity_level)
            
            # Add classification info to concept data
            enhanced_concept = concept_data.copy()
            enhanced_concept.update({
                "difficulty_level": adjusted_classification.difficulty.name.lower(),
                "bloom_level": adjusted_classification.bloom_level.name.lower(),
                "semantic_family": adjusted_classification.semantic_family,
                "prerequisites": adjusted_classification.prerequisites,
                "cognitive_load": adjusted_classification.cognitive_load,
                "educational_importance": adjusted_classification.educational_importance,
                "classification_reason": adjusted_classification.reason,
                "complexity_adjusted": complexity_level != "standard"
            })
            
            # Add to appropriate difficulty bucket
            difficulty_key = adjusted_classification.difficulty.name.lower()
            classified_concepts[difficulty_key].append(enhanced_concept)
        
        # Sort each difficulty level by educational importance
        for difficulty_level in classified_concepts:
            classified_concepts[difficulty_level].sort(
                key=lambda x: x["educational_importance"], reverse=True
            )
        
        return classified_concepts
    
    def _get_contextual_distribution(self, mode: str, complexity_level: str) -> Dict[str, float]:
        """Get distribution based on quiz mode and complexity level"""
        
        base_distributions = {
            "quiz": {
                "beginner": 0.6,      # 60% foundational for quick quiz
                "intermediate": 0.3,  # 30% process understanding
                "advanced": 0.1,      # 10% application
                "expert": 0.0         # 0% synthesis for short quiz
            },
            "exam": {
                "beginner": 0.4,      # 40% foundational
                "intermediate": 0.4,  # 40% process understanding
                "advanced": 0.15,     # 15% application
                "expert": 0.05        # 5% synthesis
            }
        }
        
        base_dist = base_distributions.get(mode, base_distributions["quiz"])
        
        # Adjust for complexity level
        if complexity_level == "simple":
            # Shift towards easier concepts
            base_dist["beginner"] += 0.2
            base_dist["intermediate"] -= 0.1
            base_dist["advanced"] = max(0, base_dist["advanced"] - 0.1)
            base_dist["expert"] = 0
            
        elif complexity_level == "advanced":
            # Shift towards harder concepts
            base_dist["beginner"] -= 0.2
            base_dist["intermediate"] += 0.1
            base_dist["advanced"] += 0.1
            
        # Normalize to ensure sum = 1.0
        total = sum(base_dist.values())
        return {k: v/total for k, v in base_dist.items()}
    
    def _adjust_for_complexity(self, classification: ConceptClassification, 
                              complexity_level: str) -> ConceptClassification:
        """Adjust concept classification based on complexity level"""
        
        if complexity_level == "simple":
            # Simplify advanced concepts
            if classification.difficulty == DifficultyLevel.EXPERT:
                new_difficulty = DifficultyLevel.ADVANCED
            elif classification.difficulty == DifficultyLevel.ADVANCED:
                new_difficulty = DifficultyLevel.INTERMEDIATE
            else:
                new_difficulty = classification.difficulty
                
            # Simplify Bloom level
            if classification.bloom_level == BloomLevel.ANALYZE:
                new_bloom = BloomLevel.UNDERSTAND
            elif classification.bloom_level == BloomLevel.APPLY:
                new_bloom = BloomLevel.UNDERSTAND
            else:
                new_bloom = classification.bloom_level
                
            # Reduce cognitive load
            new_cognitive_load = max(1, classification.cognitive_load - 2)
            
            new_reason = f"Simplified: {classification.reason}"
            
        elif complexity_level == "advanced":
            # Keep or enhance complexity
            new_difficulty = classification.difficulty
            new_bloom = classification.bloom_level
            new_cognitive_load = classification.cognitive_load
            new_reason = f"Advanced context: {classification.reason}"
            
        else:  # standard
            new_difficulty = classification.difficulty
            new_bloom = classification.bloom_level
            new_cognitive_load = classification.cognitive_load
            new_reason = classification.reason
        
        return ConceptClassification(
            concept=classification.concept,
            difficulty=new_difficulty,
            bloom_level=new_bloom,
            semantic_family=classification.semantic_family,
            prerequisites=classification.prerequisites,
            cognitive_load=new_cognitive_load,
            educational_importance=classification.educational_importance,
            reason=new_reason
        )
    
    def get_question_length_recommendation(self, mode: str, complexity_level: str) -> int:
        """Recommend appropriate question count based on mode and complexity"""
        
        base_counts = {
            "quiz": 5,
            "exam": 15
        }
        
        base_count = base_counts.get(mode, 5)
        
        if complexity_level == "simple":
            return max(3, base_count - 2)  # Shorter for simple
        elif complexity_level == "advanced":
            return base_count + 3  # Longer for advanced
        else:
            return base_count 