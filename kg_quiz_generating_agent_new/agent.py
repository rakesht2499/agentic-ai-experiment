from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext
import json
import re
from pathlib import Path

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func

from common_agents import role_formatter_agent
from common_agents.shared_rag_agent import shared_rag_role_inspector, clone_agent, shared_rag_agent
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL
from kg_quiz_generating_agent_new.educational_difficulty_classifier import EducationalDifficultyClassifier

class QuizGenerationInput(BaseModel):
    mode: Literal["quiz", "exam"] = Field(..., description="Whether to generate a short quiz or a full exam paper.")
    role: Literal["teacher", "parent", "student"] = Field("student", description="Role of the user requesting the quiz/exam.")
    subject: str = Field(..., description="The subject to generate the quiz/exam for, e.g., 'Science'.")
    class_: str = Field(..., description="The grade level for the quiz/exam, e.g., 'Class 8'.")
    chapters: Optional[List[str]] = Field(None, description="List of chapters to cover. If not provided, use entire syllabus.")
    language: Optional[str] = Field("english", description="Preferred language of the generated questions.")
    question_count: Optional[int] = Field(None, description="How many questions to generate (optional override).")
    difficulty: Optional[Literal["easy", "medium", "hard", "mixed"]] = Field("mixed", description="Difficulty level for questions.")

class KGConceptSelection(BaseModel):
    """Enhanced concept selection with semantic classification"""
    beginner_concepts: List[Dict] = Field(..., description="Foundation concepts and definitions")
    intermediate_concepts: List[Dict] = Field(..., description="Processes, relationships, and applications") 
    advanced_concepts: List[Dict] = Field(..., description="Complex applications and analysis")
    expert_concepts: List[Dict] = Field(..., description="Synthesis, evaluation, and creation")
    definitions: List[Dict] = Field(..., description="Concepts with clear definitions")
    formulas: List[Dict] = Field(..., description="Chemical formulas and equations")
    bloom_distribution: Dict[str, List[Dict]] = Field(..., description="Concepts grouped by Bloom's taxonomy")
    semantic_families: Dict[str, List[Dict]] = Field(..., description="Concepts grouped by semantic family")

class KnowledgeGraphSelector(BaseTool):
    """Enhanced tool using semantic-based difficulty classification"""
    
    def __init__(self, kg_concepts_file: str = "knowledge_graph/cbse_class10_science_kg_improved_concepts.json"):
        super().__init__(
            name="KnowledgeGraphSelector",
            description="Selects educational concepts using semantic analysis and educational hierarchy"
        )
        self.concepts = {}
        self.classifier = EducationalDifficultyClassifier()
        self.load_knowledge_graph(kg_concepts_file)
    
    def load_knowledge_graph(self, concepts_file: str):
        """Load knowledge graph concepts"""
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.concepts = json.load(f)
            print(f"✅ Loaded {len(self.concepts)} concepts with semantic classifier")
        except FileNotFoundError:
            print(f"⚠️ Knowledge graph not found: {concepts_file}")
            print("Run 'python build_knowledge_graph_improved.py' first!")
    
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        """Select concepts using semantic and educational hierarchy with context awareness"""
        
        input_data = QuizGenerationInput(**context.input.dict())
        
        # Determine complexity level based on multiple factors
        complexity_level = self._determine_complexity_level(input_data)
        
        # Filter concepts by chapter if specified
        chapter_filter = input_data.chapters[0] if input_data.chapters else None
        
        # Get concepts for the specified chapter or all concepts
        if chapter_filter:
            relevant_concepts = [
                c for c in self.concepts.values() 
                if c.get('chapter', '').lower() == chapter_filter.lower()
            ]
        else:
            relevant_concepts = list(self.concepts.values())
        
        # Filter out meaningless concepts and prioritize educational terms
        filtered_concepts = self._filter_educational_concepts(relevant_concepts)
        
        if not filtered_concepts:
            return json.dumps({"error": "No valid concepts found for the specified criteria"})
        
        # Use semantic classifier with context awareness
        classified_concepts = self.classifier.classify_concepts_for_quiz(
            filtered_concepts,
            mode=input_data.mode,
            complexity_level=complexity_level
        )
        
        # Get recommended question count
        recommended_count = self.classifier.get_question_length_recommendation(
            input_data.mode, complexity_level
        )
        
        # Group by Bloom's taxonomy levels
        bloom_distribution = self._group_by_bloom_levels(classified_concepts)
        
        # Group by semantic families
        semantic_families = self._group_by_semantic_families(classified_concepts)
        
        # Extract special categories
        definitions = self._extract_concepts_with_definitions(classified_concepts)
        formulas = self._extract_formulas(classified_concepts)
        
        # Adjust concept limits based on complexity and mode
        limits = self._get_concept_limits(input_data.mode, complexity_level)
        
        # Create enhanced selection with context metadata
        selection = KGConceptSelection(
            beginner_concepts=classified_concepts["beginner"][:limits["beginner"]],
            intermediate_concepts=classified_concepts["intermediate"][:limits["intermediate"]],
            advanced_concepts=classified_concepts["advanced"][:limits["advanced"]],
            expert_concepts=classified_concepts["expert"][:limits["expert"]],
            definitions=definitions[:limits["definitions"]],
            formulas=formulas[:limits["formulas"]],
            bloom_distribution=bloom_distribution,
            semantic_families=semantic_families
        )
        
        # Add context metadata
        selection_dict = selection.dict()
        selection_dict["context"] = {
            "complexity_level": complexity_level,
            "recommended_question_count": recommended_count,
            "mode": input_data.mode,
            "difficulty_preference": input_data.difficulty,
            "total_concepts_found": len(filtered_concepts)
        }
        
        return json.dumps(selection_dict, indent=2)
    
    def _determine_complexity_level(self, input_data: QuizGenerationInput) -> str:
        """Determine appropriate complexity level based on input context"""
        
        # Start with standard complexity
        complexity_score = 0
        
        # Adjust based on mode
        if input_data.mode == "quiz":
            complexity_score -= 1  # Simpler for quick quiz
        elif input_data.mode == "exam":
            complexity_score += 1  # More complex for exam
        
        # Adjust based on role
        if input_data.role == "parent":
            complexity_score -= 1  # Simpler for parent-led study
        elif input_data.role == "teacher":
            complexity_score += 1  # More sophisticated for teachers
        
        # Adjust based on difficulty preference
        if input_data.difficulty == "easy":
            complexity_score -= 2
        elif input_data.difficulty == "hard":
            complexity_score += 2
        elif input_data.difficulty == "mixed":
            complexity_score += 0  # Standard
        
        # Convert score to complexity level
        if complexity_score <= -2:
            return "simple"
        elif complexity_score >= 2:
            return "advanced"
        else:
            return "standard"
    
    def _get_concept_limits(self, mode: str, complexity_level: str) -> Dict[str, int]:
        """Get appropriate concept limits based on mode and complexity"""
        
        base_limits = {
            "quiz": {
                "beginner": 8, "intermediate": 6, "advanced": 3, "expert": 1,
                "definitions": 5, "formulas": 3
            },
            "exam": {
                "beginner": 12, "intermediate": 10, "advanced": 6, "expert": 4,
                "definitions": 8, "formulas": 6
            }
        }
        
        limits = base_limits.get(mode, base_limits["quiz"]).copy()
        
        if complexity_level == "simple":
            # Reduce advanced concepts, increase basic ones
            limits["beginner"] += 2
            limits["intermediate"] = max(2, limits["intermediate"] - 2)
            limits["advanced"] = max(1, limits["advanced"] - 2)
            limits["expert"] = 0
            limits["formulas"] = max(1, limits["formulas"] - 2)
            
        elif complexity_level == "advanced":
            # Increase advanced concepts
            limits["beginner"] = max(3, limits["beginner"] - 2)
            limits["intermediate"] += 2
            limits["advanced"] += 2
            limits["expert"] += 1
            
        return limits
    
    def _filter_educational_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Intelligently filter concepts to prioritize meaningful educational terms"""
        
        # Define patterns for meaningful educational concepts
        meaningful_patterns = [
            # Single scientific terms (high priority)
            r'^[a-z]+$',                    # photosynthesis, respiration, etc.
            r'^[a-z]+\s+[a-z]+$',          # carbon dioxide, amino acid, etc.
            
            # Chemical formulas (high priority)  
            r'^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$',  # H2O, CO2, NaCl, etc.
            
            # Scientific processes (medium priority)
            r'^[a-z]+tion$',               # respiration, digestion, circulation
            r'^[a-z]+sis$',                # photosynthesis, osmosis
        ]
        
        # Noise patterns to filter out (low priority/remove)
        noise_patterns = [
            r'^(this|that|these|those|the|an|a)\s',  # Articles + phrases
            r'^(questions?|activities?|exercises?)\s', # Textbook structure
            r'^(what|why|how|when|where)\s',          # Question words
            r'\s(and|or|but|is|are|was|were)\s*$',    # Sentence fragments
            r'\.{2,}',                                # Ellipsis fragments
            r'^.{50,}',                              # Very long phrases
        ]
        
        filtered = []
        
        for concept in concepts:
            concept_name = concept.get('concept', '').lower().strip()
            frequency = concept.get('frequency', 0)
            
            # Skip very short or empty concepts
            if len(concept_name) < 3:
                continue
                
            # Check if it matches noise patterns (skip if it does)
            is_noise = any(re.search(pattern, concept_name, re.IGNORECASE) 
                          for pattern in noise_patterns)
            if is_noise:
                continue
            
            # Calculate priority score
            priority_score = self._calculate_concept_priority(concept_name, frequency)
            
            # Add priority score to concept for sorting
            enhanced_concept = concept.copy()
            enhanced_concept['priority_score'] = priority_score
            
            # Only include concepts with reasonable priority
            if priority_score > 0.1:  # Threshold for inclusion
                filtered.append(enhanced_concept)
        
        # Sort by priority score (highest first) 
        filtered.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return filtered
    
    def _calculate_concept_priority(self, concept_name: str, frequency: int) -> float:
        """Calculate priority score for educational concepts"""
        
        score = 0.0
        
        # Base score from frequency (normalized)
        score += min(frequency / 50.0, 0.4)  # Max 40% from frequency
        
        # Bonus for scientific vocabulary
        scientific_terms = [
            'photosynthesis', 'respiration', 'digestion', 'circulation', 'excretion',
            'nutrition', 'metabolism', 'enzyme', 'glucose', 'oxygen', 'carbon',
            'chlorophyll', 'stomata', 'transpiration', 'diffusion', 'osmosis',
            'acid', 'base', 'salt', 'ph', 'litmus', 'indicator', 'neutralization',
            'cell', 'nucleus', 'cytoplasm', 'mitochondria', 'chloroplast',
            'protein', 'carbohydrate', 'vitamin', 'mineral', 'ion', 'element'
        ]
        
        for term in scientific_terms:
            if term in concept_name:
                score += 0.3  # Significant bonus for scientific terms
                break
        
        # Bonus for chemical formulas
        if re.match(r'^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$', concept_name):
            score += 0.25  # Bonus for chemical formulas
        
        # Bonus for single/double word scientific terms
        word_count = len(concept_name.split())
        if word_count == 1:
            score += 0.2  # Single words are often key terms
        elif word_count == 2:
            score += 0.1  # Two words can be good (carbon dioxide)
        elif word_count >= 5:
            score -= 0.3  # Long phrases are usually noise
        
        # Bonus for process terms (ending in -tion, -sis)
        if re.search(r'(tion|sis)$', concept_name):
            score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _group_by_bloom_levels(self, classified_concepts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Group concepts by Bloom's taxonomy levels"""
        bloom_groups = {
            "remember": [],
            "understand": [],
            "apply": [],
            "analyze": []
        }
        
        for difficulty_level in classified_concepts.values():
            for concept in difficulty_level:
                bloom_level = concept.get("bloom_level", "remember")
                if bloom_level in bloom_groups:
                    bloom_groups[bloom_level].append(concept)
        
        return bloom_groups
    
    def _group_by_semantic_families(self, classified_concepts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Group concepts by semantic families"""
        family_groups = {}
        
        for difficulty_level in classified_concepts.values():
            for concept in difficulty_level:
                family = concept.get("semantic_family", "general_science")
                if family not in family_groups:
                    family_groups[family] = []
                family_groups[family].append(concept)
        
        return family_groups
    
    def _extract_concepts_with_definitions(self, classified_concepts: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract concepts that have clear definitions"""
        definitions = []
        
        for difficulty_level in classified_concepts.values():
            for concept in difficulty_level:
                if concept.get("definition") and len(concept["definition"]) > 20:
                    definitions.append(concept)
        
        # Sort by educational importance
        definitions.sort(key=lambda x: x.get("educational_importance", 0), reverse=True)
        return definitions
    
    def _extract_formulas(self, classified_concepts: Dict[str, List[Dict]]) -> List[Dict]:
        """Extract chemical formulas and equations"""
        formulas = []
        
        for difficulty_level in classified_concepts.values():
            for concept in difficulty_level:
                if (concept.get("type") == "formula" or 
                    concept.get("semantic_family") == "formula_family"):
                    formulas.append(concept)
        
        # Sort by educational importance
        formulas.sort(key=lambda x: x.get("educational_importance", 0), reverse=True)
        return formulas

class KGQuizPrepTool(BaseTool):
    """Enhanced quiz prep tool that uses knowledge graph concepts"""
    
    def __init__(self):
        super().__init__(
            name="KGQuizPrepTool",
            description="Generates quiz parameters enhanced with knowledge graph concept selection"
        )
    
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        input_data = QuizGenerationInput(**context.input.dict())
        
        # Get context from KG selection (this will be available from previous tool call)
        complexity_level = "standard"  # Default, will be overridden by KG context
        
        # Create enhanced prompt with semantic and educational hierarchy guidance
        prompt = f"""
You are an advanced AI educator using semantic concept classification and educational hierarchy for quiz generation.

📊 **Quiz Parameters:**
- Class: {input_data.class_}
- Subject: {input_data.subject}
- Chapters: {', '.join(input_data.chapters) if input_data.chapters else 'Full syllabus'}
- Mode: {"Short quiz (3-5 questions)" if input_data.mode == "quiz" else "Full-length exam (10-15 questions)"}
- Difficulty: {input_data.difficulty}
- Language: {input_data.language}
- Role: {input_data.role}

🎯 **Context-Aware Question Strategy:**
Based on the KG analysis, you will receive concept selections with context metadata including:
- complexity_level: Determines language complexity and concept depth
- recommended_question_count: Optimal number of questions for the context
- semantic_families: Related concept groups for logical clustering

📚 **Adaptive Question Generation:**

**For SIMPLE complexity (struggling students, parent-led, easy difficulty):**
- Use everyday language: "plants" instead of "autotrophic organisms"
- Focus on foundational concepts: basic definitions and simple processes
- Short, clear questions with obvious answer choices
- Encourage confidence building with clear win questions
- Example: "What do plants need to make food?" vs "What are the raw materials for photosynthesis?"

**For STANDARD complexity (regular students, mixed difficulty):**
- Use appropriate scientific terms with explanations
- Balance foundational concepts with process understanding
- Include some application questions
- Mix question types for comprehensive coverage
- Example: "What raw materials do plants use for photosynthesis?" with options that teach

**For ADVANCED complexity (strong students, teachers, hard difficulty):**
- Use precise scientific terminology
- Include experimental methodology and applications
- Add analysis and synthesis questions
- Challenge students with multi-step reasoning
- Example: "Analyze the experimental setup used to demonstrate that CO2 is essential for photosynthesis"

🧠 **Language Adaptation Guidelines:**

**SIMPLE Level Language:**
- "Plants make food" (not "autotrophic nutrition")
- "Green parts of plants" (not "chlorophyll-containing structures")  
- "Plant breathing" (not "cellular respiration")
- Use analogies: "like a factory" or "like a kitchen"

**STANDARD Level Language:**
- "Photosynthesis" with brief explanation
- "Chlorophyll" with function explanation
- "Carbon dioxide and water" clearly stated
- Scientific terms with context

**ADVANCED Level Language:**
- "Autotrophic organisms" appropriately used
- "Light-dependent and light-independent reactions"
- "Experimental methodology" discussions
- Complex multi-concept relationships

🎭 **Role-Specific Adaptations:**

**👩‍👧 Parent Role:**
- Extra encouraging language: "Great job learning about..."
- Connect to daily life: "Like how we need food for energy..."
- Simpler explanations in answer feedback
- Positive reinforcement in question phrasing

**👨‍🎓 Student Role:**
- Motivational language: "Test your knowledge..."
- Progressive difficulty to build confidence
- Clear learning objectives in questions
- Hints within question phrasing

**👩‍🏫 Teacher Role:**
- Include pedagogical notes: "[Bloom Level: Understand]"
- Mark allocation suggestions: "[2 marks]"
- Common misconceptions to address
- Cross-curricular connections

📈 **Semantic Family-Based Question Flow:**
1. **Start with core family concepts** (highest educational_importance)
2. **Progress through prerequisite chains** logically
3. **Include cross-family connections** for synthesis
4. **End with application/analysis** appropriate to complexity level

🚀 **Final Instructions:**
- **ALWAYS check the context metadata** from KnowledgeGraphSelector to determine actual complexity_level
- **Adjust language complexity** based on the detected level (simple/standard/advanced)
- **Respect recommended_question_count** for appropriate quiz length
- **Use semantic_families** for logical question grouping
- **Prioritize high educational_importance concepts** from the selection
- **Include classification_reason insights** to inform question context

Generate questions that are perfectly calibrated to the student's level while leveraging intelligent concept selection!
"""
        
        return prompt.strip()

# Enhanced quiz generator that uses KG concepts
kg_quiz_generator_agent = LlmAgent(
    name="KGQuizGeneratorAgent", 
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are an advanced AI educator using semantic concept classification and educational hierarchy for quiz generation.
    
    You will receive:
    1. RAG content from SharedRagAgent_quiz: {"subject": "Science", "class_": "Class 10", "content": "textbook content..."}
    2. Enhanced KG concept selection from KnowledgeGraphSelector with context metadata:
       - complexity_level: "simple", "standard", or "advanced"
       - recommended_question_count: Optimal number for the context
       - beginner_concepts, intermediate_concepts, etc. with semantic families
       - bloom_distribution and semantic_families for intelligent grouping
    
    🎯 **Your Context-Aware Process:**
    1. **First**, call KnowledgeGraphSelector to get semantically classified concepts WITH context
    2. **Then**, call KGQuizPrepTool to get enhanced quiz parameters  
    3. **Parse the context metadata** to determine complexity_level and recommended_question_count
    4. **Generate questions** using intelligent concept selection AND appropriate language complexity
    
    📊 **Adaptive Question Strategy Based on Context:**
    
    **🟢 SIMPLE Complexity (quiz + easy + parent role):**
    - **Question Count**: 3-4 questions max
    - **Language**: Simple, everyday terms
    - **Focus**: Basic definitions and obvious processes
    - **Examples**: 
      - ❌ "What are the raw materials required for autotrophic nutrition?"
      - ✅ "What do plants need to make their own food?"
      - Options: a) Sunlight and soil b) Air and water c) Water and carbon dioxide d) Leaves and roots
    
    **🟡 STANDARD Complexity (quiz + mixed + student role):**
    - **Question Count**: 4-5 questions
    - **Language**: Scientific terms with explanations
    - **Focus**: Core concepts with some processes
    - **Examples**:
      - ✅ "What raw materials do plants use for photosynthesis?"
      - Options include brief explanations: a) Carbon dioxide and water (from air and soil) b) Oxygen and glucose...
    
    **🟠 ADVANCED Complexity (exam + hard + teacher role):**
    - **Question Count**: 6-8 questions for quiz, 12-15 for exam
    - **Language**: Precise scientific terminology
    - **Focus**: Experimental methodology, analysis, synthesis
    - **Examples**:
      - ✅ "In the experimental setup to demonstrate that CO2 is essential for photosynthesis, what is the role of KOH?"
      - Include multi-step reasoning and cross-concept connections
    
    🧠 **Intelligent Concept Usage:**
    
    **From beginner_concepts (high educational_importance):**
    - Use for foundational questions that build confidence
    - Prioritize concepts with clear definitions
    - Simple language even for scientific terms
    
    **From intermediate_concepts (semantic families):**
    - Use for process understanding questions
    - Group related concepts logically (acid-base family, life processes)
    - Standard scientific language with explanations
    
    **From advanced_concepts (prerequisite chains):**
    - Use for application and analysis questions
    - Respect learning dependencies
    - Precise terminology appropriate to complexity level
    
    **From semantic_families:**
    - Create question clusters around related concepts
    - Ensure logical flow from basic to complex within families
    - Connect across families for synthesis questions (advanced only)
    
    ✅ **Context-Responsive Generation Guidelines:**
    
    **Language Adaptation:**
    - **SIMPLE**: "plants" not "autotrophic organisms", "plant food-making" not "photosynthesis"
    - **STANDARD**: "photosynthesis" with brief explanation, scientific terms with context
    - **ADVANCED**: Full scientific terminology, experimental language, technical precision
    
    **Question Length Respect:**
    - **ALWAYS use recommended_question_count** from context metadata
    - Don't exceed limits even if more concepts are available
    - Quality over quantity - better fewer well-crafted questions
    
    **Bloom's Taxonomy Adaptation:**
    - **SIMPLE**: 80% Remember, 20% Understand (definitions and basic processes)
    - **STANDARD**: 50% Remember, 40% Understand, 10% Apply (balanced coverage)
    - **ADVANCED**: 30% Remember, 40% Understand, 20% Apply, 10% Analyze (higher-order thinking)
    
    🎭 **Role-Specific Language:**
    
    **Parent Context:**
    - Encouraging tone: "Great! Let's test what you've learned..."
    - Real-world connections: "Just like how we eat food for energy..."
    - Simple explanations in answer choices
    
    **Student Context:**
    - Motivational language: "Challenge yourself with..."
    - Confidence-building progression
    - Clear learning cues in questions
    
    **Teacher Context:**
    - Professional tone with pedagogical notes
    - Include Bloom level indicators: [Remember], [Understand], [Apply]
    - Mark allocation suggestions: [2 marks], [3 marks]
    
    🚨 **Critical Implementation:**
    1. **Parse context metadata** from KnowledgeGraphSelector result first
    2. **Extract complexity_level and recommended_question_count** before generating
    3. **Adapt ALL language** based on detected complexity level
    4. **Respect question count limits** strictly
    5. **Use semantic families** for logical concept grouping
    6. **Prioritize educational_importance** scores from concepts
    7. **Follow prerequisite chains** for logical progression
    
    Remember: Intelligent concept selection + appropriate complexity = perfect quiz for the learner!
    """,
    tools=[KnowledgeGraphSelector(), KGQuizPrepTool()]
)

# Clarifier agent (reusing from original)
clarifier_agent = LlmAgent(
    name="KGQuizClarifierAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
You are an input clarification assistant for KG-enhanced quiz generation.

Required fields:
- class_: Grade level (e.g., Class 6, Class 10)
- subject: Academic subject (e.g., Science, Math) 
- chapters: List of chapters (optional but recommended for KG concept selection)
- mode: 'quiz' or 'exam'
- role: 'teacher', 'parent', or 'student'
- difficulty: 'easy', 'medium', 'hard', or 'mixed' (optional, defaults to 'mixed')

If any required fields are missing, ask for clarification.

Response format:
```json
{
  "needs_clarification": false,
  "clarified_input": <complete input>
}
```

OR

```json
{
  "needs_clarification": true, 
  "follow_up": "<specific question to get missing info>"
}
```
""",
    input_schema=QuizGenerationInput
)

# Enhanced processing agent with KG integration
kg_processing_agent = SequentialAgent(
    name="KGProcessingAgent",
    sub_agents=[
        clone_agent(shared_rag_agent, "quiz"),  # Get RAG content
        kg_quiz_generator_agent,  # Generate questions using KG + RAG
        LlmAgent(
            name="KGRoleFormatterAgent",
            model=GEMINI_FLASH_MODEL,
            instruction="""
            1. Call shared_rag_role_inspector to get the user's role
            2. Call role_formatter_agent with:
               - role: user's role from step 1
               - content: Quiz content from previous step (KG-enhanced questions)
               - formatter_type: "quiz"
            3. Handle the JSON response:
               - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your quiz. Please try again."
               - If error_logs is empty: Return the formatter_content as the final response
            """,
            tools=[shared_rag_role_inspector, role_formatter_agent]
        )
    ],
    description="KG-enhanced quiz generation: RAG retrieval → KG concept selection → question generation → role formatting"
)

# Main orchestrator for KG-enhanced quiz generation
kg_quiz_prep_orchestrator_agent = LlmAgent(
    name="KGQuizPrepOrchestratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are a Knowledge Graph-enhanced quiz generation orchestrator.
    
    🔄 **Process Flow:**
    1. **Clarification**: Call KGQuizClarifierAgent to validate input completeness
    2. **If clarification needed**: Return the follow-up question and STOP
    3. **If input complete**: Call KGProcessingAgent for enhanced quiz generation
    
    🎯 **KG Enhancement Benefits:**
    - Systematic coverage of important educational concepts
    - Difficulty-based question distribution  
    - Chemical formula questions for chemistry
    - Process-based questions for biology
    - Definition questions for key terms
    
    ⚠️ **Critical Rules:**
    - NEVER re-call clarifier after processing
    - NEVER modify output from KGProcessingAgent
    - For teachers: Include answer keys when available
    - Always clearly label answer sections
    
    The KG enhancement ensures quizzes systematically cover the most educationally important concepts from each chapter!
    """,
    input_schema=QuizGenerationInput,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=kg_processing_agent)
    ]
)

# Export the main agent
root_agent = kg_quiz_prep_orchestrator_agent