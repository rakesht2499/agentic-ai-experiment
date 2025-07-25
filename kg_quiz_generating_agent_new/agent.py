from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext
import json
from pathlib import Path

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func

from common_agents import role_formatter_agent
from common_agents.shared_rag_agent import vector_rag_agent, shared_rag_role_inspector, clone_agent
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

class QuizGenerationInput(BaseModel):
    mode: Literal["quiz", "exam"] = Field(..., description="Whether to generate a short quiz or a full exam paper.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="Role of the user requesting the quiz/exam.")
    subject: str = Field(..., description="The subject to generate the quiz/exam for, e.g., 'Science'.")
    class_: str = Field(..., description="The grade level for the quiz/exam, e.g., 'Class 8'.")
    chapters: Optional[List[str]] = Field(None, description="List of chapters to cover. If not provided, use entire syllabus.")
    language: Optional[str] = Field("english", description="Preferred language of the generated questions.")
    question_count: Optional[int] = Field(None, description="How many questions to generate (optional override).")
    difficulty: Optional[Literal["easy", "medium", "hard", "mixed"]] = Field("mixed", description="Difficulty level for questions.")

class KGConceptSelection(BaseModel):
    """Selected concepts from knowledge graph for quiz generation"""
    easy_concepts: List[Dict] = Field(..., description="High-frequency fundamental concepts")
    medium_concepts: List[Dict] = Field(..., description="Medium-frequency processes and applications") 
    hard_concepts: List[Dict] = Field(..., description="Low-frequency specialized terms and formulas")
    definitions: List[Dict] = Field(..., description="Concepts with clear definitions")
    formulas: List[Dict] = Field(..., description="Chemical formulas and equations")

class KnowledgeGraphSelector(BaseTool):
    """Tool to select relevant concepts from knowledge graph based on chapter and difficulty"""
    
    def __init__(self, kg_concepts_file: str = "knowledge_graph/cbse_class10_science_kg_improved_concepts.json"):
        super().__init__(
            name="KnowledgeGraphSelector",
            description="Selects educational concepts from knowledge graph for targeted quiz generation"
        )
        self.concepts = {}
        self.load_knowledge_graph(kg_concepts_file)
    
    def load_knowledge_graph(self, concepts_file: str):
        """Load knowledge graph concepts"""
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.concepts = json.load(f)
            print(f"✅ Loaded {len(self.concepts)} concepts from knowledge graph")
        except FileNotFoundError:
            print(f"⚠️ Knowledge graph not found: {concepts_file}")
            print("Run 'python build_knowledge_graph_improved.py' first!")
    
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        """Select concepts based on input parameters"""
        
        input_data = QuizGenerationInput(**context.input.dict())
        
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
        
        # Group concepts by difficulty based on frequency and type
        easy_concepts = []
        medium_concepts = []  
        hard_concepts = []
        definitions = []
        formulas = []
        
        for concept in relevant_concepts:
            concept_type = concept.get('type', 'term')
            frequency = concept.get('frequency', 0)
            concept_name = concept.get('concept', '')
            definition = concept.get('definition')
            
            # Skip meaningless concepts
            if len(concept_name) < 3 or concept_name in ['this', 'that', 'what', 'there', 'when']:
                continue
            
            # Categorize by difficulty (frequency-based)
            if frequency >= 30:
                easy_concepts.append(concept)
            elif frequency >= 10:
                medium_concepts.append(concept)
            else:
                hard_concepts.append(concept)
            
            # Special categories
            if definition and len(definition) > 20:
                definitions.append(concept)
            
            if concept_type == 'formula':
                formulas.append(concept)
        
        # Sort by frequency (most important first)
        easy_concepts.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        medium_concepts.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        hard_concepts.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Limit concepts to prevent overwhelming
        selection = KGConceptSelection(
            easy_concepts=easy_concepts[:10],
            medium_concepts=medium_concepts[:8],
            hard_concepts=hard_concepts[:5],
            definitions=definitions[:8],
            formulas=formulas[:6]
        )
        
        return json.dumps(selection.dict(), indent=2)

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
        
        # Create enhanced prompt with KG guidance
        prompt = f"""
You are an advanced AI educator using knowledge graph-enhanced concept selection for quiz generation.

📊 **Quiz Parameters:**
- Class: {input_data.class_}
- Subject: {input_data.subject}
- Chapters: {', '.join(input_data.chapters) if input_data.chapters else 'Full syllabus'}
- Mode: {"Short quiz (5-8 questions)" if input_data.mode == "quiz" else "Full-length exam (15-20 questions)"}
- Difficulty: {input_data.difficulty}
- Language: {input_data.language}
- Role: {input_data.role}
- Question count: {input_data.question_count or ('6' if input_data.mode == 'quiz' else '15')}

🎯 **Question Distribution Strategy:**
- Easy (40%): High-frequency fundamental concepts and definitions
- Medium (40%): Processes, applications, and medium-frequency terms  
- Hard (20%): Specialized terms, chemical formulas, and complex relationships

📝 **Question Types to Include:**
1. **Definition Questions**: "What is [concept]?" using concepts with clear definitions
2. **Formula Recognition**: "What is the chemical formula for [compound]?" 
3. **Process Questions**: "Explain the process of [scientific process]"
4. **Application Questions**: "Give an example of [concept] in daily life"
5. **Multiple Choice**: Use related concepts as distractors

🎭 **Role-based Tone:**
- 👩‍🏫 **Teacher**: Professional, instructional. Include marks/difficulty indicators.
- 👩‍👧 **Parent**: Friendly, encouraging. Use simple language and relatable examples.
- 👨‍🎓 **Student**: Motivating, confidence-building. Mix easy wins with challenges.

🚀 **Enhanced Features:**
- Use high-frequency concepts for foundational questions
- Include chemical formulas for chemistry chapters  
- Focus on processes for biology/chemistry concepts
- Generate comprehensive coverage of chapter concepts
- Avoid repetitive question structures

Generate questions that systematically cover the most important educational concepts for maximum learning impact!
"""
        
        return prompt.strip()

# Enhanced quiz generator that uses KG concepts
kg_quiz_generator_agent = LlmAgent(
    name="KGQuizGeneratorAgent", 
    model=GEMINI_PRO_MODEL,
    instruction="""
You are an advanced AI educator using knowledge graph-enhanced quiz generation.

You will receive:
1. RAG content from SharedRagAgent_quiz: {"subject": "Science", "class_": "Class 10", "content": "textbook content..."}
2. KG concept selection from KnowledgeGraphSelector: Lists of concepts by difficulty, definitions, formulas
3. Quiz parameters from KGQuizPrepTool: Enhanced prompt with KG guidance

🎯 **Your Process:**
1. **First**, call KnowledgeGraphSelector to get relevant concepts for the chapter/subject
2. **Then**, call KGQuizPrepTool to get enhanced quiz generation parameters  
3. **Finally**, generate questions using BOTH the RAG content AND the selected KG concepts

📊 **Question Generation Strategy:**
- **Use KG concepts** to ensure you cover the most important educational topics
- **Use RAG content** to get accurate textbook information and context
- **Combine both** for comprehensive, targeted questions

🧪 **For Chemistry Chapters (like Chapter 2 - Acids, Bases, Salts):**
- Easy: "What is an acid?" (from definitions)
- Medium: "Explain neutralization process" (from processes) 
- Hard: "Balance: HCl + NaOH → ?" (from formulas)

🔬 **For Biology Chapters:**
- Easy: "What is photosynthesis?" (from definitions)
- Medium: "Explain how stomata work" (from processes)
- Hard: "What is the role of chloroplasts in glucose production?" (from relationships)

✅ **Quality Guidelines:**
- Generate ONLY questions (no answers unless role is teacher)
- Use concepts from KG selection to ensure educational relevance
- Cross-reference with RAG content for accuracy
- Follow role-specific tone from KGQuizPrepTool
- If RAG content contains "RAG_RETRIEVAL_FAILED", rely more heavily on KG concepts

🚨 **Critical**: Always call both KnowledgeGraphSelector AND KGQuizPrepTool before generating questions!
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
        clone_agent(vector_rag_agent, "quiz"),  # Get RAG content
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