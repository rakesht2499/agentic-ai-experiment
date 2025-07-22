from typing import List, Literal, Optional, override
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext

from exam_generating_agent_new.prompts import QUIZ_PREP_ORCHESTRATOR_PROMPT
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

from exam_generating_agent_new.prompts import instructions_for_question_input_validator


class QuestionGenerationInput(BaseModel):
    mode: Literal["quiz", "exam"] = Field(..., description="Whether to generate a short quiz or a full exam paper.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="Role of the user requesting the quiz/exam.")
    subject: str = Field(..., description="The subject to generate the quiz/exam for, e.g., 'Science'.")
    class_: str = Field(..., description="The grade level for the quiz/exam, e.g., 'Class 8'.")
    chapters: Optional[List[str]] = Field(None,
                                          description="List of chapters to cover. If not provided, use entire syllabus.")
    language: Optional[str] = Field("english", description="Preferred language of the generated questions.")
    question_count: Optional[int] = Field(None, description="How many questions to generate (optional override).")

class RagAgentInput(BaseModel):
    subject: str = Field(..., description="E.g., Science")
    class_: str = Field(..., description="E.g., Class 7")
    chapters: List[str] = Field(..., description="List of chapters to fetch")

class RagAgentOutput(BaseModel):
    context: str = Field(..., description="Retrieved content or fallback message")


# --- Sub-Agents --- #
clarifier_agent = LlmAgent(
    name="QuizClarifierAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are an input clarification assistant for quiz and exam generation.

    You will receive a structured input with fields:
    - class_: Grade level (e.g., Class 6, Class 10)
    - subject: Academic subject (e.g., Science, Math)
    - chapters: List of chapters to cover (optional)
    - mode: 'quiz' or 'exam'
    - role: 'teacher', 'parent', or 'student'
    - question_count: (optional) how many questions to generate
    - language: Preferred output language (default is English)

    ---

    If **any of these required fields are missing or unclear**, you must respond with a clarification:
    - class_
    - subject
    - chapters (optional but helpful)
    - mode
    - role ← 🛑 this is mandatory for tone & formatting

    ---

    ### Response format:

    If input is complete:
    ```json
    {
      "needs_clarification": false,
      "clarified_input": <exact input>
    },
    ```
    If anything is missing:
    ```json
    {
      "needs_clarification": true,
      "follow_up": "<Ask a clear and specific question to complete the missing input>"
    }
    Examples of follow_up:
    "Can you tell me the class/grade level for the quiz?"
    "What subject is this assessment for?"
    "Is this for a teacher, parent, or student?"
    Never assume or invent values.
    
    ## ⚠️ Important Notes
    
    - `"role"` is a **critical path field** because it determines formatting agent.
    - If it's missing and you don’t clarify it → formatter routing fails or becomes ambiguous.
    - This updated instruction also encourages **natural language clarifying questions**.
    """,
    input_schema=QuestionGenerationInput
)


# 1. Input Validator Agent
input_validator_agent = LlmAgent(
    name="QuestionInputValidatorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=instructions_for_question_input_validator,
)

teacher_quiz_formatter_agent = LlmAgent(
    name="TeacherQuizFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are formatting a quiz or exam for a teacher to print or assign in class.

1. Present questions in a clean numbered list.
2. Maintain proper structure with:
   - ✅ Marks per question
   - ✅ Section headers (if exam)
   - ✅ Blank space for answers
3. Do not simplify or answer the questions.
4. Preserve subject-specific terminology.

At the end, add:
📘 For: Class {{ input.class_ }}, Subject: {{ input.subject }}, Chapters: {{ chapter_list }}
"""
)

parent_quiz_formatter_agent = LlmAgent(
    name="ParentQuizFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are helping a parent revise a quiz or exam with their child at home.

1. Reword questions in simple, parent-friendly language.
2. Add short encouraging tips after each question like:
   - "You’ve got this!"
   - "Let’s recall what we saw in the diagram!"
3. Keep it warm and easy to read. Use everyday vocabulary.
4. Do not answer the questions.

At the end:
- ✅ "You did a great job guiding your child!"
- 📘 Based on: Class {{ input.class_ }}, Subject: {{ input.subject }}
"""
)

rag_agent = LlmAgent(
    name="RagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are connected to NCERT textbook content.

Given a query and metadata like subject and chapter, retrieve relevant textbook content from the textbook.

Otherwise, return the most relevant answer from the textbook. Do not invent or guess.
"""
)

student_quiz_formatter_agent = LlmAgent(
    name="StudentQuizFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are a quiz master giving an interactive quiz directly to a student.
    
    🧠 Here's how you behave:
    - Present **only one** question at a time.
    - After each question, wait for the student's answer before sending the next.
    - Use friendly, encouraging language.
    - After 2–3 questions, check in:
       - "Need a break?"
       - "Want a hint?"
       - "You’re doing amazing, keep going! ⚡"
    
    🤖 Important:
    - Do **not** give answers.
    - Keep track of which questions have been asked.
    - Never repeat a question.
    - At the end, say:
      ✅ "Great work!"
      📘 "Questions from Class {{ input.class_ }}, Subject: {{ input.subject }}, Chapters: {{ input.chapter_list }}"
    
    🎯 Example Interaction Flow:
    1. "Ready to test your knowledge on Heredity? Let’s go! 🤓"
    2. "Q1: What is the term for the study of how traits are passed from one generation to the next?"
    ⏸️ *[Wait for response]*
    3. "Awesome! Let’s try another one. ⚡"
    4. "Q2: When Gregor Mendel crossed a pure tall and pure dwarf pea plant..."
    
    Stay in this loop until all questions are asked.
    """
)


class QuizPrepTool(BaseTool):
    name = "QuizPrepTool"
    description = "Generates a quiz or exam paper with creative, non-repetitive questions for a given subject and class."

    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        input_data = QuestionGenerationInput(**context.input.dict())

        # Compose the task instruction
        prompt = f"""
    You are a quiz and exam generator. Create creative and original questions **based on NCERT-style curriculum**.

    - Class: {input_data.class_}
    - Subject: {input_data.subject}
    - Chapters: {', '.join(input_data.chapters) if input_data.chapters else 'Full syllabus'}
    - Mode: {"Short quiz" if input_data.mode == "quiz" else "Full-length exam"}
    - Language: {input_data.language}
    - Role: {input_data.role}
    - Question count: {input_data.question_count or ('5' if input_data.mode == 'quiz' else '15')}

    Role-based tone:
    - 👩‍🏫 **Teacher** → Balanced, instructional phrasing. Add marks or difficulty where relevant.
    - 👩‍👧 **Parent** → Friendly tone, questions should be supportive and engaging for kids.
    - 👨‍🎓 **Student** → Encouraging tone. Add a mix of simple and challenging questions.

    ⚠️ Don’t repeat questions if the tool is called multiple times.
    📌 Prefer real-world relatable examples when possible.
    """

        return prompt.strip()

quiz_prep_tool = QuizPrepTool(name="QuizPrepToolAgent", description="Quiz Prep Tool")

quiz_generator_agent = LlmAgent(
    name="QuizGeneratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are an AI educator tasked with generating highly engaging and non-repetitive quiz or exam questions.

Always use the QuizPrepTool to fetch:
- Class
- Subject
- Mode (quiz or exam)
- Chapters (if specified)
- Role-specific tone
- Language preference
- Question count

✅ Your job is to generate only the questions — do NOT explain answers.
✅ Avoid repetition in wording or structure even across multiple calls.
✅ Always follow the tone appropriate to the role (teacher, parent, or student).
✅ Align with NCERT-style content and level.
""",
    tools=[quiz_prep_tool],
)

quiz_prep_orchestrator_agent = LlmAgent(
    name="QuizPrepOrchestratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction=QUIZ_PREP_ORCHESTRATOR_PROMPT,
    input_schema=QuestionGenerationInput,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=input_validator_agent),
        agent_tool.AgentTool(agent=rag_agent),
        agent_tool.AgentTool(agent=quiz_generator_agent),
        agent_tool.AgentTool(agent=teacher_quiz_formatter_agent),
        agent_tool.AgentTool(agent=parent_quiz_formatter_agent),
        agent_tool.AgentTool(agent=student_quiz_formatter_agent),
    ],
)

root_agent=quiz_prep_orchestrator_agent