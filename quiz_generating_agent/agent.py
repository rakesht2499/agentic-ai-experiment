from typing import List, Literal, Optional, override
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext

from common_agents import role_formatter_agent
from common_agents.gcs_uploader_tool import GcsUploaderTool
from common_agents.pdf_generation_tool import generate_pdf_with_answers
from common_agents.shared_rag_agent import shared_rag_agent, shared_rag_role_inspector
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

from quiz_generating_agent.prompts import instructions_for_question_input_validator

import os


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
    - If it's missing and you don't clarify it → formatter routing fails or becomes ambiguous.
    - This updated instruction also encourages **natural language clarifying questions**.
    """,
    input_schema=QuestionGenerationInput
)

input_validator_agent = LlmAgent(
    name="QuestionInputValidatorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=instructions_for_question_input_validator,
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

    ⚠️ Don't repeat questions if the tool is called multiple times.
    📌 Prefer real-world relatable examples when possible.
    """

        return prompt.strip()

quiz_prep_tool = QuizPrepTool(name="QuizPrepToolAgent", description="Quiz Prep Tool")
gcs_uploader_tool = GcsUploaderTool()

# NEW PDF Generator Tool
class PDFGeneratorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="PDFGeneratorTool",
            description = "Generates a PDF with questions and answers and uploads it to GCS."
        )

    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        input_data = context.input.dict()
        questions: List[str] = input_data.get("questions", [])
        answers: List[str] = input_data.get("answers", [])
        subject: str = input_data.get("subject")
        class_: str = input_data.get("class_")
        mode: str = input_data.get("mode", "quiz")

        filename = f"{subject}_{class_}_{mode}.pdf".replace(" ", "_").lower()
        filepath = f"/tmp/{filename}"

        generate_pdf_with_answers(questions, answers, filepath)  # implement this method

        upload_input = {
            "type": mode,
            "path": filepath
        }

        # Reuse the uploader tool
        await gcs_uploader_tool.run_async(
            InvocationContext(input=upload_input), tool_context
        )

        return f"PDF successfully created and uploaded: {filepath}"

# pdf_generator_tool = PDFGeneratorTool()

quiz_generator_agent = LlmAgent(
    name="QuizGeneratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are an AI educator tasked with generating highly engaging and non-repetitive quiz or exam questions.

You will receive input from the previous step (SharedRagAgent) in the format:
{"subject": "Science", "class_": "Class 10", "content": "NCERT textbook content..."}

Process:
1. Extract the subject, class, and textbook content from the previous step
2. Use the QuizPrepTool to fetch additional parameters (role, mode, language, etc.)
3. Generate questions based on the textbook content and quiz parameters

Always use the QuizPrepTool to fetch:
- Class (verify against SharedRagAgent output)
- Subject (verify against SharedRagAgent output)  
- Mode (quiz or exam)
- Chapters (if specified)
- Role-specific tone
- Language preference
- Question count

✅ Your job is to generate only the questions based on the NCERT content from SharedRagAgent — do NOT explain answers.
✅ If the content field contains "RAG_RETRIEVAL_FAILED", create general questions for the subject/class and mention the limitation.
✅ Avoid repetition in wording or structure even across multiple calls.
✅ Always follow the tone appropriate to the role (teacher, parent, or student).
✅ Align questions with the specific NCERT content provided by SharedRagAgent.
""",
    tools=[quiz_prep_tool],
)


processing_agent = SequentialAgent(
    name="ProcessingAgent",
    sub_agents=[
        shared_rag_agent,
        quiz_generator_agent,

        # NEW agent: Generate PDF with answers
        LlmAgent(
            name="QuizPDFGeneratorAgent",
            model=GEMINI_PRO_MODEL,
            instruction="""
            You are a document publishing assistant.

            Input will contain:
            - questions: List of all generated questions
            - answers: List of corresponding answers
            - subject, class_, mode: for naming the output file

            Process:
            1. Format all questions neatly for printing
            2. On a new page, format the answers under "Answer Key"
            3. Generate a pdf out of questions & answers & store it in your local, then give me the path you've saved the file
            4. Call gcs_uploader_tool with the following parameters:
            ```json
            local_file: Path to the local pdf file you created
            gcs_uri:  gs://shahayak-agentic-ai-gpl-muskeeters/quiz/test_paper.pdf
            content_type: "quiz"
            ```

            ✅ Output only the success message or failure info returned from gcs_uploader_tool
            ❌ Do not modify questions or answers
            """,
            tools=[gcs_uploader_tool]
        ),

        # Existing role formatter
        LlmAgent(
            name="RoleFormatterAgent",
            model=GEMINI_PRO_MODEL,
            instruction="""
            1. Call shared_rag_role_inspector to get the user's role
            2. Call role_formatter_agent with:
               - role: user's role from step 1
               - content: Quiz content from previous step
               - formatter_type: "quiz"
            3. Handle the JSON response:
               - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your quiz. Please try again."
               - If error_logs is empty: Return the formatter_content as the final response
            """,
            tools=[shared_rag_role_inspector, role_formatter_agent]
        )
    ],
    description="Handles RAG retrieval, quiz generation, PDF creation, and role-based formatting sequentially"
)

quiz_generating_agent = LlmAgent(
    name="quiz_generating_agent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are the quiz/exam generation orchestrator.

    Step-by-step:

    1. First, call **QuizClarifierAgent** with the raw user input.

    2. If the response includes `"needs_clarification": true`, return the `"follow_up"` field as your output. Do NOT proceed further.

    3. If `"needs_clarification": false`, extract `"clarified_input"` and use that as your **new input** to call **ProcessingAgent**.

    ⚠️ NEVER skip calling QuizClarifierAgent first.
    ⚠️ NEVER assume any required field like `role` or `subject`—clarify it first.
    ⚠️ If you proceed without clarified input, it may crash the tool due to schema validation failure.

    ALWAYS trust clarified_input before passing it to any downstream agent or tool.
    """,
    input_schema=QuestionGenerationInput,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ],
)

root_agent=quiz_generating_agent