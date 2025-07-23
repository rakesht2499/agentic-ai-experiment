from typing import List, Literal, Optional, override
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import agent_tool, BaseTool, ToolContext

from common_agents import role_formatter_agent
from exam_generating_agent_new.prompts import QUIZ_PREP_ORCHESTRATOR_PROMPT
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

from exam_generating_agent_new.prompts import instructions_for_question_input_validator


def quiz_rag_postprocess_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    callback_context.state["role"] = "student"  # Will be overridden by orchestrator input
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[Quiz Callback] Inspected original response text: '{original_text[:100]}...'")
            if original_text == "RAG_RETRIEVAL_FAILED":
                callback_context.state["rag_failed"] = True
            else:
                callback_context.state["rag_failed"] = False
            return llm_response
        elif llm_response.content.parts[0].function_call:
            print(f"[Quiz Callback] Inspected response: Contains function call '{llm_response.content.parts[0].function_call.name}'. No text modification.")
            return None
        else:
            print("[Quiz Callback] Inspected response: No text content found.")
            return None
    elif llm_response.error_message:
        print(f"[Quiz Callback] Inspected response: Contains error '{llm_response.error_message}'. No modification.")
        return None
    else:
        print("[Quiz Callback] Inspected response: Empty LlmResponse.")
        return None


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

rag_agent = LlmAgent(
    name="RagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are connected to NCERT textbook content.

Given a query and metadata like subject and chapter, retrieve relevant textbook content from the textbook.

Otherwise, return the most relevant answer from the textbook. Do not invent or guess.
""",
    after_model_callback=quiz_rag_postprocess_callback
)

class RoleInspectorTool(BaseTool):
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        return context.session.state.get("role", "unknown")

role_inspector_tool = RoleInspectorTool(name="RoleInspectorTool", description="Inspects the current user role from the agent state context and returns it.")

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


processing_agent = SequentialAgent(
    name="ProcessingAgent",
    sub_agents=[
        rag_agent,
        quiz_generator_agent,
        LlmAgent(
            name="RoleFormatterAgent",
            model=GEMINI_PRO_MODEL,
            instruction="""
            1. Call RoleInspectorTool to get the user's role
            2. Call role_formatter_agent with:
               - role: user's role from step 1
               - content: Quiz content from previous step
               - formatter_type: "quiz"
            3. Handle the JSON response:
               - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your quiz. Please try again."
               - If error_logs is empty: Return the formatter_content as the final response
            """,
            tools=[role_inspector_tool, role_formatter_agent]
        )
    ],
    description="Handles RAG retrieval, quiz generation, and role-based formatting sequentially"
)

quiz_prep_orchestrator_agent = LlmAgent(
    name="QuizPrepOrchestratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are a quiz/exam generation orchestrator with clear separation of concerns:

    1. **First, call QuizClarifierAgent** to check if the user input is complete and clear.

    2. **Check the clarifier response**:
       - If it contains "needs_clarification": true → Return the follow_up question directly to user and STOP
       - If it contains "needs_clarification": false → Continue to step 3

    3. **Call ProcessingAgent** to handle RAG retrieval, quiz generation, and role-based formatting.

    IMPORTANT: Never call ProcessingAgent if clarification is needed. Always ask user first.
    IMPORTANT: Do not call clarifierAgent once you receive the output from ProcessingAgent
    """,
    input_schema=QuestionGenerationInput,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ],
)

root_agent=quiz_prep_orchestrator_agent