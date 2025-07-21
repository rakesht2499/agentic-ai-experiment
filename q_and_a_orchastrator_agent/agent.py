from asyncio import Event
from typing import Optional, AsyncGenerator, override

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import LlmAgent
from google.adk.models import LlmResponse
from google.adk.tools import agent_tool, google_search, BaseTool, ToolContext

from models.constants import GEMINI_PRO_MODEL
from q_and_a_orchastrator_agent.prompts import QNA_ORCHESTRATOR_PROMPT
from diagram_generating_agent.agent import flowchart_agent

from pydantic import BaseModel, Field
from typing import Literal

class ClarifierInput(BaseModel):
    query: str = Field(..., description="User input that may be vague or incomplete.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="User's role to help shape clarifying response.")

def rag_postprocess_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    callback_context.state["role"] = "parent"
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[Callback] Inspected original response text: '{original_text[:100]}...'")  # Log snippet
            if original_text == "RAG_RETRIEVAL_FAILED":
                callback_context.state["rag_failed"] = True
            else:
                callback_context.state["rag_failed"] = False
            return llm_response
        elif llm_response.content.parts[0].function_call:
            print(
                f"[Callback] Inspected response: Contains function call '{llm_response.content.parts[0].function_call.name}'. No text modification.")
            return None  # Don't modify tool calls in this example
        else:
            print("[Callback] Inspected response: No text content found.")
            return None
    elif llm_response.error_message:
        print(f"[Callback] Inspected response: Contains error '{llm_response.error_message}'. No modification.")
        return None
    else:
        print("[Callback] Inspected response: Empty LlmResponse.")
        return None  # Nothing to modify

rag_agent = LlmAgent(
    name="RagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are connected to NCERT textbook content.

Given a query and metadata like subject and chapter, retrieve relevant textbook content from the textbook.

Otherwise, return the most relevant answer from the textbook. Do not invent or guess.
""",
    after_model_callback=rag_postprocess_callback
)

clarifier_agent = LlmAgent(
    name="ClarifierAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are an input clarification assistant.
    
    Given a user query and role (teacher, parent, student), determine:
    
    1. If the query is clear and answerable using textbook content:
       - Respond with: { "clarified_query": "<same as input>", "needs_clarification": false }
    
    2. If the query is vague or missing context (e.g., "explain this", "draw the diagram"):
       - Ask one short follow-up question to clarify intent.
       - Respond with: { "clarified_query": "", "needs_clarification": true, "follow_up": "Can you tell me which chapter or topic you're referring to?" }
    
    Role-based tone:
    - 👨‍🏫 Teacher → Direct and formal
    - 👩 Parent → Supportive and gentle
    - 👨‍🎓 Student → Friendly and patient
    
    Never guess. Always ask if uncertain.
    """,
)


class FormatterInput(BaseModel):
    raw_answer: str = Field(..., description="The answer retrieved from RAG or knowledge base.")
    query: str = Field(..., description="The original user question.")
    subject: str
    class_: str
    chapter: str

teacher_formatter_agent = LlmAgent(
    name="TeacherFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are an assistant helping a teacher explain a textbook concept to students in a classroom setting.
    
    1. Format the answer in a clear and structured teaching tone.
    2. Include a simple analogy if appropriate (e.g., real-life objects, classroom items).
    3. If the topic is suitable, suggest:
       - A chalkboard diagram idea
       - A quick activity or question the teacher can ask
    4. Maintain the original facts strictly — do not hallucinate.
    
    Wrap your final output with:
    - ✅ Summary line
    - 📍 Chapter reference
    """,
)

parent_formatter_agent = LlmAgent(
    name="ParentFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are helping a parent explain a concept to their child at home.

1. Use a warm, supportive tone.
2. Rephrase the raw answer in simple, easy-to-understand language.
3. Offer one example from daily life (e.g., cooking, walking, playing).
4. If possible, offer to translate key words or concepts into Hindi/Marathi (mention only, TranslatorAgent will do it).

At the end, add:
- ✅ "You did great explaining this!"
- Optional: "Would you like this in your local language?"
""",
)

student_formatter_agent = LlmAgent(
    name="StudentFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are a friendly tutor helping a student understand a textbook concept.

1. Break the explanation into 2–3 simple steps.
2. Use plain language, short sentences, and relatable comparisons.
3. At the end, ask a small check-in question:
   - "Want a quick quiz on this?"
   - "Can you guess what happens next?"

Maintain clarity. Avoid slang or jokes unless the question is casual.

Include:
- ✨ Memory hook (e.g., “Think of this like a sponge…”)
- 📘 “Based on Chapter __” at the end
""",
)

class TranslatorInput(BaseModel):
    content: str = Field(..., description="The formatted final answer to be translated.")
    target_language: str = Field(..., description="Language to translate into, e.g., 'Hindi', 'Marathi'")

translator_agent = LlmAgent(
    name="TranslatorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are a responsible education-grade translator.
    
    Translate the given content into the target language with:
    
    - Accurate concept preservation
    - Simplified local language (avoid over-academic tone)
    - No code-switching (don’t mix English mid-sentence)
    - No extra info — only translation
    
    Be sensitive to dialect and readability.
    """,
    )

from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str

def search_google(query: str) -> str:
    # Mock or real implementation via SerpAPI / custom search
    return "According to a recent article, Newton's laws are..."

from google.adk.agents import BaseAgent, InvocationContext
from typing import Any

# class FormatterRouterAgent(BaseAgent):
#     @override
#     async def _run_async_impl(
#             self, ctx: InvocationContext
#     ) -> AsyncGenerator[Event, None]:
#         role = ctx.session.state["role"]
#         if role == "teacher":
#             async for event in teacher_formatter_agent._run_async_impl(ctx):
#                 yield event
#             # yield teacher_formatter_agent._run_async_impl(ctx)
#         elif role == "parent":
#             async for event in parent_formatter_agent._run_async_impl(ctx):
#                 yield event
#         elif role == "student":
#             async for event in student_formatter_agent._run_async_impl(ctx):
#                 yield event
#         else:
#             raise NotImplementedError(
#                 f'role {ctx.session.state["role"]} is not supported.'
#             )
# formatter_router_agent = FormatterRouterAgent(name="FormatterRouterAgent")

class QnAOrchestratorInput(BaseModel):
    query: str = Field(..., description="User's question or voice-transcribed input.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="Role of the user.")
    board: str
    subject: str
    class_: str
    chapter: Optional[str] = None
    language: Optional[str] = Field("english", description="Preferred language for output")


from typing import Optional

class RoleInspectorTool(BaseTool):
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        return context.session.state.get("role", "unknown")

role_inspector_tool = RoleInspectorTool(name="RoleInspectorTool", description="Inspects the current user role from the agent state context and returns it.")

qna_orchestrator_agent = LlmAgent(
    name="QnAOrchestratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction=QNA_ORCHESTRATOR_PROMPT,
    tools=[
        role_inspector_tool,
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=rag_agent),
        agent_tool.AgentTool(agent=teacher_formatter_agent),
        agent_tool.AgentTool(agent=student_formatter_agent),
        agent_tool.AgentTool(agent=parent_formatter_agent),
        # agent_tool.AgentTool(agent=formatter_router_agent),
        agent_tool.AgentTool(agent=flowchart_agent),
        agent_tool.AgentTool(agent=translator_agent),
    ],
    input_schema=QnAOrchestratorInput,
)


root_agent=qna_orchestrator_agent


