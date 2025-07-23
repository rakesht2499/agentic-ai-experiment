from asyncio import Event
from typing import Optional, AsyncGenerator, override

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.models import LlmResponse
from google.adk.tools import agent_tool, google_search, BaseTool, ToolContext

from common_agents import role_formatter_agent
from models.constants import GEMINI_PRO_MODEL
from q_and_a_orchastrator_agent.prompts import QNA_ORCHESTRATOR_PROMPT
from diagram_generating_agent.agent import flowchart_agent

from pydantic import BaseModel, Field
from typing import Literal

class ClarifierInput(BaseModel):
    query: str = Field(..., description="User input that may be vague or incomplete.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="User's role to help shape clarifying response.")

def rag_postprocess_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    callback_context.state["role"] = "teacher"
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
    You are an expert teacher assistant helping a teacher explain a textbook concept to students in class.
    
    Structure:
    1. Begin with a teacher-friendly opener like: "Here’s how you can teach this:"
    2. Explain the concept in clear, simple, and structured steps.
    3. Add a relatable analogy using classroom or real-world objects.
    4. Suggest at least ONE of the following:
       - A chalkboard diagram idea
       - A simple classroom activity
       - A quick discussion question
    
    Constraints:
    - Keep tone professional but warm (not robotic).
    - Do not invent information. Stick to textbook facts.
    
    Wrap with:
    - ✅ A 1-line recap
    - 📍 "Based on Chapter <chapter name>"
    """
)

parent_formatter_agent = LlmAgent(
    name="ParentFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are a supportive helper guiding a parent in explaining a textbook concept to their child.

Tone & Style:
- Use a kind, encouraging, empathetic tone.
- Assume the parent may not remember school concepts.
- Never use complex or academic language.

Structure:
1. Start with reassurance: "No worries! Here’s how you can explain it at home:"
2. Simplify the concept in everyday words.
3. Use an example from daily life (cooking, walking, shopping, home chores).
4. Gently offer to translate: "Would you like this in your local language?"

Wrap with:
- ✅ “You did a great job explaining this!”
- Mention TranslatorAgent only if language ≠ English.
"""
)

student_formatter_agent = LlmAgent(
    name="StudentFormatterAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are a cheerful, encouraging tutor helping a student understand a textbook concept.

Tone:
- Friendly, fun, clear — like a favorite older sibling or coach.
- Encourage the student and make them feel confident.

Structure:
1. Begin with: "Let’s learn this together!"
2. Explain in 2–3 steps using short, simple sentences.
3. Add a memory hook using a relatable object: "Think of it like a sponge..."
4. End with a quick quiz or check-in: 
   - "Want to try a quick question?"
   - "What do you think happens next?"

Wrap with:
- ✨ The memory hook
- 📘 “Based on Chapter <chapter>”
"""
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

# --- Step 1: Clarification Exit Condition ---
def clarification_exit_condition(context):
    """Exit early if clarification is complete (needs_clarification = false)"""
    # Get the last response from clarifier_agent
    if hasattr(context, 'last_response') and context.last_response:
        try:
            # Parse the clarifier response to check if clarification is complete
            response_text = context.last_response.content.parts[0].text if context.last_response.content and context.last_response.content.parts else ""
            
            # Check if response contains needs_clarification: false
            if '"needs_clarification": false' in response_text or '"needs_clarification":false' in response_text:
                return True  # Exit the loop early
                
        except Exception:
            pass  # Continue loop if parsing fails
    
    return False  # Continue looping

# --- Step 1: Clarification Loop Agent (max 2 retries) ---
clarification_loop_agent = LoopAgent(
    name="ClarificationLoopAgent",
    sub_agent=clarifier_agent,
    max_iterations=2,
    exit_condition=clarification_exit_condition,
    description="Clarifies user query with up to 2 retries, exits early when complete"
)

# --- Step 2: RAG Retrieval Agent ---
rag_retrieval_agent = LlmAgent(
    name="RAGRetrievalAgent", 
    model=GEMINI_PRO_MODEL,
    instruction="""
    Call the RAG agent to retrieve textbook content based on the clarified query.
    If no content is found, set state['rag_failed'] = true.
    """,
    tools=[agent_tool.AgentTool(agent=rag_agent)]
)

# --- Step 3: Role-Based Formatting Agent ---
role_formatting_agent = LlmAgent(
    name="RoleFormattingAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    1. Call RoleInspectorTool to get the user's role
    2. Call role_formatter_agent with:
       - role: user's role from step 1
       - content: RAG response from previous step
       - formatter_type: "qna"
    3. Handle the JSON response:
       - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your answer. Please try asking your question again."
       - If error_logs is empty: Return the formatter_content as the final response
    """,
    tools=[
        role_inspector_tool,
        role_formatter_agent
    ]
)

# --- Step 4: Complete Q&A Orchestrator (Single Sequential Flow) ---
qna_orchestrator_agent = SequentialAgent(
    name="QnAOrchestratorAgent",
    sub_agents=[
        clarification_loop_agent,
        rag_retrieval_agent,
        role_formatting_agent
    ],
    input_schema=QnAOrchestratorInput,
    description="Complete Q&A orchestrator: Clarification → RAG retrieval → Role-based formatting"
)

root_agent = qna_orchestrator_agent


