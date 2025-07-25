from typing import Literal
from typing import Optional, override

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import agent_tool, BaseTool, ToolContext
from pydantic import BaseModel, Field

from common_agents import role_formatter_agent
from common_agents.shared_rag_agent import shared_rag_agent, shared_rag_role_inspector, vector_rag_agent
from models.constants import GEMINI_PRO_MODEL

class ClarifierInput(BaseModel):
    query: str = Field(..., description="User input that may be vague or incomplete.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="User's role to help shape clarifying response.")

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
    - No code-switching (don't mix English mid-sentence)
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

from google.adk.agents import InvocationContext

class AnswerOrchestratorInput(BaseModel):
    query: str = Field(..., description="User's question or voice-transcribed input.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="Role of the user.")
    board: str
    subject: str
    class_: str
    chapter: Optional[str] = None
    language: Optional[str] = Field("english", description="Preferred language for output")


from typing import Optional

# --- Step 1: Processing Sequential Agent (RAG → Role Format) ---
processing_agent = SequentialAgent(
    name="ProcessingAgent",
    sub_agents=[
        # uses vector based RAG agent
        shared_rag_agent,
        LlmAgent(
            name="RoleFormatterAgent",
            model=GEMINI_PRO_MODEL,
            instruction="""
            You will receive structured output from the SharedRagAgent in the format:
            {"subject": "Science", "class_": "Class 10", "content": "NCERT textbook content..."}
            
            Process:
            1. Call SharedRagRoleInspector to get the user's role
            2. Extract the "content" field from the SharedRagAgent's output from the previous step
            3. Call role_formatter_agent with:
               - role: user's role from step 1
               - content: the extracted content field from SharedRagAgent (NOT the entire JSON structure)
               - formatter_type: "answer"
            4. Handle the JSON response from role_formatter_agent:
               - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your answer. Please try asking your question again."
               - If error_logs is empty: Return the formatter_content as the final response
            
            IMPORTANT: Only pass the actual textbook content (from the "content" field) to the role_formatter_agent, not the entire structured response.
            """,
            tools=[shared_rag_role_inspector, role_formatter_agent]
        )
    ],
    description="Handles RAG retrieval and role-based formatting sequentially"
)

# --- Step 2: Main Answer Orchestrator (Clarity Check + Processing) ---
answer_orchestrator_agent = LlmAgent(
    name="AnswerOrchestratorAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
You are a role-aware AI Answer orchestrator assisting students, parents, and teachers with textbook-based answers with clear separation of concerns. Strictly adhere to the following flow without deviation:

    STEP 1: Call ClarifierAgent to determine if the user's query is clear.

    STEP 2: Analyze ClarifierAgent's response:
       - If and ONLY if it contains "needs_clarification": true, immediately return the "follow_up" question to the user. DO NOT proceed further. STOP.
       - If "needs_clarification": false, proceed directly to STEP 3.

    STEP 3: Call ProcessingAgent for handling RAG retrieval and role-based formatting.

    CRITICAL INSTRUCTION (STRICTLY ENFORCED):
    - NEVER, under any circumstance, call ClarifierAgent again after you've received output from ProcessingAgent.
    - Once ProcessingAgent returns a response, consider it FINAL and directly return that response to the user.

    FAILURE TO FOLLOW THIS INSTRUCTION IS UNACCEPTABLE AND WILL BE CONSIDERED A CRITICAL ERROR.
    """,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ],
    input_schema=AnswerOrchestratorInput
)

root_agent = answer_orchestrator_agent


