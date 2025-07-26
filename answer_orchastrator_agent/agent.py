from typing import Literal
from typing import Optional

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import agent_tool, BaseTool, ToolContext, google_search
from pydantic import BaseModel, Field

from common_agents import role_formatter_agent
from common_agents.shared_rag_agent import shared_rag_agent, shared_rag_role_inspector, clone_agent, vector_rag_agent
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

class ClarifierInput(BaseModel):
    query: str = Field(..., description="User input that may be vague or incomplete.")
    role: Literal["teacher", "parent", "student"] = Field(..., description="User's role to help shape clarifying response.")

clarifier_agent = LlmAgent(
    name="ClarifierAgent",
    model=GEMINI_FLASH_MODEL,
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
    model=GEMINI_FLASH_MODEL,
    instruction="""
    You are a responsible education-grade translator.
    
    Translate the given content into the target language with:
    
    - Accurate concept preservation
    - Simplified local language (avoid over-academic tone)
    - Cultural adaptation where suitable
    
    Respond with the translated content directly.
    """
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
    board: Optional[str] = Field("CBSE", description="The board that this question belongs to.")
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
        clone_agent(vector_rag_agent, "answer"),
        LlmAgent(
            name="RoleFormatterAgent",
            model=GEMINI_FLASH_MODEL,
            instruction="""
            You will receive output from the SharedRagAgent_answer from the previous step.
            
            Process:
            1. Call SharedRagRoleInspector to get the user's role
            2. Parse the JSON response from SharedRagAgent_answer to extract the content:
               - The response should be in format: {"subject": "...", "class_": "...", "content": "..."}
               - Extract ONLY the "content" field value 
               - If the response is not valid JSON, use the entire response as content
            3. Call role_formatter_agent with:
               - role: user's role from step 1
               - content: the extracted content field from step 2 (NOT the entire JSON structure)
               - formatter_type: "answer"
            4. Handle the JSON response from role_formatter_agent:
               - If error_logs is NOT empty: Return "I apologize, there was an issue formatting your answer. Please try asking your question again."
               - If error_logs is empty: Return ONLY the formatter_content as the final response
                - 🔍 Then use GoogleSearchTool to search for a relevant video explanation of the query in the user's preferred language. Append the search result & mention it was fetched from google search.
               - if no class is mentioned then find out from NCERT books where this concept is mentioned and for that respective class search for the link to get better results.
               - Display the query which you are using to search in youtube.
               - Don't give more than 2 videos.
               - Only add those links which are available on youtube and does not show "This video isn't available any more"
               - Return the YouTube video link with a short message like:
                    `🎥 Here's a video explanation I found for you: [video_title] — [video_url] (via Google Search)`
            
            CRITICAL: 
            - Return ONLY the final formatted content from role_formatter_agent
            - Do NOT include any intermediate outputs, JSON structures, or raw textbook content
            - The output should be the engaging, role-specific response (e.g., "Hey there, future scientist!")
            - Remove any duplicate or unformatted content
            - If content is "RAG_RETRIEVAL_FAILED", return "I couldn't find relevant information about your query in the textbook. Please try asking a more specific question."
            """,
            tools=[shared_rag_role_inspector, role_formatter_agent, google_search]
        )
    ],
    description="Handles RAG retrieval and role-based formatting sequentially"
)

# --- Step 2: Main Answer Orchestrator (Clarity Check + Processing) ---
answer_orchestrator_agent = LlmAgent(
    name="AnswerOrchestratorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
You are a role-aware AI Answer orchestrator assisting students, parents, and teachers with textbook-based answers with clear separation of concerns. Strictly adhere to the following flow without deviation:

    STEP 1: Call ClarifierAgent to determine if the user's query is clear.

    STEP 2: Analyze ClarifierAgent's response:
       - If and ONLY if it contains "needs_clarification": true, immediately return the "follow_up" question to the user. DO NOT proceed further. STOP.
       - If "needs_clarification": false, proceed directly to STEP 3.

    STEP 3: Call ProcessingAgent for handling RAG retrieval and role-based formatting.

    STEP 4: Extract and return ONLY the final formatted response from ProcessingAgent:
       - The ProcessingAgent returns output from multiple sub-agents
       - Return ONLY the final formatted content (the role-specific formatted answer)
       - DO NOT include any intermediate RAG outputs or JSON structures
       - The final output should be the engaging, role-appropriate response that starts with greetings like "Hey there, future scientist!"

    CRITICAL INSTRUCTIONS (STRICTLY ENFORCED):
    - NEVER call ClarifierAgent again after you've received output from ProcessingAgent
    - Return ONLY the final formatted answer, not intermediate outputs
    - The response should be clean, engaging, and directly useful to the user
    - Remove any duplicate or intermediate content

    FAILURE TO FOLLOW THIS INSTRUCTION IS UNACCEPTABLE AND WILL BE CONSIDERED A CRITICAL ERROR.
    """,
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ],
    input_schema=AnswerOrchestratorInput
)

root_agent = answer_orchestrator_agent


