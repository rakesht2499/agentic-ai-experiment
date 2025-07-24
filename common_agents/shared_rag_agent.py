"""
Shared RAG Agent for Sahayak 2.0 - ADK Compatible

This module provides a centralized RAG agent that can be reused across all orchestrators
(Answer, Quiz, LessonPlanner, etc.) with standardized input/output schemas and role management.
"""

import os
from typing import Optional, Literal

from nltk.sentiment.util import output_markdown
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext

from vertexai.rag.utils.resources import RagRetrievalConfig
from vertexai.preview.rag import RagResource, Filter
from vertexai.preview.rag.rag_retrieval import retrieval_query

from models.constants import GEMINI_PRO_MODEL

load_dotenv()


class SharedRagInput(BaseModel):
    """Standardized input schema for the shared RAG agent"""
    query: str = Field(..., description="User query for textbook content retrieval")
    subject: str = Field(..., description="Academic subject (e.g., Science, Math)")
    class_: str = Field(..., description="Grade level (e.g., Class 10, 8)")
    chapter: Optional[str] = Field(None, description="Optional specific chapter name or number")
    board: Optional[str] = Field("CBSE", description="Educational board (default: CBSE)")


class SharedRagOutput(BaseModel):
    """Standardized output schema for the shared RAG agent"""
    subject: str = Field(..., description="The subject for which content was retrieved")
    class_: str = Field(..., description="The class/grade level")
    content: str = Field(..., description="Retrieved textbook content or fallback message")


def shared_rag_postprocess_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """
    Shared postprocess callback for role management and RAG failure detection.
    Sets role to 'student' by default - this will be overridden by orchestrator input.
    """
    callback_context.state["role"] = "teacher"  # Default, will be overridden by orchestrator
    
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            print(f"[Shared RAG] Inspected response: '{original_text[:100]}...'")
            
            # Detect RAG retrieval failure
            if "RAG_RETRIEVAL_FAILED" in original_text or "No relevant textbook content found" in original_text:
                callback_context.state["rag_failed"] = True
            else:
                callback_context.state["rag_failed"] = False
            return llm_response
        elif llm_response.content.parts[0].function_call:
            print(f"[Shared RAG] Function call detected: '{llm_response.content.parts[0].function_call.name}'")
            return None
        else:
            print("[Shared RAG] No text content found")
            return None
    elif llm_response.error_message:
        print(f"[Shared RAG] Error detected: '{llm_response.error_message}'")
        callback_context.state["rag_failed"] = True
        return None
    else:
        print("[Shared RAG] Empty response")
        return None


def filtered_rag_retrieval_tool(input_data: SharedRagInput) -> SharedRagOutput:
    """
    Core RAG retrieval tool that fetches content from Vertex RAG corpus
    """
    try:
        rag_resource = RagResource(
            rag_corpus=f'projects/{os.getenv("GOOGLE_CLOUD_PROJECT")}/locations/us-central1/ragCorpora/{os.getenv("RAG_CORPORA_ID")}'
        )

        # Build metadata filter
        metadata_filter = f'"board"="{input_data.board}" AND "subject"="{input_data.subject}"'
        if input_data.chapter:
            metadata_filter += f' AND "chapter"="{input_data.chapter}"'

        result = retrieval_query(
            text=input_data.query,
            rag_resources=[rag_resource],
            similarity_top_k=5,
            vector_distance_threshold=0.7,
            rag_retrieval_config=RagRetrievalConfig(
                filter=Filter(metadata_filter=metadata_filter),
            )
        )

        if not result.rag_chunks:
            content = "RAG_RETRIEVAL_FAILED"
        else:
            content = "\n\n".join([chunk.data.string_value for chunk in result.rag_chunks])

        return SharedRagOutput(
            subject=input_data.subject,
            class_=input_data.class_,
            content=content
        )

    except Exception as e:
        print(f"[Shared RAG] Error during retrieval: {e}")
        return SharedRagOutput(
            subject=input_data.subject,
            class_=input_data.class_,
            content="RAG_RETRIEVAL_FAILED"
        )


class SharedRagInspectorTool(BaseTool):
    """Tool to inspect current user role from session state"""
    
    async def run_async(self, context, tool_context: ToolContext) -> str:
        return context.session.state.get("role", "unknown")


# Create shared RAG agent instance
shared_rag_agent = LlmAgent(
    name="SharedRagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    You are a shared RAG agent for Sahayak 2.0 educational assistant.
    
    Your job is to retrieve relevant NCERT textbook content based on user queries and educational metadata.
    
    Process:
    1. Receive a query with subject, class, and optional chapter information
    2. Use filtered_rag_retrieval_tool to fetch relevant content from the textbook corpus
    3. Return standardized output with subject, class, and content fields
    
    IMPORTANT:
    - Always return the standardized output schema: {"subject", "class_", "content"}
    - If no content is found, content should contain "RAG_RETRIEVAL_FAILED"
    - Never invent or hallucinate textbook content
    - Log the query and role information for tracing
    """,
    # tools=[filtered_rag_retrieval_tool],
    after_model_callback=shared_rag_postprocess_callback,
    input_schema=SharedRagInput,
    output_schema=SharedRagOutput
)

# rag_agent = LlmAgent(
#     name="RagAgent",
#     model=GEMINI_PRO_MODEL,
#     instruction="""
#     You need to receive
#     """,
#     output_schema=SharedRagOutput,
# )

# Create role inspector tool instance
shared_rag_role_inspector = SharedRagInspectorTool(
    name="SharedRagRoleInspector", 
    description="Inspects the current user role from the agent state context and returns it."
) 