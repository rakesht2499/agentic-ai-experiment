"""
Shared RAG Agent for Sahayak 2.0 - ADK Compatible

This module provides a centralized RAG agent that can be reused across all orchestrators
(Answer, Quiz, LessonPlanner, etc.) with standardized input/output schemas and role management.
"""

import os
from typing import Optional, List

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.cloud.aiplatform_v1 import RetrieveContextsResponse
from pydantic import BaseModel, Field
from vertexai.preview.rag import RagResource
from vertexai.preview.rag.rag_retrieval import retrieval_query

from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

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
    
    print(f"Shared RAG postprocessing callback, {llm_response}")
    
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

def process_rag_contexts(result: RetrieveContextsResponse) -> List[str]:
    retrieved_chunks = []
    for context in result.contexts.contexts:
        # Access the text from the context object
        if hasattr(context, 'text') and context.text:
            text = context.text
            # Extract just the content part from the structured text
            if 'content ' in text:
                # Extract content between 'content ' and 'metadata'
                content_start = text.find('content ') + len('content ')
                content_end = text.find('\nmetadata')
                if content_end != -1:
                    chunk_content = text[content_start:content_end].strip()
                else:
                    chunk_content = text[content_start:].strip()
                retrieved_chunks.append(chunk_content)
            else:
                # Fallback: use the entire text if format is unexpected
                retrieved_chunks.append(text)
    return retrieved_chunks

def remove_duplicates_from_chunks(retrieved_chunks: List[str]) -> List[str]:
    if retrieved_chunks:
        # Light deduplication: remove exact duplicates while preserving order
        deduplicated_chunks = []
        seen_chunks = set()

        # The overlap check is now smarter: we only declare overlap if 80%+ of smaller exists in the larger.
        for chunk in retrieved_chunks:
            if chunk not in seen_chunks:
                # Check for significant overlap (e.g. > 80% match) with previously accepted chunks
                is_significant_overlap = False
                for existing_chunk in deduplicated_chunks:
                    smaller_chunk = chunk if len(chunk) < len(existing_chunk) else existing_chunk
                    larger_chunk = existing_chunk if len(chunk) < len(existing_chunk) else chunk

                    # If 80% or more of the smaller chunk is inside the larger
                    if len(smaller_chunk) / len(larger_chunk) >= 0.8 and smaller_chunk in larger_chunk:
                        is_significant_overlap = True
                        break

                if not is_significant_overlap:
                    deduplicated_chunks.append(chunk)
                    seen_chunks.add(chunk)

        return deduplicated_chunks
    return []

def filtered_rag_retrieval_tool(input_data: SharedRagInput) -> str:
    """
    Core RAG retrieval tool that fetches content from Vertex RAG corpus
    Returns either joined chunks or "RAG_RETRIEVAL_FAILED"
    """
    print(f"RAG Input Data: {input_data}")
    
    # Handle both dict and SharedRagInput object formats
    if isinstance(input_data, dict):
        # Convert dict to SharedRagInput object with defaults
        query = input_data.get('query', '')
        subject = input_data.get('subject', '')
        class_ = input_data.get('class_', '')
        chapter = input_data.get('chapter', None)
        board = input_data.get('board', 'CBSE')  # Default to CBSE if not provided
    else:
        # Already a SharedRagInput object
        query = input_data.query
        subject = input_data.subject
        class_ = input_data.class_
        chapter = input_data.chapter
        board = input_data.board or 'CBSE'  # Default to CBSE if None

    print(f"RAG Retrieval Tool Input: {query}, {subject}, {class_}, {chapter}, {board}")
    
    # Direct retrieval
    try:
        rag_resource = RagResource(
            rag_corpus=f'projects/{os.getenv("GOOGLE_CLOUD_PROJECT")}/locations/us-central1/ragCorpora/{os.getenv("RAG_CORPORA_ID")}'
        )

        # Build metadata filter with proper escaping
        # metadata_filter = f'"board"="{board}" AND "subject"="{subject}"'
        # if chapter:
        #     metadata_filter += f' AND "chapter"="{chapter}"'

        result = retrieval_query(
            text=query,
            rag_resources=[rag_resource],
            similarity_top_k=5,
            vector_distance_threshold=0.5,
        )

        print(f"[DIRECT] RAG Retrieval Tool Result: {type(result)}")
        print(f"RAG Retrieval Tool Result Contexts: {result.contexts}")

        # Handle RetrieveContextsResponse object properly
        if not result.contexts or len(result.contexts.contexts) == 0:
            return "RAG_RETRIEVAL_FAILED"
        else:
            # Extract content from RagContexts.contexts list
            retrieved_chunks = process_rag_contexts(result)
            deduplicated_chunks = remove_duplicates_from_chunks(retrieved_chunks)

            if deduplicated_chunks:
                # Join chunks with special separator for parsing later
                return "CHUNKS_FOUND:" + "\n\n---CHUNK---\n\n".join(deduplicated_chunks)
            else:
                return "RAG_RETRIEVAL_FAILED"

    except Exception as e:
        print(f"[Shared RAG] Error during retrieval: {e}")
        return "RAG_RETRIEVAL_FAILED"

class ChunkFilterInput(BaseModel):
    query: str = Field(..., description="The user's original query or topic")
    chunks: List[str] = Field(..., description="List of candidate RAG chunks from vector DB")

class ChunkFilterOutput(BaseModel):
    filtered_chunks: List[str] = Field(..., description="Filtered list of chunks relevant to the query")

rag_chunk_filter_agent = LlmAgent(
    name="RagChunkFilterAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    🎯 Your job is to act as a chunk filter agent in the Sahayak 2.0 system.

    Given:
    - A **user query**
    - A **list of textbook chunks** retrieved via RAG

    You MUST:
    1. Read the user's query carefully.
    2. Evaluate all chunks provided.
    3. Select and return ONLY those chunks that are **highly relevant** to the query.
    4. If no chunks are relevant, return an empty list.

    ⚠️ IMPORTANT:
    - Do NOT modify or summarize chunk content.
    - Do NOT hallucinate or generate new information.
    - You are only filtering, not formatting or answering.

    🛠️ Return only this JSON format:
    ```json
    {
      "filtered_chunks": [ ...only the relevant chunks as strings... ]
    }
    """,
    input_schema=ChunkFilterInput,
)

class SharedRagInspectorTool(BaseTool):
    """Tool to inspect current user role from session state"""
    
    async def run_async(self, context, tool_context: ToolContext) -> str:
        return context.session.state.get("role", "unknown")


# Create a wrapper function to call the rag_chunk_filter_agent
def chunk_filter_tool(input_data: ChunkFilterInput) -> ChunkFilterOutput:
    """
    Wrapper tool that calls the rag_chunk_filter_agent internally
    """
    try:
        # Call the rag_chunk_filter_agent
        result = rag_chunk_filter_agent.run(input_data)
        return result
    except Exception as e:
        print(f"[Chunk Filter] Error during filtering: {e}")
        # Return empty filtered chunks on error
        return ChunkFilterOutput(filtered_chunks=[])


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
    after_model_callback=shared_rag_postprocess_callback,
    input_schema=SharedRagInput,
    output_schema=SharedRagOutput
)

vector_rag_agent = LlmAgent(
    name="SharedRagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
🎓 You are the `SharedRagAgent`, a retrieval-focused LLM agent in the Sahayak 2.0 AI system.

    Your job is to retrieve **accurate NCERT textbook content** from a class- and subject-specific vector database and return it as raw content.

--------------------
    🚧 ALWAYS FOLLOW THIS EXACT SEQUENCE:
    
    1. ✅ **First**, call the `filtered_rag_retrieval_tool` using the following fields:
   - `query` (from user input)
   - `subject`
   - `class_`
   - `chapter` (optional)

       This will return either:
       - String starting with "CHUNKS_FOUND:": Contains textbook chunks separated by "---CHUNK---"
       - String "RAG_RETRIEVAL_FAILED": No relevant content found

    2. 🧹 **Then**, construct and return the final response using this EXACT JSON format:

```json
{
  "subject": "<subject_from_input>",
  "class_": "<class_from_input>",
      "content": "<raw_textbook_content>"
}
    ```
    
    ⚠️ IMPORTANT RULES:
    
    ❌ DO NOT generate or invent any content on your own.
    ❌ DO NOT add any greetings, role-specific formatting, or conversational elements.
    ❌ DO NOT add phrases like "To understand..." or "Let's break it down..."
    
    ✅ Use only what you receive from the filtered_rag_retrieval_tool.
    ✅ Return PURE textbook content without any additional formatting or explanations.
    
    🔁 Call filtered_rag_retrieval_tool exactly once per request.
    
    🧼 If you receive "CHUNKS_FOUND:" response, extract the chunks after the colon, split by "---CHUNK---", and join them with double line breaks to create clean textbook content.
    
    🤖 If you receive "RAG_RETRIEVAL_FAILED", return it as the content value.
    
    ✅ ALWAYS return VALID JSON in the exact format specified above. The response must be parseable JSON.
    
    Examples of content field (RAW textbook content only):
    
    ✔️ "Photosynthesis is the process by which green plants make their own food using sunlight, carbon dioxide, and water..."
    ✔️ Direct textbook explanations without conversational elements
    ✔️ Scientific facts and processes as written in NCERT
    
    ❌ NOT: "To understand photosynthesis, let's break it down..."
    ❌ NOT: "Hey there, future scientist..."
    ❌ NOT: Any role-specific or conversational formatting
    """,
    tools=[
        filtered_rag_retrieval_tool
    ],
    after_model_callback=shared_rag_postprocess_callback,
    input_schema=SharedRagInput
)

from copy import deepcopy

def clone_agent(agent: LlmAgent, name_suffix: str) -> LlmAgent:
    return LlmAgent(
        name=f"SharedRagAgent_{name_suffix}",
        model=agent.model,
        instruction=agent.instruction,
        tools=deepcopy(agent.tools),
        input_schema=agent.input_schema,
        after_model_callback=agent.after_model_callback
    )

# Create role inspector tool instance
shared_rag_role_inspector = SharedRagInspectorTool(
    name="SharedRagRoleInspector", 
    description="Inspects the current user role from the agent state context and returns it."
) 