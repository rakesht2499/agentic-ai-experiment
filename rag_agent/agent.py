import os

import vertexai
from google.adk import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag

from dotenv import load_dotenv
from vertexai.rag.utils.resources import RagRetrievalConfig

from models.constants import GEMINI_PRO_MODEL
from .prompt import instruction_prompt_v1

load_dotenv()


#
# rag_retrieval_config = rag.RagRetrievalConfig(
#     top_k=10,
#     filter=rag.Filter(metadata_filter='subject="Science" AND class="10"'),  # Optional
# )
#
# response = rag.retrieval_query(
#     rag_resources=[
#         rag.RagResource(
#             rag_corpus="CBSE_Class10",
#             # Optional: supply IDs from `rag.list_files()`.
#             # rag_file_ids=["rag-file-1", "rag-file-2", ...],
#         )
#     ],
#     text="What is RAG and why it is helpful?",
#     rag_retrieval_config=rag_retrieval_config,
# )
#



# ask_vertex_retrieval = VertexAiRagRetrieval(
#     name='retrieve_rag_documentation',
#     description=(
#         'Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,'
#     ),
#     rag_resources=[
#         rag.RagResource(
#             rag_corpus="projects/rag-engine-vertex-ai-project/locations/us-central1/ragCorpora/5037276183213899776"
#         )
#     ],
#     similarity_top_k=5,
#     vector_distance_threshold=0.7,
# )
#
# ragAgent = Agent(
#     model=GEMINI_PRO_MODEL,
#     name='ask_rag_agent',
#     instruction=instruction_prompt_v1,
#     tools=[
#         ask_vertex_retrieval,
#     ]
# )
#
# root_agent = ragAgent



from vertexai.preview import rag
from pydantic import BaseModel, Field

class RagQueryInput(BaseModel):
    query: str = Field(..., description="User query for textbook Q&A")
    board: str = Field(..., description="e.g. CBSE")
    subject: str = Field(..., description="e.g. Science")
    class_: str = Field(..., description="e.g. 10")
    chapter: str = Field(None, description="Optional chapter name or number")


from vertexai.preview.rag import RagResource, Filter
from vertexai.preview.rag.rag_retrieval import retrieval_query

def filtered_rag_tool(input: RagQueryInput) -> str:
    rag_resource = RagResource(
        rag_corpus="projects/YOUR_PROJECT_ID/locations/us-central1/ragCorpora/YOUR_CORPUS_ID"
    )

    result = retrieval_query(
        text=input.query,
        rag_resources=[rag_resource],
        similarity_top_k=5,
        vector_distance_threshold=0.7,
        rag_retrieval_config=RagRetrievalConfig(
            filter=Filter(metadata_filter=f'"board="{input.board}" AND "subject={input.subject}" AND chapter="{input.chapter}"'),
        )
    )

    if not result.rag_chunks:
        return "No relevant textbook content found."

    return "\n\n".join([chunk.data.string_value for chunk in result.rag_chunks])


from google.adk.agents import LlmAgent

rag_agent = LlmAgent(
    name="RAGAgent",
    model="gemini-1.5-pro",
    instruction="Retrieve the most relevant textbook answer using the RAG retrieval tool, based on the provided query and metadata.",
    tools=[filtered_rag_tool]
)

