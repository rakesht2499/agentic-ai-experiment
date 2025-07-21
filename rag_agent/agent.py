import os

from dotenv import load_dotenv
from vertexai.rag.utils.resources import RagRetrievalConfig

load_dotenv()

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
        rag_corpus=f'projects/{os.getenv("GOOGLE_CLOUD_PROJECT")}/locations/us-central1/ragCorpora/{os.getenv("RAG_CORPORA_ID")}'
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

