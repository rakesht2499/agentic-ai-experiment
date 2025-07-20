from google.adk import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from dotenv import load_dotenv
from .prompt import instruction_prompt_v1

load_dotenv()

ask_vertex_retrieval = VertexAiRagRetrieval(
    name='retrieve_rag_documentation',
    description=(
        'Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,'
    ),
    rag_resources=[
        rag.RagResource(
            rag_corpus="projects/rag-engine-vertex-ai-project/locations/us-central1/ragCorpora/5037276183213899776"
        )
    ],
    similarity_top_k=5,
    vector_distance_threshold=0.7,
)

ragAgent = Agent(
    model='gemini-2.5-flash',
    name='ask_rag_agent',
    instruction=instruction_prompt_v1,
    tools=[
        ask_vertex_retrieval,
    ]
)

root_agent = ragAgent