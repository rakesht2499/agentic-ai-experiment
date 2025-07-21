import os

import vertexai
from google.adk import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag

from dotenv import load_dotenv

from models.constants import GEMINI_PRO_MODEL
from .prompt import instruction_prompt_v1

load_dotenv()

vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")

embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-004"
    )
)

vertexai.init(project=PROJECT_ID, location="us-central1")

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_name,
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
                filter=rag.utils.resources.Filter(vector_distance_threshold=0.5),
            ),
        ),
    )
)

rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool]
)
response = rag_model.generate_content("Why is the sky blue?")
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

ragAgent = Agent(
    model=GEMINI_PRO_MODEL,
    name='ask_rag_agent',
    instruction=instruction_prompt_v1,
    tools=[
        ask_vertex_retrieval,
    ]
)

root_agent = ragAgent