import re
import vertexai
from vertexai import rag
from vertexai.preview.rag import (
    RagCorpus,
    list_corpora,
    import_files,
    TransformationConfig,
    ChunkingConfig
)

# === CONFIGURATION ===
PROJECT_ID = "rag-engine-vertex-ai-project"
REGION = "us-central1"
CORPUS_NAME = "CBSE_Class10_Science"
GCS_JSON_PATH = "gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/science/class10_science.json"

if __name__ == "__main__":
    # === INITIALIZE VERTEX AI ===
    vertexai.init(project=PROJECT_ID, location=REGION)
    print(f"🔧 Vertex AI initialized for project: {PROJECT_ID}, region: {REGION}")

    # === CREATE CORPUS ===
    print(f"📦 Creating or reusing corpus: {CORPUS_NAME}")
    corpus = None
    # try:
    #     # Clean corpus name for use as display name
    #     display_name = re.sub(r"[^a-zA-Z0-9_-]", "_", CORPUS_NAME)
    #
    #     # Configure embedding model - using text-embedding-004 which is supported for RAG
    #     embedding_model_config = rag.RagEmbeddingModelConfig(
    #         vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
    #             publisher_model="publishers/google/models/text-embedding-004"
    #         )
    #     )
    #
    #     # Create the corpus
    #     corpus = rag.create_corpus(
    #         display_name=display_name,
    #         backend_config=rag.RagVectorDbConfig(
    #             rag_embedding_model_config=embedding_model_config
    #         ),
    #     )
    #     print(f"✅ Created new corpus: {corpus.name}")
    # except Exception as e:
    # print(f"ℹ️ Corpus may already exist: {e}")
    corpora = list_corpora()
    for c in corpora:
        print(f"Corpus Display Name: {c.display_name}")
    matching_corpus = next((c for c in corpora if c.display_name == CORPUS_NAME), None)
    if matching_corpus:
        corpus = matching_corpus
        print(f"✅ Reusing existing corpus: {corpus.name}")
    else:
        raise RuntimeError("❌ Failed to create or find corpus.")

    # === CONFIGURE CHUNKING ===
    print(f"\n⚙️ Configuring chunking parameters...")
    transformation_config = TransformationConfig(
        chunking_config=ChunkingConfig(
            chunk_size=2000,  # Optimal size based on your analysis
            chunk_overlap=100  # Small overlap to maintain context
        )
    )

    # === INGEST FILE WITH STRUCTURED CHUNKING ===
    print(f"\n📤 Importing file from GCS with structured chunking...")
    result = import_files(
        corpus_name="projects/rag-engine-vertex-ai-project/locations/us-central1/ragCorpora/4172585054758764544",
        paths=[GCS_JSON_PATH],
        transformation_config=transformation_config,
        # max_embedding_requests_per_min=1000,  # Rate limiting
    )

    if result:
        print(f"🎉 Import completed successfully into corpus: {corpus.name}")
    else:
        print("❌ Import failed!")

    print(f"📊 Chunking config: 2000 chars per chunk, 100 char overlap")