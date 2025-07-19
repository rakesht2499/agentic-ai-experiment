import vertexai
from vertexai.preview import rag
from vertexai.preview.generative_models import RagCorpus

# === CONFIGURATION ===
PROJECT_ID = "rag-engine-vertex-ai-project"
REGION = "us-central1"
CORPUS_NAME = "CBSE_Class10_Science"
GCS_JSON_PATH = "gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/science/class10_science.json"

if __name__ == "__main__":
    rag = RagCorpus()
    # === INITIALIZE VERTEX AI ===
    vertexai.init(project=PROJECT_ID, location=REGION)
    print(f"🔧 Vertex AI initialized for project: {PROJECT_ID}, region: {REGION}")

    # === CREATE CORPUS ===
    print(f"📦 Creating or reusing corpus: {CORPUS_NAME}")
    try:
        corpus = RagCorpus.create(display_name=CORPUS_NAME)
        print(f"✅ Created new corpus: {corpus.name}")
    except Exception as e:
        print(f"ℹ️ Corpus may already exist: {e}")
        corpora = rag.list_corpora()
        matching_corpus = next((c for c in corpora if c.display_name == CORPUS_NAME), None)
        if matching_corpus:
            corpus = matching_corpus
            print(f"✅ Reusing existing corpus: {corpus.name}")
        else:
            raise RuntimeError("❌ Failed to create or find corpus.")

    # === INGEST FILE WITH AUTO CHUNKING ===
    print(f"\n📤 Importing file from GCS with automatic chunking...")
    corpus.import_files(
        file_paths=[GCS_JSON_PATH],
        chunk_size=1024,
        chunk_overlap=200,
        metadata={
            "subject": "Science",
            "class": "10",
            "board": "CBSE",
            "source": "CBSE_Textbook"
        }
    )

    print(f"\n🎉 Import completed successfully into corpus: {corpus.name}")
