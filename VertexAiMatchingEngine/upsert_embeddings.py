from google.cloud import aiplatform
from VertexAiMatchingEngine.constants import PROJECT_ID, REGION, INDEX_NAME
import json
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.matching_engine import MatchingEngineIndexEndpoint, MatchingEngineIndex
from google.cloud import storage
from google.cloud.aiplatform.matching_engine import MatchingEngineClient
from VertexAiMatchingEngine.constants import PROJECT_ID, LOCATION, INDEX_ID, BUCKET_NAME
import json
import uuid
import vertexai
import os


def upsert_embedding(index, chapter_id, embedding, grade, subject, chapter):
    index.upsert(
        datapoints=[
            {
                "id": chapter_id,
                "embedding": embedding,
                "restricts": [
                    {"namespace": "grade", "allow_list": [grade]},
                    {"namespace": "subject", "allow_list": [subject]},
                    {"namespace": "chapter", "allow_list": [chapter]},
                ]
            }
        ]
    )
    print(f"✅ Upserted {chapter_id}")

if __name__ == "__main__":
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Load your sample textbook data
    with open("sample_textbook_data.json") as f:
        data = json.load(f)

    # Step 1: Embed the content
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

    embeddings = []
    ids = []
    metadata_list = []

    for item in data:
        text = item["content"]
        metadata = item.get("metadata", {})

        embedding = model.get_embeddings([text])[0].values
        vector_id = str(uuid.uuid4())

        embeddings.append(embedding)
        ids.append(vector_id)
        metadata_list.append(metadata)

    # Step 2: Save embeddings and metadata to a JSONL file
    jsonl_lines = []
    for idx in range(len(ids)):
        jsonl_lines.append(json.dumps({
            "id": ids[idx],
            "embedding": embeddings[idx],
            "metadata": metadata_list[idx]
        }))

    os.makedirs("temp", exist_ok=True)
    jsonl_path = "temp/embeddings.jsonl"
    with open(jsonl_path, "w") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    # Step 3: Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob("embeddings/embeddings.jsonl")
    blob.upload_from_filename(jsonl_path)

    gcs_path = f"gs://{BUCKET_NAME}/embeddings/embeddings.jsonl"
    print("✅ Uploaded embeddings to:", gcs_path)

    # Step 4: Upsert into Matching Engine
    index = MatchingEngineIndex(index_name=INDEX_ID)
    index.upsert_datapoints(gcs_path=gcs_path)
    print("✅ Upserted datapoints to index.")

