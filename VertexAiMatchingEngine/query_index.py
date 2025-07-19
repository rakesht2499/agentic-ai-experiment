from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from VertexAiMatchingEngine.constants import PROJECT_ID, REGION


def query_index(query_text):
    aiplatform.init(project=PROJECT_ID, location=REGION)
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name="YOUR_INDEX_ENDPOINT")

    query_vector = embedding_model.get_embeddings([query_text])[0].values

    results = endpoint.find_neighbors(
        deployed_index_id=endpoint.deployed_index_id,
        queries=[query_vector],
        num_neighbors=1,
        filter='grade = "4" AND subject = "EVS" AND chapter = "3"',
    )

    if results and results[0]:
        top_result = results[0][0]
        print(f"🔍 Best match: {top_result.id} with score {top_result.distance}")
    else:
        print("❌ No match found.")

if __name__ == "__main__":
    query_index("What are parts of a plant?")
