from google.cloud import aiplatform
from old_and_only_for_reference.VertexAiMatchingEngine.constants import PROJECT_ID, REGION, INDEX_NAME, DIMENSIONS

def create_index():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_NAME,
        dimensions=DIMENSIONS,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        shard_size="SHARD_SIZE_SMALL",
        index_update_method="STREAM_UPDATE",
    )

    print("✅ Index created:", index.resource_name)
    return index

if __name__ == "__main__":
    create_index()
