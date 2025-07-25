import os
from google.cloud import storage
from dotenv import load_dotenv

# 🔐 Load credentials from .env if needed
load_dotenv()

# ⚙️ CONFIGURATION
PROJECT_ID = "rag-engine-vertex-ai-project"
BUCKET_NAME = "shahayak-agentic-ai-gpl-muskeeters"
DESTINATION_FOLDER = "pdf/user-uploads"

# def upload_file_to_gcs(local_file_path: str):
#     """Uploads a local file to the specified GCS bucket and folder."""
#     if not os.path.exists(local_file_path):
#         print("❌ File does not exist:", local_file_path)
#         return None
#
#     try:
#         file_name = os.path.basename(local_file_path)
#         destination_blob_name = f"{DESTINATION_FOLDER}/{file_name}"
#
#         # Initialize GCS client
#         storage_client = storage.Client(project=PROJECT_ID)
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob = bucket.blob(destination_blob_name)
#
#         # Upload
#         blob.upload_from_filename(local_file_path)
#         gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
#
#         print(f"✅ Uploaded successfully to: {gcs_uri}")
#         return gcs_uri
#
#     except Exception as e:
#         print(f"❌ Upload failed due to: {e}")
#         return None

def upload_to_gcs(local_file, gcs_uri):
    """Upload the processed JSONL file and content .txt files to GCS"""
    try:
        # Parse GCS URI for the main JSONL file
        bucket_name = gcs_uri.split("/")[2]
        blob_path_jsonl = "/".join(gcs_uri.split("/")[3:])

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        # 1. Upload the main JSONL file
        blob_jsonl = bucket.blob(blob_path_jsonl)
        blob_jsonl.upload_from_filename(local_file)
        print(f"✅ Uploaded {local_file} to {gcs_uri}")

        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}")
        return False

if __name__ == "__main__":
    local_path = input("📂 Enter the local file path to upload: ").strip()
    if local_path:
        upload_to_gcs(local_path)
    else:
        print("⚠️ No path provided.")
