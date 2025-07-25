import os
from typing import Optional
from pydantic import BaseModel, Field
from google.cloud import storage
from google.adk.tools import BaseTool, ToolContext
from google.adk.agents import InvocationContext

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

class GcsUploadInput(BaseModel):
    local_file: str = Field(..., description="Path to the local file to upload")
    gcs_uri: str = Field(..., description="Full GCS URI where the file should be uploaded (e.g., gs://bucket/path/file.jsonl)")
    content_type: Optional[str] = Field(None, description="Optional MIME type")

class GcsUploaderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="GcsUploaderTool",
            description="Uploads a local file to Google Cloud Storage given a GCS URI."
        )
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        input_data = GcsUploadInput(**context.input.dict())

        if not PROJECT_ID:
            return "❌ Environment variable `GOOGLE_CLOUD_PROJECT` is not set."

        if not input_data.gcs_uri.startswith("gs://"):
            return f"❌ Invalid GCS URI: {input_data.gcs_uri}. Must start with `gs://`"

        try:
            bucket_name = input_data.gcs_uri.split("/")[2]
            blob_path = "/".join(input_data.gcs_uri.split("/")[3:])

            storage_client = storage.Client(project=PROJECT_ID)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(input_data.local_file)

            return f"✅ File `{input_data.local_file}` uploaded to `{input_data.gcs_uri}`"
        except Exception as e:
            return f"❌ Upload failed: {str(e)}"
