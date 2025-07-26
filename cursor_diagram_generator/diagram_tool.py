import time
import os
import uuid
from google.cloud import storage
from vertexai.preview.vision_models import ImageGenerationModel
from google.adk.tools import BaseTool
from google.adk.agents import InvocationContext
from google.adk.tools import ToolContext
from pydantic import BaseModel, Field

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func

PROJECT_ID = "rag-engine-vertex-ai-project"
BUCKET = "shahayak-agentic-ai-gpl-muskeeters"
PREFIX = "image_generation"

class DiagramInput(BaseModel):
    prompt: str = Field(description="The prompt for diagram generation")

class RealDiagramGeneratorTool(BaseTool):
    """Proper ADK tool for generating diagrams and uploading to GCS"""

    def __init__(self):
        super().__init__(
            name="generate_diagram_real",
            description="Generates a diagram from a prompt and uploads to GCS. Returns local path and GCS URI."
        )

    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        """Generate diagram and upload to GCS"""
        print("🚀🚀🚀 REAL DIAGRAM TOOL CALLED! 🚀🚀🚀")
        
        # Get prompt from context
        prompt = ""
        if hasattr(context, 'input') and context.input:
            if isinstance(context.input, dict):
                prompt = context.input.get('prompt', '') or str(context.input)
            elif hasattr(context.input, 'prompt'):
                prompt = context.input.prompt
            else:
                prompt = str(context.input)
        
        if not prompt:
            return "Error: No prompt provided for diagram generation."
            
        print(f"🎯 Generating diagram for prompt: '{prompt}'")
        
        # 1. Generate filename
        timestamp = int(time.time())
        uuid_str = uuid.uuid4().hex[:8]
        filename = f"diagram_{timestamp}_{uuid_str}.png"
        
        # Ensure directory exists
        os.makedirs("./generated_diagrams", exist_ok=True)
        local_path = f"./generated_diagrams/{filename}"
        
        try:
            # 2. Generate image using Vertex AI
            print("🎨 Generating image with imagen-3.0-generate-002...")
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            response = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="1:1"
            )
            
            # Save locally
            response.images[0].save(local_path)
            print(f"💾 Image saved locally: {local_path}")
            
            # 3. Upload to GCS
            print("☁️ Uploading to GCS...")
            client = storage.Client(project=PROJECT_ID)
            bucket = client.bucket(BUCKET)
            blob_path = f"{PREFIX}/{filename}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path, content_type="image/png")
            
            gcs_uri = f"gs://{BUCKET}/{blob_path}"
            public_url = f"https://storage.googleapis.com/{BUCKET}/{blob_path}"
            
            print(f"✅ Uploaded to GCS: {gcs_uri}")
            
            # Return structured response
            return f"""Diagram file: {local_path}
GCS URI: {gcs_uri}
Public URL: {public_url}
Caption: Educational diagram generated for: {prompt}"""
            
        except Exception as e:
            error_msg = f"❌ Error generating diagram: {e}"
            print(error_msg)
            return error_msg

# Create the tool instance
real_diagram_tool = RealDiagramGeneratorTool()
print(f"🔧 ✅ RealDiagramGeneratorTool created: {real_diagram_tool.name}")
print(f"🔧 ✅ Tool description: {real_diagram_tool.description}")

def generate_diagram(prompt: str):
    """Simple diagram generation with direct GCS upload (for direct function calls)"""
    print(f"🚀 REAL TOOL CALLED: generate_diagram('{prompt}')")
    
    # 1. Generate filename
    timestamp = int(time.time())
    uuid_str = uuid.uuid4().hex[:8]
    filename = f"diagram_{timestamp}_{uuid_str}.png"
    
    # Ensure directory exists
    os.makedirs("./generated_diagrams", exist_ok=True)
    local_path = f"./generated_diagrams/{filename}"
    
    try:
        # 2. Generate image using Vertex AI
        print("Generating image with imagen-3.0-generate-002...")
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        
        # Save locally
        response.images[0].save(local_path)
        print(f"✅ Image saved locally: {local_path}")
        
        # 3. Upload to GCS
        print("Uploading to GCS...")
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET)
        blob_path = f"{PREFIX}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, content_type="image/png")
        
        gcs_uri = f"gs://{BUCKET}/{blob_path}"
        public_url = f"https://storage.googleapis.com/{BUCKET}/{blob_path}"
        
        print(f"✅ Uploaded to GCS: {gcs_uri}")
        
        return {
            "local_path": local_path,
            "gcs_uri": gcs_uri,
            "public_url": public_url,
            "caption": f"Educational diagram generated for: {prompt}"
        }
        
    except Exception as e:
        print(f"❌ Error generating diagram: {e}")
        return {
            "local_path": local_path if os.path.exists(local_path) else None,
            "gcs_uri": None,
            "public_url": None,
            "caption": f"Error: {e}"
        } 