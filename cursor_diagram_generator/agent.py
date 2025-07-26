import uuid
import os
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext
from pydantic import BaseModel, Field
from google.cloud import storage

from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

from cursor_diagram_generator.prompts import prompt_for_refiner_agent, prompt_for_validator_agent, prompt_for_flowchart_agent, prompt_for_diagram_generating_agent
from models.constants import GEMINI_FLASH_MODEL

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func

# === GCS CONFIGURATION ===
PROJECT_ID = "rag-engine-vertex-ai-project"
GCS_BUCKET = "shahayak-agentic-ai-gpl-muskeeters"
GCS_IMAGE_PREFIX = "image_generation"

def upload_to_gcs(local_path: str, bucket_name: str = GCS_BUCKET, prefix: str = GCS_IMAGE_PREFIX):
    """Upload a local file to Google Cloud Storage with proper error handling"""
    import time
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        fname = os.path.basename(local_path)
        # Make it unique & nested nicely
        object_name = f"{prefix}/{int(time.time())}_{uuid.uuid4().hex}_{fname}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path, content_type="image/png")

        gcs_uri = f"gs://{bucket_name}/{object_name}"
        print(f"✅ Uploaded to GCS: {gcs_uri}")
        return gcs_uri, blob.public_url if hasattr(blob, 'public_url') else None
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        return None, None


# --- Step 1: Flow Extraction --- #
def convert_prompt_to_flow(prompt: str) -> str:
    """
    Uses Gemini to convert a generic natural language prompt into a structured diagram flow.
    """
    if prompt.count("->") >= 1 and all(" " not in step.strip() for step in prompt.split("->")):
        # Already a clean flow like A->B->C
        return prompt.strip()

    print("--- TOOL: Calling LLM to interpret freeform prompt ---")
    system_instruction = f"""
You are a helpful assistant that converts detailed natural language descriptions of processes, lifecycles, or workflows into clear, structured flowcharts.

Given a user prompt like:

"{prompt}"

Respond only with a step-by-step sequence in the format:

Step1 -> Step2 -> Step3 -> ... -> FinalStep

The output must include all intermediate stages and reflect logical progression.
Do not explain or add context — just return the direct linear or branching flow.

Your job is to ensure the response is suitable for generating a diagram using Graphviz or similar tools.


"""
    try:
        model = GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(system_instruction)
        print(f"LLM Response: {response.text}")
        return response.text.strip()
    except Exception as e:
        print(f"Failed to call Gemini: {e}")
        return "Prompt -> Processing -> Output"

# --- Old DiagramGeneratorTool removed - using new RealDiagramGeneratorTool below --- #


# --- Output Schemas --- #
class ValidatorOutput(BaseModel):
    is_clear: bool = Field(description="Indicates whether the prompt is clear enough.")
    feedback: str = Field(description="Feedback or suggestions to improve the prompt.")

class ReviewerOutput(BaseModel):
    approved: bool = Field(description="Whether the prompt is good enough to proceed.")
    feedback: str = Field(description="Specific comments for improvement.")

class RefinerOutput(BaseModel):
    refined_prompt: str = Field(description="The improved version of the user's prompt.")

# --- Tool Return Model --- #
class DiagramFinalAnswer(BaseModel):
    diagram_result: str = Field(description="File path or URL of the generated diagram.")
    caption: str = Field(description="Descriptive caption for the generated diagram.")


class DiagramOrchestratorOutput(BaseModel):
    final_prompt: str = Field(description="The final prompt used.")
    diagram_file_path: str = Field(description="Path to the generated diagram.")
    final_caption: str = Field(description="Caption for the diagram.")

# --- Agents --- #
prompt_validator_agent = LlmAgent(
    name="prompt_validator_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=prompt_for_validator_agent,
    output_schema=ValidatorOutput
)

reviewer_agent = LlmAgent(
    name="reviewer_agent",
    model=GEMINI_FLASH_MODEL,
    instruction="Review the refined diagram prompt. If it's detailed and clear, set 'approved' to True. Otherwise, give concise feedback and set 'approved' to False.",
    output_schema=ReviewerOutput
)

prompt_refiner_agent = LlmAgent(
    name="prompt_refiner_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=prompt_for_refiner_agent,
    output_schema=RefinerOutput
)

# Use a simple tool pattern that definitely works
class DiagramGeneratorTool(BaseTool):
    name = "generate_diagram_real"
    description = "Generates a diagram from a prompt and uploads to GCS. Returns local path and GCS URI."

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
        
        # Import here to avoid circular imports
        import time
        import uuid
        from google.cloud import storage
        from vertexai.preview.vision_models import ImageGenerationModel
        
        # GCS Configuration
        PROJECT_ID = "rag-engine-vertex-ai-project"
        BUCKET = "shahayak-agentic-ai-gpl-muskeeters" 
        PREFIX = "image_generation"
        
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

# Create the tool instance using the working pattern (no constructor parameters needed)
diagram_generator_tool = DiagramGeneratorTool(name="generate_diagram_real", description="Generates a diagram from a prompt and uploads to GCS")
print(f"🔧 ✅ DiagramGeneratorTool created: {diagram_generator_tool.name}")
print(f"🔧 ✅ Tool description: {diagram_generator_tool.description}")

diagram_generation_agent = LlmAgent(
    name="diagram_generation_agent",
    model=GEMINI_FLASH_MODEL,
    description="Generates a diagram and uploads to GCS using proper ADK tool.",
    instruction="""
You have access to a tool called "generate_diagram_real".

Call this tool with the user's prompt to generate a diagram and upload it to GCS.

IMPORTANT: 
- Actually call the tool, don't make up responses
- The tool will return the local path and GCS URI
- Return exactly what the tool provides

Look for "🚀🚀🚀 REAL DIAGRAM TOOL CALLED!" in the logs to confirm the tool was called.
""",
    tools=[diagram_generator_tool],
)

diagram_generating_agent = LlmAgent(
    name="diagram_generating_agent",
    model=GEMINI_FLASH_MODEL,
    description="Simple diagram generation without complex orchestration.",
    instruction="""
Call the diagram_generation_agent tool.

Do NOT simulate or print tool calls. Do NOT output code like "generate_diagram(...)".

Call the tool and return its output exactly as provided.

Expected output format:
Diagram file: /path/to/file.png
GCS URI: gs://bucket/path/file.png
Public URL: [url or N/A]
Caption: [description]

If output is missing "GCS URI:" line, the diagram failed to upload to cloud storage.
""",
    tools=[
        agent_tool.AgentTool(agent=diagram_generation_agent)
    ],
)

root_agent=diagram_generating_agent