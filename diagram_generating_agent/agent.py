import os
import uuid
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool
from google.cloud import storage
from pydantic import BaseModel, Field

from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

from diagram_generating_agent.prompts import prompt_for_refiner_agent, prompt_for_validator_agent, \
    prompt_for_flowchart_agent, prompt_for_diagram_generating_agent
from models.constants import GEMINI_FLASH_MODEL


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

def upload_to_gcs(local_file: str, gcs_uri: str) -> bool:
    try:
        bucket_name = gcs_uri.split("/")[2]
        blob_path_jsonl = "/".join(gcs_uri.split("/")[3:])

        storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
        bucket = storage_client.bucket(bucket_name)

        blob_jsonl = bucket.blob(blob_path_jsonl)
        blob_jsonl.upload_from_filename(local_file)
        print(f"✅ Uploaded {local_file} to {gcs_uri}")

        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}")
        return False

# --- Step 2: Diagram Generation --- #
def generate_diagram(prompt: str) -> str:
    """
    Generates a diagram or image from a prompt. Returns local file path.
    """
    output_filename = f"output_{uuid.uuid4()}.png"
    print(f"--- TOOL: Received prompt: '{prompt}' ---")

    try:
        print("--- TOOL: Detected IMAGE prompt. Using Vertex AI ---")
        model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-06-06")
        seed = uuid.uuid4().int % (2 ** 32)
        print(f"Using seed: {seed}")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        # Save image
        response.images[0].save(output_filename)
        print(f"--- TOOL: Generated diagram image {os.path.abspath(output_filename)} ---")
        res = upload_to_gcs(os.path.abspath(output_filename), f"gs://shahayak-agentic-ai-gpl-muskeeters-images/image_generation/{output_filename}")
        print(f"--- TOOL: Upload to GCS  {res} ---")
        print(f"--- TOOL: Image saved to {output_filename} ---")
        return f"Image successfully generated and saved to: {output_filename}"
    except Exception as e:
        return f"Error during visual generation: {e}"


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

diagram_generation_agent = LlmAgent(
    name="diagram_generation_agent",
    model=GEMINI_FLASH_MODEL,
    description="Generates a diagram and caption based on the final prompt.",
    instruction=prompt_for_flowchart_agent,
    tools=[generate_diagram],
)

processing_agent = SequentialAgent(
    name="processing_agent",
    sub_agents=[
        prompt_refiner_agent,
        reviewer_agent,
        diagram_generation_agent
    ]
)

diagram_generating_agent = LlmAgent(
    name="diagram_generating_agent",
    model=GEMINI_FLASH_MODEL,
    description="Controls the end-to-end diagram generation pipeline.",
    instruction=prompt_for_diagram_generating_agent,
    tools=[
        agent_tool.AgentTool(agent=prompt_validator_agent),
        agent_tool.AgentTool(agent=processing_agent),
    ],
)

root_agent=diagram_generating_agent