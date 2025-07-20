import os
import uuid
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from pydantic import BaseModel, Field
from typing import TypedDict

# ### --- NEW SECTION: Imports for real image and diagram generation --- ###
from vertexai.preview.vision_models import ImageGenerationModel
import graphviz

# --- Configuration for Vertex AI --- ###
# IMPORTANT: Replace with your Google Cloud project and location
os.environ["GOOGLE_CLOUD_PROJECT"] = "image-generation-sahayak"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# ### --- CHANGED SECTION: The new, powerful tool that replaces the dummy one --- ###
def generate_visual_content(prompt: str) -> str:
    """
    Generates a visual asset based on the prompt.
    If the prompt contains 'diagram', 'flowchart', or 'cycle', it creates a diagram.
    Otherwise, it generates a realistic image using Vertex AI.
    Returns the file path of the generated image.
    """
    output_filename = f"output_{uuid.uuid4()}.png"
    print(f"--- TOOL: Received prompt: '{prompt}' ---")

    # Decision logic: Diagram or Image?
    # if any(keyword in prompt.lower() for keyword in ["diagram", "flowchart", "cycle"]):
    #     print("--- TOOL: Detected diagram request. Using Graphviz. ---")
    #     try:
    #         # Simple example for the water cycle
    #         dot = graphviz.Digraph('WaterCycle', comment='The Water Cycle')
    #         dot.edge('Ocean', 'Evaporation')
    #         dot.edge('Evaporation', 'Condensation (Clouds)')
    #         dot.edge('Condensation (Clouds)', 'Precipitation')
    #         dot.edge('Precipitation', 'Collection (Ocean)')
    #         dot.render(outfile=output_filename, format='png', view=False, cleanup=True)
    #         print(f"--- TOOL: Diagram saved to {output_filename} ---")
    #         return f"Diagram successfully generated and saved to: {output_filename}"
    #     except Exception as e:
    #         return f"Failed to generate diagram: {e}"
    # else:
    print("--- TOOL: Detected image request. Using Vertex AI Imagen. ---")
    try:
        # Initialize Vertex AI client
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        seed = uuid.uuid4().int % (2 ** 32)
        print(f"Using seed: {seed}")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        # Save the generated image to a file
        response.images[0].save(output_filename)
        print(f"--- TOOL: Image saved to {output_filename} ---")
        return f"Image successfully generated and saved to: {output_filename}"
    except Exception as e:
        return f"Failed to generate image from Vertex AI: {e}"

# --- Agent Type Definitions (using Pydantic) ---

class FinalAnswer(BaseModel):
    image_result: str = Field(description="File path of the generated image.")
    caption: str = Field(description="Descriptive caption for the generated image.")

class ValidatorOutput(BaseModel):
    is_clear: bool = Field(description="Indicates whether the prompt is clear enough.")
    feedback: str = Field(description="Feedback or suggestions to improve the prompt.")

class RefinerOutput(BaseModel):
    refined_prompt: str = Field(description="The improved version of the user's prompt.")

class ReviewerOutput(BaseModel):
    approved: bool = Field(description="Whether the prompt is good enough to proceed.")
    feedback: str = Field(description="Specific comments for improvement.")

# --- Agent Definitions ---

# 1. PromptValidatorAgent
prompt_validator_agent = LlmAgent(
    name="PromptValidatorAgent",
    model="gemini-1.5-flash",
    instruction="Analyze the user's prompt for clarity (subject, context, elements). If ambiguous (e.g., 'the water cycle'), set 'is_clear' to False and provide feedback on what's missing.",
    output_schema=ValidatorOutput
)

# 2. ReviewerAgent
reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model="gemini-1.5-flash",
    instruction="Review the refined prompt. If it is now sufficiently detailed, set 'approved' to True. Otherwise, set 'approved' to False and give concise feedback.",
    output_schema=ReviewerOutput
)

# 3. PromptRefinerAgent
prompt_refiner_agent = LlmAgent(
    name="PromptRefinerAgent",
    model="gemini-1.5-flash",
    instruction="Receive a prompt and feedback. Generate a 'refined_prompt' that is either a direct suggestion or a clarifying question to the user.",
    output_schema=RefinerOutput
)

# ### --- CHANGED SECTION: The ImageGenerationAgent now uses the new tool --- ###
image_generation_agent = LlmAgent(
    name="ImageGenerationAgent",
    model="gemini-1.5-flash",
    description="Generates an image and caption from a final prompt.",
    instruction="""
    You will receive a final, refined prompt.
    1. Call the `generate_visual_content` tool with this prompt.
    2. Create a concise, descriptive caption for the image based on the prompt.
    3. Return the file path from the tool and the caption.
    """,
    tools=[generate_visual_content]
)


# --- Main Orchestrator Agent ---
# ### --- CHANGED SECTION: Updated instructions and output schema for the orchestrator --- ###
class OrchestratorOutput(BaseModel):
    final_prompt: str = Field(description="The final prompt used for generation.")
    image_file_path: str = Field(description="The local file path of the generated image.")
    final_caption: str = Field(description="The final caption for the image.")

root_agent = LlmAgent(
    name="image_generating_agent",
    model="gemini-1.5-flash",
    description="Orchestrates the entire image generation flow from prompt validation to final output.",
    instruction="""
    You are an orchestrator for a visual content pipeline.
    Follow these steps:
    1.  Call `PromptValidatorAgent` with the initial prompt.
    2.  If the prompt is not clear, enter a loop (max 3 times):
        a. Call `PromptRefinerAgent` with the prompt and feedback.
        b. Call `ReviewerAgent` with the newly refined prompt.
        c. If the reviewer approves, exit the loop. Otherwise, use the new feedback for the next iteration.
    3.  Once a prompt is approved (or the loop finishes), call `ImageGenerationAgent` with the final prompt.
    4.  Extract the file path and caption from the result.
    5.  Return the final prompt, the image file path, and the caption as the final answer.
    """,
    # output_schema=OrchestratorOutput,
    tools=[
        agent_tool.AgentTool(agent=prompt_validator_agent),
        agent_tool.AgentTool(agent=reviewer_agent),
        agent_tool.AgentTool(agent=prompt_refiner_agent),
        agent_tool.AgentTool(agent=image_generation_agent),
    ],
)