import uuid
import os
from google.adk.agents import LlmAgent, InvocationContext
from google.adk.tools import agent_tool, BaseTool, ToolContext
from pydantic import BaseModel, Field

from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

from diagram_generating_agent.prompts import prompt_for_refiner_agent, prompt_for_validator_agent, prompt_for_flowchart_agent, prompt_for_diagram_generating_agent
from models.constants import GEMINI_FLASH_MODEL

# For Python 3.11 compatibility
try:
    from typing import override
except ImportError:
    def override(func):
        return func


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

# --- Step 2: Diagram Generation Tool --- #
class DiagramGeneratorTool(BaseTool):
    """Proper ADK tool for generating diagrams"""

    def __init__(self):
        super().__init__(
            name="generate_diagram",
            description="Generates a diagram or image from a prompt and returns the local file path"
        )

    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> str:
        """
        Generates a diagram or image from a prompt. Returns local file path.
        """
        # Get the prompt from the context input - handle different input formats
        prompt = ""
        if hasattr(context, 'input') and context.input:
            if isinstance(context.input, dict):
                prompt = context.input.get('prompt', '') or context.input.get('text', '') or str(context.input)
            elif hasattr(context.input, 'prompt'):
                prompt = context.input.prompt
            elif hasattr(context.input, 'text'):
                prompt = context.input.text
            else:
                prompt = str(context.input)

        # If still no prompt, try to get it from the tool context or use a default
        if not prompt and hasattr(tool_context, 'input'):
            prompt = str(tool_context.input)

        if not prompt or prompt.strip() == "":
            return "Error: No prompt provided for diagram generation. Please provide a description of the diagram you want to create."

        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "generated_diagrams")
        os.makedirs(output_dir, exist_ok=True)

        # Create full file path
        filename = f"diagram_{uuid.uuid4().hex[:8]}.png"
        output_filepath = os.path.join(output_dir, filename)

        print(f"--- TOOL: Received prompt: '{prompt}' ---")
        print(f"--- TOOL: Will save to: {output_filepath} ---")

        try:
            print("--- TOOL: Using Vertex AI Image Generation ---")

            # Try multiple model versions for better compatibility
            model_versions = [
                "imagen-4.0-generate-preview-06-06",
                "imagegeneration@006",
                "imagegeneration@005"
            ]

            model = None
            for version in model_versions:
                try:
                    print(f"Trying model version: {version}")
                    model = ImageGenerationModel.from_pretrained(version)
                    break
                except Exception as version_error:
                    print(f"Failed to load {version}: {version_error}")
                    continue

            if not model:
                return "Error: Unable to load any image generation model. Please check your Vertex AI configuration and permissions."

            seed = uuid.uuid4().int % (2 ** 32)
            print(f"Using seed: {seed}")

            # Generate the image
            response = model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="1:1"
            )

            # Save image with better error handling
            if response and response.images and len(response.images) > 0:
                try:
                    response.images[0].save(output_filepath)

                    # Verify the file was actually saved
                    if os.path.exists(output_filepath):
                        file_size = os.path.getsize(output_filepath)
                        print(f"--- TOOL: Image saved successfully to {output_filepath} (Size: {file_size} bytes) ---")
                        return f"Image successfully generated and saved to: {output_filepath}"
                    else:
                        return f"Error: Image generation completed but file was not saved to {output_filepath}. Check directory permissions."

                except Exception as save_error:
                    return f"Error saving image to {output_filepath}: {str(save_error)}"
            else:
                return "Error: No image was generated by the model. Please try a different prompt."

        except Exception as e:
            error_msg = f"Error during image generation: {str(e)}"
            print(error_msg)
            # Provide more helpful error message
            if "403" in str(e) or "permission" in str(e).lower():
                error_msg += "\nThis might be a permissions issue. Please check your Vertex AI setup and ensure image generation is enabled."
            elif "quota" in str(e).lower():
                error_msg += "\nThis might be a quota issue. Please check your Vertex AI quotas."
            return error_msg

# Create the tool instance
diagram_generator_tool = DiagramGeneratorTool()


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
    instruction="""
You are a diagram generation agent. Your job is to:

1. Receive a refined prompt for diagram generation
2. Call the generate_diagram tool with the prompt to create the actual diagram
3. Return both the generated file path and a descriptive caption

When calling generate_diagram, make sure to pass the prompt as a string parameter.

If successful, respond with:
- File path: [the file path returned by the tool]
- Caption: [a brief description of what the diagram shows]

If there's an error, explain what went wrong and suggest next steps.
""",
    tools=[diagram_generator_tool],
)

diagram_generating_agent = LlmAgent(
    name="diagram_generating_agent",
    model=GEMINI_FLASH_MODEL,
    description="Controls the end-to-end diagram generation pipeline.",
    instruction="""
You orchestrate a diagram generation flow. Follow these steps exactly:

1. **Validation**: Call 'prompt_validator_agent' to evaluate if the user's prompt is clear and complete.
   - If validation fails, return the feedback message to ask for clarification.
   - Do NOT proceed if essential information is missing.

2. **Refinement** (if needed): Call 'prompt_refiner_agent' to improve the prompt.
   - Then call 'reviewer_agent' to approve the refined prompt.
   - You may repeat refinement up to 2 times if not approved.

3. **Generation**: Once you have a clear, approved prompt, call 'diagram_generation_agent' with the final prompt.
   - The diagram_generation_agent will handle calling the generate_diagram tool internally.
   - It will return both the file path and caption.

4. **Response**: Return the results clearly:
   - Diagram file: [filename]
   - Caption: [description]

**CRITICAL ERROR HANDLING**:
- If any step fails, provide specific error details rather than generic messages.
- If diagram generation fails, suggest alternative approaches or prompt modifications.
- Always try to complete the process rather than giving up with vague error messages.

**NEVER** respond with generic error messages like "internal error" without trying the actual tools first.
""",
    tools=[
        agent_tool.AgentTool(agent=prompt_validator_agent),
        agent_tool.AgentTool(agent=prompt_refiner_agent),
        agent_tool.AgentTool(agent=reviewer_agent),
        agent_tool.AgentTool(agent=diagram_generation_agent)
    ],
)

root_agent=diagram_generating_agent