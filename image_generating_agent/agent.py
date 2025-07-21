import os
import uuid
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from pydantic import BaseModel, Field

from vertexai.preview.vision_models import ImageGenerationModel
from models.constants import GEMINI_FLASH_MODEL

def generate_visual_content(prompt: str) -> str:
    """
    Generates a visual asset based on the prompt.
    If the prompt contains 'diagram', 'flowchart', or 'cycle', it creates a diagram.
    Otherwise, it generates a realistic image using Vertex AI.
    Returns the file path of the generated image.
    """
    output_filename = f"output_{uuid.uuid4()}.png"
    print(f"--- TOOL: Received prompt: '{prompt}' ---")
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

# 2. ReviewerAgent
reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="Review the refined prompt. If it is now sufficiently detailed, set 'approved' to True. Otherwise, set 'approved' to False and give concise feedback.",
)

# 3. PromptRefinerAgent
prompt_refiner_agent = LlmAgent(
    name="PromptRefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Receive a prompt and feedback. Your primary goal is to generate a highly elaborated and exceptionally detailed 'refined_prompt' that is specifically tailored for rigorous educational purposes.

    **Strict Instructions for Educational Image Generation:**
    1.  **Contextual Understanding & Educational Goal:** Analyze the original prompt meticulously to grasp its core subject, the specific learning objectives it aims to support, and the intended educational level (e.g., elementary, high school, university).
    2.  **Educational Suitability & Safety (Non-Negotiable):** The refined prompt MUST ensure the generated image is **absolutely appropriate, exceptionally clear, and maximally beneficial** for teachers and students learning about the subject. **Under no circumstances** should the image contain elements that could be perceived as scary, confusing, misleading, or graphically disturbing. Prioritize simplicity and clarity over excessive realism if the latter compromises educational value.
    3.  **Comprehensive Elaboration and Granular Detail:** Expand the prompt significantly, integrating comprehensive descriptive elements that will meticulously guide the image generation model to create an accurate, highly informative, and visually captivating representation. Consider and include details about:
        -   **Specific Key Components/Elements:** List all essential parts, structures, or concepts that must be clearly visible and distinguishable.
        -   **Visual Style & Aesthetic:** Define the desired artistic approach (e.g., "anatomically accurate diagram," "schematic illustration with bold lines," "realistic photographic quality," "simplified cartoon for young learners," "cross-sectional view," "exploded view"). Specify color palettes (e.g., "vibrant and distinct colors," "naturalistic tones," "pastel shades"), lighting (e.g., "bright, even illumination," "soft studio lighting"), and texture.
        -   **Composition & Perspective:** Detail the exact viewpoint (e.g., "anterior view," "lateral cross-section," "overview from a slight angle," "close-up on specific organ"). Specify framing (e.g., "full body," "focused on a particular region").
        -   **Context & Environment:** If applicable, describe the surrounding environment or background to provide context without distracting from the main subject (e.g., "sterile laboratory background," "natural habitat," "against a plain white backdrop for clarity").
        -   **Educational Enhancements & Labeling:** Emphasize the need for:
            -   **Clear, Legible Labeling:** Specify if labels should be integrated directly, or if the image should be designed for subsequent labeling (e.g., "clearly defined regions for future labeling," "numbered components corresponding to a legend").
            -   **Simplified Views:** Indicate if complex subjects should be presented in a simplified, pedagogical manner.
            -   **High Resolution & Print Quality:** Ensure the image will be suitable for projection or printing.
            -   **Elimination of Distractions:** Explicitly state that background clutter or distracting elements should be avoided.
            -   **Inclusion of Scale/Proportion:** If relevant, suggest elements that convey size or scale.
    4.  **Exemplary Detail (for "human nervous system"):**
        "Generate a highly detailed, pedagogically optimized anatomical illustration of the complete human nervous system. The image should feature a clear anterior view of a human figure, with the brain, spinal cord, and all major peripheral nerves distinctly visible and accurately proportioned. Employ a **schematic yet realistic rendering style**, utilizing **vibrant, contrasting colors** to differentiate between the central nervous system (brain, spinal cord) and the peripheral nervous system, as well as distinct nerve branches. The illumination should be **bright and even**, ensuring no shadows obscure any part. The background must be a **clean, pure white** to maximize clarity and focus on the anatomy. The illustration should be designed for a **university-level anatomy textbook**, with **clearly demarcated regions suitable for subsequent labeling** by students. Absolutely no elements that could be perceived as morbid, clinical, or overly graphic should be present. The overall impression should be one of **scientific precision, accessibility, and clean educational utility.**"
    5.  **Direct Suggestion:** The output 'refined_prompt' should be a comprehensive, self-contained prompt ready for direct input to the image generation tool.
    """,
)

# ### --- CHANGED SECTION: The ImageGenerationAgent now uses the new tool --- ###
image_generation_agent = LlmAgent(
    name="ImageGenerationAgent",
    model=GEMINI_FLASH_MODEL,
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

root_agent = SequentialAgent(
    name="image_generating_agent",
    sub_agents=[
        prompt_refiner_agent,
        reviewer_agent,
        image_generation_agent,
    ]
)
