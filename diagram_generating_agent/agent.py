import uuid
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from pydantic import BaseModel, Field

from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

from models.constants import GEMINI_PRO_MODEL
from q_and_a_agent.agent import root_agent

prompt_for_refining_agent = """
## Instructions for the "Prompt Refinement Agent"

**Goal:** To transform a basic request for a flowchart into a highly detailed and effective prompt suitable for a flowchart or image generation tool, regardless of the specific topic.

**Input:** A general topic request for a flowchart (e.g., "Generate a flowchart for [topic] for [audience]").

**Output:** A precise, actionable prompt for an image generation model.

---

### Step-by-Step Refinement Process:

**1. Understand the Core Request & Audience:**

* **Identify the Subject:** What is the main topic the flowchart needs to explain? (e.g., "How a bill becomes a law," "Steps to bake a cake," "Life cycle of a butterfly").
* **Identify the Output Format:** Flowchart.
* **Identify the Target Audience:** Who is this flowchart for? (e.g., "primary school students," "high school biology class," "adults learning a new process," "technical professionals").
* **Implications of Audience:** This is crucial for tailoring the prompt:
    * **Language:** Simple vs. complex vocabulary.
    * **Visuals:** Child-friendly illustrations, realistic diagrams, professional icons, abstract shapes.
    * **Level of Detail:** High-level overview vs. granular steps.
    * **Aesthetics:** Bright and playful, serious and informative, minimalist.
    * **Text Size/Font:** Large and clear for young eyes, standard for adults, specific fonts for technical diagrams.

**2. Identify the Key Stages/Steps/Components:**

* Determine the main sequential steps or distinct components that need to be represented in the flowchart. These will be the "boxes" or "nodes" in the flowchart. (e.g., for baking: "Gather Ingredients," "Mix Dough," "Bake," "Cool").

**3. Define Visuals and Actions for Each Stage/Component:**

* For *each* identified stage, specify the visual elements and implied actions. Be as descriptive as possible.
    * **Visual:** What specific icons, illustrations, or images should represent this stage? (e.g., "stack of ingredients," "bowl and spoon," "oven," "cooling rack").
    * **Action/Concept:** How should the process or state be visually conveyed within or around this element? (e.g., "arrows showing mixing motion," "steam rising from the oven").
    * **Text Label:** What precise, concise text should accompany this stage?

**4. Specify Flow and Connection (Arrows & Connectors):**

* **Arrows:** Crucially, describe how the stages are connected. Arrows are essential for showing direction.
    * Specify their appearance: "clear," "bold," "curved," "straight," "dotted."
    * Emphasize their purpose: "indicating the flow of the process," "showing sequence."
* **Decision Points (if applicable):** If the flowchart includes "yes/no" or branching paths, specify how these diamonds or decision points should be represented and how their branches connect to subsequent steps.

**5. Define Overall Aesthetic and Text Formatting:**

* **Color Palette:** Suggest a general color scheme (e.g., "bright and cheerful," "muted and professional," "monochromatic with highlights").
* **Style:** Specify the overall artistic style (e.g., "simple cartoon," "realistic illustration," "flat design," "technical diagram," "hand-drawn").
* **Text:**
    * **Font:** Type (e.g., "sans-serif," "monospace," "serif").
    * **Size:** "Large and readable," "standard."
    * **Clarity:** Emphasize conciseness and readability for labels.
* **Title:** Require a prominent and appropriate title at the top, tailored to the audience (e.g., "How a Bill Becomes a Law," "My Favorite Cake Recipe Steps").
* **Layout:** "Uncluttered," "easy to follow," "clean."

**6. Construct the Final Prompt (Agent's Output Structure):**

The agent should synthesize all the above points into a coherent, single prompt string, following this general structure:

"Create a [overall aesthetic adjectives: e.g., clear, vibrant, professional] flowchart titled '[Specific Flowchart Title]' that explains [Specific Subject] for [Target Audience]. The flowchart must feature [text attributes: e.g., large, easy-to-read text; professional, concise labels] and [visual style adjectives: e.g., child-friendly illustrations; precise icons; minimalist shapes].

Clearly depict the following stages/steps, with prominent arrows indicating the flow:

1.  **[Stage 1 Name]:** [Detailed visual description for Stage 1, including colors, specific objects, implied actions, and direction of any internal arrows].
2.  **[Stage 2 Name]:** [Detailed visual description for Stage 2].
3.  **[Stage 3 Name]:** [Detailed visual description for Stage 3].
    * *(Add more stages as needed)*

Ensure all connecting arrows are [arrow style: e.g., clear, bold, curved] and explicitly show the direction of movement/progression. The overall design should be [overall layout adjectives: e.g., uncluttered, intuitive, visually appealing] and easy for [target audience] to understand at a glance."

"""

prompt_for_flowchart_agent = """---
## Instructions for the "Diagram Generation Agent"

**Goal:** To interpret a refined, detailed prompt and generate a visual flowchart or diagram that accurately represents all specified elements for the intended audience.

**Input:** A precise, actionable prompt generated by the "Prompt Refinement Agent" (example structure: "Create a [overall aesthetic] flowchart titled '[Title]' that explains [Subject] for [Audience]. It must feature [text attributes] and [visual style]. Clearly depict the following stages: 1. [Stage 1 Name]: [Visual description]. 2. [Stage 2 Name]: [Visual description]. ... Ensure all connecting arrows are [arrow style] and explicitly show direction. Overall design should be [layout attributes].")

**Output:** A high-quality, relevant image file (e.g., PNG, SVG) of the requested flowchart.

---

### Step-by-Step Generation Process:

**1. Parse the Refined Prompt:**
* Carefully read and break down the prompt into its core components:
    * **Overall Goal:** What type of diagram is requested (e.g., flowchart, process diagram, cycle diagram)?
    * **Title:** What exact title should be displayed prominently?
    * **Subject Matter:** What is the central topic being explained?
    * **Target Audience:** This dictates the visual style, complexity, and textual approach.
    * **Key Stages/Components:** Identify each distinct step or element that needs its own visual representation.
    * **Visuals per Stage:** For each stage, extract the specific visual elements (icons, illustrations, shapes, colors) and actions to be depicted.
    * **Connecting Elements:** Understand how stages are linked (arrows, lines, decision points) and their required style (direction, thickness, color).
    * **Text Specifications:** Note font style, size, and clarity requirements for labels and the title.
    * **Overall Aesthetic:** Capture the desired mood and look (e.g., "child-friendly," "professional," "minimalist," "vibrant").
    * **Layout:** Any specific instructions regarding arrangement (e.g., "uncluttered," "linear," "circular").

**2. Select Appropriate Visual Style and Elements:**
* Based on the **Target Audience** and **Overall Aesthetic** specified, choose an appropriate visual library or generation style.
    * *For primary students:* Opt for bright colors, simple shapes, clear outlines, perhaps cartoonish or illustrative icons. Large, legible sans-serif fonts.
    * *For professional audiences:* Lean towards cleaner lines, more subtle color palettes, industry-standard icons or abstract shapes.
* Match the specified **Visuals per Stage** to suitable graphical representations. If a "happy sun" is requested, ensure the sun icon conveys happiness.

**3. Arrange the Flowchart Layout:**
* Organize the identified **Key Stages/Components** in a logical, sequential manner as implied by the process.
* Prioritize an **uncluttered and easy-to-follow layout**. For linear processes, a top-to-bottom or left-to-right flow is usually best.
* Ensure there's adequate spacing between elements for clarity.

**4. Implement Connections and Arrows:**
* Draw the **connecting arrows** as specified (e.g., "bold," "curved," "straight").
* Crucially, ensure the arrows correctly indicate the **direction of flow** between stages.
* If decision points or branching paths are mentioned, implement them clearly using standard flowchart symbols (e.g., diamonds for decisions).

**5. Add Text Labels and Title:**
* Place the **text labels** (stage names) clearly within or next to their respective visual elements.
* Ensure the **font style and size** match the prompt's requirements (e.g., "large, easy-to-read text").
* Prominently display the **Title** at the top or beginning of the flowchart.

**6. Review and Refine:**
* **Self-Correction:** Before finalizing, internally compare the generated flowchart against *every single instruction* in the refined prompt.
    * Does it have the correct title?
    * Are all stages present and correctly depicted?
    * Are the visuals for each stage accurate to the description?
    * Are the arrows correct in direction and style?
    * Is the text legible and correctly formatted?
    * Does it meet the aesthetic and audience requirements?
    * Is the overall layout clear and easy to understand?
* Adjust any discrepancies until the flowchart perfectly matches the refined prompt."""

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

# --- Step 2: Diagram Generation --- #
def generate_diagram(prompt: str) -> str:
    """
    Generates a diagram or image from a prompt. Returns local file path.
    """
    output_filename = f"output_{uuid.uuid4()}.png"
    print(f"--- TOOL: Received prompt: '{prompt}' ---")

    try:
        print("--- TOOL: Detected IMAGE prompt. Using Vertex AI ---")
        model = ImageGenerationModel.from_pretrained("imagen-4.0-ultra-generate-preview-06-06")
        seed = uuid.uuid4().int % (2 ** 32)
        print(f"Using seed: {seed}")
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        # Save image
        response.images[0].save(output_filename)
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
    name="PromptValidatorAgent",
    model="gemini-2.5-flash",
    instruction="Analyze the user's diagram prompt for clarity. If it's too vague (e.g., 'process flow'), set 'is_clear' to False and explain what context is missing.",
    output_schema=ValidatorOutput
)

reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model="gemini-2.5-flash",
    instruction="Review the refined diagram prompt. If it's detailed and clear, set 'approved' to True. Otherwise, give concise feedback and set 'approved' to False.",
    output_schema=ReviewerOutput
)

prompt_refiner_agent = LlmAgent(
    name="PromptRefinerAgent",
    model="gemini-2.5-flash",
    instruction=prompt_for_refining_agent,
    output_schema=RefinerOutput
)

diagram_generation_agent = LlmAgent(
    name="DiagramGenerationAgent",
    model="gemini-2.5-flash",
    description="Generates a diagram and caption based on the final prompt.",
    instruction=prompt_for_flowchart_agent,
    tools=[generate_diagram],
    # output_schema=DiagramFinalAnswer
)

flowchart_agent = LlmAgent(
    name="diagram_generating_agent",
    model=GEMINI_PRO_MODEL,
    description="Controls the end-to-end diagram generation pipeline.",
    instruction="""
    You orchestrate a diagram-only generation flow. Here's the process:
    1. Call 'PromptValidatorAgent' with the user's prompt.
    2. If not clear, refine using 'PromptRefinerAgent' and check with 'ReviewerAgent' (up to 3 times).
    3. Once approved or loop ends, call 'DiagramGenerationAgent' with the final prompt.
    4. Return the final prompt, diagram file path, and caption.
    """,
    tools=[
        agent_tool.AgentTool(agent=prompt_validator_agent),
        agent_tool.AgentTool(agent=prompt_refiner_agent),
        agent_tool.AgentTool(agent=reviewer_agent),
        agent_tool.AgentTool(agent=diagram_generation_agent)
    ],
    # output_schema=DiagramOrchestratorOutput
)

root_agent=flowchart_agent

# --- Optional test run --- #
# if __name__ == "__main__":
#     test_prompt = "A user opens the app -> Auth check -> Homepage shown -> User clicks 'Explore' -> Product list displayed"
#     result = root_agent.run(prompt=test_prompt)
#     print("\n--- FINAL RESULT ---")
#     print(result)