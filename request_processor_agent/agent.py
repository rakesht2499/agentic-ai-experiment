import os
from typing import Optional
from pydantic import BaseModel, Field
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import agent_tool

from q_and_a_agent.agent import root_agent as q_and_a_agent
from diagram_generating_agent.agent import root_agent as diagram_generating_agent
from image_generating_agent.agent import root_agent as image_generating_agent

# --- Set your GCP environment ---
# os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"  # Update this
# os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

def createToolFromAgent(agent):
    return agent_tool.AgentTool(agent=agent)

instruction_prompt_root_agent = """
🎯 Your Role:
You are a **routing agent** in an AI-powered educational assistant system. Your sole responsibility is to decide **which specialized agent should handle a user's input** based on the type of query.

You will **NOT answer the question yourself.**  
You will **not include additional explanations** — just call the correct tool/agent for the task.

There are three tools available to you:

---

🧠 Tool 1: Q&A Agent (`ask_rag_agent`)
Use this for:
- Academic questions like definitions, explanations, or textbook-based doubts.
- Conceptual queries like "What is photosynthesis?" or "Explain Newton’s First Law."
- Subject-specific questions related to **Science, Math, Social Science, etc.**
- Anything that sounds like a textbook-based question or a classroom doubt.

❌ Do NOT use this for:
- Diagram generation
- Visual or creative content generation

---

📊 Tool 2: Flowchart/Diagram Generator Agent (`diagram_generating_agent`)
Use this for:
- Requests to generate **flowcharts, cycle diagrams, or step-by-step visuals**
- Examples: 
  - "Create a flowchart of the water cycle"
  - "Give me a diagram of the digestive system"
  - "Show the life cycle of a frog"

These are typically **academic visualizations** tied to textbook concepts.

❌ Do NOT use this for:
- Creative image generation like "A tiger chasing a deer"
- General Q&A

---

🎨 Tool 3: Image Generation Agent (`image_generating_agent`)
Use this for:
- Creative, visual imagery unrelated to textbook diagrams
- Examples:
  - "Draw a tiger in a jungle"
  - "Show a mountain landscape at sunset"
  - "Generate an image of a classroom"

This is used when the request is for **art-style illustrations or realistic images**.

❌ Do NOT use this for:
- Educational diagrams
- Q&A or textbook explanations

---

🔍 Decision Rules:
- Read the user's input carefully.
- Determine the intent: Is it a question, a diagram request, or an image request?
- Select the **most appropriate tool** and route the query to that agent only.
- Be strict — always call only ONE of the three tools per input.

---

🧪 Examples:

User: "Explain the process of photosynthesis"  
➡️ Use Tool 1: Q&A Agent

User: "Make a flowchart showing the process of blood circulation"  
➡️ Use Tool 2: Diagram Generator

User: "Show me a dense forest with animals in it"  
➡️ Use Tool 3: Image Generator

User: "Define Newton’s Laws"  
➡️ Use Tool 1: Q&A Agent

User: "Draw the stages of water cycle"  
➡️ Use Tool 2: Diagram Generator

User: "Generate an image of a teacher in a village classroom"  
➡️ Use Tool 3: Image Generator

---

⚠️ Important:
- Do not try to answer the question yourself.
- Do not generate multiple responses.
- Always call only the most relevant tool based on the content and format of the question.

"""
# --- Input Schema ---
class MultimodalUserInput(BaseModel):
    text: Optional[str] = Field(None, description="User's textual input")
    image: Optional[bytes] = Field(None, description="Uploaded image content (binary)")
    audio: Optional[bytes] = Field(None, description="Uploaded audio content (binary)")
    video: Optional[bytes] = Field(None, description="Uploaded video content (binary)")

# --- Output Schema ---
class MultimodalAnalysisOutput(BaseModel):
    combined_insights: str = Field(..., description="Summary of combined analysis")
    next_agent: str = Field(..., description="Suggested downstream agent")

# --- Sub-Agent: Text Analyzer ---
class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    text_insight: str

text_agent = LlmAgent(
    name="TextInsightAgent",
    model="gemini-2.5-flash",
    instruction="Analyze the given text and extract key details.",
    input_schema=TextInput,
    output_schema=TextOutput
)

# --- Sub-Agent: Image Analyzer ---
class ImageInput(BaseModel):
    image: bytes

class ImageOutput(BaseModel):
    image_insight: str

image_agent = LlmAgent(
    name="ImageInsightAgent",
    model="gemini-2.5-flash",
    instruction="If image is present analyze the uploaded image content and summarize what is seen, if image not present no need to do anything",
    input_schema=ImageInput,
    output_schema=ImageOutput
)

# --- Sub-Agent: Audio Analyzer ---
class AudioInput(BaseModel):
    audio: bytes

class AudioOutput(BaseModel):
    audio_insight: str

audio_agent = LlmAgent(
    name="AudioInsightAgent",
    model="gemini-2.5-flash",
    instruction="If audio is provided or present then transcribe and summarize the audio file provided, if not present then no need to do anything",
    input_schema=AudioInput,
    output_schema=AudioOutput
)

# --- Sub-Agent: Video Analyzer ---
class VideoInput(BaseModel):
    video: bytes

class VideoOutput(BaseModel):
    video_insight: str

video_agent = LlmAgent(
    name="VideoInsightAgent",
    model="gemini-2.5-flash",
    instruction="If video is present then summarize what happens in the uploaded video, else no need to do anything",
    input_schema=VideoInput,
    output_schema=VideoOutput
)

# --- Fusion Agent: Combines all available insights ---
class FusionInput(BaseModel):
    text_insight: Optional[str]
    image_insight: Optional[str]
    audio_insight: Optional[str]
    video_insight: Optional[str]

fusion_agent = LlmAgent(
    name="FusionAgent",
    model="gemini-2.5-flash",
    instruction=instruction_prompt_root_agent,
    # instruction="Combine text, image, audio, and video insights into a cohesive summary and recommend the next best agent.",
    input_schema=FusionInput,
    # output_schema=MultimodalAnalysisOutput
    tools=[
            createToolFromAgent(q_and_a_agent),
            createToolFromAgent(diagram_generating_agent),
            createToolFromAgent(image_generating_agent),
        ]
)

# --- Root Orchestration Agent ---
root_agent = SequentialAgent(
    name="request_processor_agent",
    sub_agents=[
        text_agent,
        image_agent,
        audio_agent,
        video_agent,
        fusion_agent,
    ],
)