import asyncio
import os
from google.adk import App
from google.adk.agents import Agent
from google.adk.live import LiveSessionConfig
from google.adk.io import Text, AudioOutput
from google.adk.tools import Tool, tool

# --- Tool Definition ---
# This tool is designed to provide specific, pre-defined information in Hindi.
# It demonstrates how an agent can use a tool to fetch language-specific content,
# rather than performing a direct translation itself. For general content generation
# (like writing an essay), the LLM's core generative capabilities will be used,
# guided by its instruction to respond in English.
@tool
def get_hindi_info(topic: str) -> str:
    """
    Provides a short paragraph of information about a given topic in Hindi.
    The agent will call this tool if it determines the user is asking for
    information on a relevant topic that matches the tool's capabilities.
    """
    # Simple logic to return a Hindi paragraph based on the topic.
    # The LLM's instruction will guide it to use this tool for specific Hindi responses.
    if "पेड़" in topic or "वृक्ष" in topic or "trees" in topic.lower():
        return (
            "पेड़ हमारे पर्यावरण के लिए बहुत महत्वपूर्ण हैं। वे हमें ऑक्सीजन देते हैं, "
            "कार्बन डाइऑक्साइड अवशोषित करते हैं, और पक्षियों तथा जानवरों के लिए घर प्रदान करते हैं। "
            "हमें अधिक पेड़ लगाने चाहिए और उनकी रक्षा करनी चाहिए।"
        )
    elif "मौसम" in topic.lower() or "weather" in topic.lower():
        return "आज का मौसम सुहावना है और धूप खिली हुई है।"
    else:
        return f"मुझे '{topic}' के बारे में हिंदी में जानकारी नहीं मिल पा रही है। कृपया किसी अन्य विषय के बारे में पूछें।"

# --- Agent Definition ---
# This defines the core intelligence and behavior of your agent.
root_agent = Agent(
    name="essay_audio_agent",
    # Use the Gemini Live 2.5 Flash Preview model for real-time, multilingual capabilities.
    model="gemini-live-2.5-flash-preview",
    description=(
        "An agent that can write essays on various topics in English and respond with audio."
    ),
    instruction=(
        "You are a helpful agent that can write essays on topics requested by the user. "
        "When asked to write an essay, generate a comprehensive and informative essay in English. "
        "Focus on providing detailed explanations suitable for the requested audience. "
        "For example, if asked about 'trees for an 8th standard student', write an essay explaining the importance of trees "
        "in simple, engaging English. "
        "Ensure your responses are ready to be converted to audio output."
    ),
    # The agent is equipped with the tool to provide Hindi information,
    # but its primary essay generation will come from the LLM itself.
    tools=[get_hindi_info],
)

# --- ADK App Initialization ---
# This sets up the ADK application.
app = App()

# Register the agent with the ADK application.
app.agent(root_agent)

# --- Live Session Definition ---
# This defines how the ADK application handles real-time interactions (like voice or text chat).
# The runtime_config is crucial here for setting the language for audio output.
@app.live_session(
    # Configure the live session for English (en-US) for both STT and TTS.
    # This ensures that when the agent's text response is generated,
    # it is then converted into English speech.
    runtime_config=LiveSessionConfig(
        language_code="en-US"
    )
)
async def essay_audio_session(ctx):
    """
    Handles real-time voice/text conversations.
    It receives user input, passes it to the agent, and sends back the agent's response as audio.
    """
    # Get the initial user input. This will be text, even if spoken (after STT).
    user_input: Text = await ctx.input.get_next(Text)

    # Invoke the agent with the user's input.
    # The agent, guided by its instruction, will process the English input
    # and generate an essay response in English.
    agent_response = await root_agent.invoke(user_input)

    # Send the agent's response back to the user as audio.
    # The runtime_config's language_code="en-US" will ensure Text-to-Speech
    # generates English audio from the agent's English text response.
    await ctx.output.send(AudioOutput(text=agent_response.text))

# To run this ADK application locally:
# 1. Save the code as 'main.py' in a new directory (e.g., 'my_audio_agent').
# 2. Ensure you have the 'google-adk' library installed: `pip install google-adk`
# 3. Set your Google API Key as an environment variable: `export GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
#    (Replace "YOUR_API_KEY_HERE" with your actual API key from Google AI Studio).
#    Alternatively, configure Google Cloud Vertex AI credentials if preferred for production.
# 4. Navigate to your project directory in the terminal and run: `adk web`
# 5. Open your browser to the URL provided by 'adk web' (usually http://localhost:8000)
#    and interact with your agent. When you ask for an essay, the response will be played as audio.
