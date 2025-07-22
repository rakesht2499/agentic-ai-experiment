from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from models.constants import GEMINI_PRO_MODEL
from request_processor_agent.agent import root_agent as request_processor_root_agent

def createToolFromAgent(agent):
    return agent_tool.AgentTool(agent=agent)

instruction_prompt_root_agent = """
🎯 Your Role:
You are a **routing agent** in an AI-powered educational assistant system. Your sole responsibility is to route all the inputs provided to you to **request_processor_agent**.

You will **NOT answer the question yourself.**  
You will **not include additional explanations** — just call this *request_processor_agent* tool/agent for the task.
"""

root_agent = LlmAgent(
    name="shahayak_agent",
    model=GEMINI_PRO_MODEL,
    instruction=instruction_prompt_root_agent,
    tools=[
        createToolFromAgent(request_processor_root_agent),
    ],
)
