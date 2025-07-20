from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool

from topic_learner_agent.prompts import instruction_prompt_for_qanda, instruction_prompt_subject_extractor
from rag_agent.agent import ragAgent


subject_extractor = LlmAgent(
    name="subject_extractor_agent",
    model="gemini-2.5-flash",
    instruction=instruction_prompt_subject_extractor,
    output_key="refined_topic_query_for_rag_retrieval",
)

ragAgent = LlmAgent(
    name="topic_learner_agent",
    model="gemini-2.5-flash",
    tools=[agent_tool.AgentTool(agent=ragAgent)],
    instruction=instruction_prompt_for_qanda,
)

root_agent = SequentialAgent(
    name="root_agent_agent",
    sub_agents=[subject_extractor, ragAgent],
)