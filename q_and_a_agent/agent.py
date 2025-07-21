from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models import LlmRequest
from google.adk.tools import agent_tool, ToolContext

from models.constants import GEMINI_FLASH_MODEL
from q_and_a_agent.prompts import instruction_prompt_for_qanda, instruction_prompt_subject_extractor
from rag_agent.agent import rag_agent


class NoHistoryAgent(LlmAgent):
    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        print("🚫 Not injecting any chat history into prompt")

        # Wipe conversation history that ADK may have provided
        llm_request.messages = []

        # Optionally, just keep the last user message:
        # llm_request.messages = [{"role": "user", "content": llm_request.messages[-1]["content"]}]

        await super().process_llm_request(
            tool_context=tool_context,
            llm_request=llm_request,
        )

subject_extractor = NoHistoryAgent(
    name="subject_extractor_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=instruction_prompt_subject_extractor,
    output_key="refined_topic_query_for_rag_retrieval",
)

ragAgent = NoHistoryAgent(
    name="q_and_a_agent",
    model=GEMINI_FLASH_MODEL,
    tools=[agent_tool.AgentTool(agent=rag_agent)],
    instruction=instruction_prompt_for_qanda,
)

root_agent = SequentialAgent(
    name="root_agent_agent",
    sub_agents=[subject_extractor, ragAgent],
)