import os
import uuid
from typing import List

from google.adk.tools import agent_tool
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent

from common_agents.shared_rag_agent import shared_rag_agent
from lesson_planning_agent.prompts import (
    input_validator_prompt,
    planner_composer_prompt,
    refiner_prompt,
    lesson_formatter_prompt,
)
from models.constants import GEMINI_FLASH_MODEL


# --- Input and Output Models --- #
class LessonPlanInput(BaseModel):
    standard: str = Field(..., description="e.g., ['3', '4', '5'] for multi-grade class")
    subject: str = Field(..., description="Subject like 'Science', 'Math'")
    chapters: List[str] = Field(..., description="Topics or chapters to be covered")
    timeframe: str = Field(..., description="Timeframe like '2 weeks'")
    preferred_theme: str = Field(None, description="Optional cross-grade theme like 'Water', 'Farming', etc.")


class LessonPlanOutput(BaseModel):
    validated: bool = Field(..., description="Input validation result")
    aligned_chapters: List[str] = Field(..., description="NCERT-aligned and deduplicated chapter list")
    lesson_plan: str = Field(..., description="Multi-grade synchronized lesson plan")
    refinement_question: str = Field(..., description="Friendly follow-up suggestion or question")


# --- Sub-Agents --- #

# 1. Input Validator Agent
clarifier_agent = LlmAgent(
    name="input_validator_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=input_validator_prompt,
    # output_schema=LessonPlanOutput
)

# 3. Planner Composer Agent
planner_composer_agent = LlmAgent(
    name="PlannerComposerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=planner_composer_prompt,
    # output_schema=LessonPlanOutput
)

# 4. Refiner Agent
refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=refiner_prompt,
    # output_schema=LessonPlanOutput
)

formatter_agent = LlmAgent(
    name="LessonPlanFormatterAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=lesson_formatter_prompt,
)


# --- Root Agent: Lesson Planner Orchestrator --- #
processing_agent = SequentialAgent(
    name="ProcessingAgent",
    description="Interactive lesson planning agent with NCERT alignment and multimodal guidance",
    sub_agents=[
        shared_rag_agent,
        planner_composer_agent,
        refiner_agent,
        formatter_agent
    ],
)

lesson_planning_agent = LlmAgent(
    name="LessonPlanningAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
You are a structured lesson planning orchestrator for a classroom-focused AI system. Your job is to coordinate between agents and return a weekly plan for the teacher. Follow the flow below **exactly**:

---

1. ✅ First, call the `ClarifierAgent` to ensure the input is complete and aligned.

2. ❓ If the clarifier response has `needs_clarification = true`:
   - Immediately return ONLY the `follow_up_question`.
   - Do NOT proceed further.

3. ✅ If `needs_clarification = false`:
   - Proceed to call the `ProcessingAgent` to generate the full weekly lesson plan.

---

⚠️ CRITICAL INSTRUCTIONS — DO NOT VIOLATE:
- NEVER call `ClarifierAgent` again after calling `ProcessingAgent`.
- NEVER modify, enhance, simplify, or reformat the output returned by `ProcessingAgent`.
- Do NOT inject tone changes, extra summaries, or section headings.
- If `ProcessingAgent` already contains a summary or multi-week plan, return it exactly as is.

---

Your role is strictly that of an orchestrator. Any deviation from these steps will result in logic failure.
""",
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ]
)


root_agent = lesson_planning_agent
