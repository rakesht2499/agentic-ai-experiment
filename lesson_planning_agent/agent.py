import os
import uuid
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.adk.agents import SequentialAgent, ParallelAgent

from models.constants import GEMINI_FLASH_MODEL
from lesson_planning_agent.prompts import (
    input_validator_prompt,
    rag_prompt,
    planner_composer_prompt,
    refiner_prompt,
    lesson_formatter_prompt
)


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
input_validator_agent = LlmAgent(
    name="input_validator_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=input_validator_prompt,
    output_schema=LessonPlanOutput
)

# 2. RAG Agent
rag_agent = LlmAgent(
    name="rag_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=rag_prompt,
    output_schema=LessonPlanOutput
)

# 3. Planner Composer Agent
planner_composer_agent = LlmAgent(
    name="PlannerComposerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=planner_composer_prompt,
    output_schema=LessonPlanOutput
)

# 4. Refiner Agent
refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=refiner_prompt,
    output_schema=LessonPlanOutput
)

post_validation_parallel_agent = ParallelAgent(
    name="PostValidationParallelAgent",
    sub_agents=[
        rag_agent,
        planner_composer_agent,
        refiner_agent
    ]
)

formatter_agent = LlmAgent(
    name="LessonPlanFormatterAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=lesson_formatter_prompt,
    output_schema=LessonPlanOutput
)


# --- Root Agent: Lesson Planner Orchestrator --- #
root_agent = SequentialAgent(
    name="lesson_planning_agent",
    # model=GEMINI_PRO_MODEL,
    description="Interactive lesson planning agent with NCERT alignment and multimodal guidance",
    # instruction="""
    # Step-by-step plan:
    # 1. Call InputValidatorAgent to check and confirm user inputs.
    # 2. If input is invalid, return refinement_question to user and stop.
    # 3. Call RagAgent to align chapters with NCERT syllabus.
    # 4. Call PlannerComposerAgent to generate the lesson plan.
    # 5. Call RefinerAgent to generate a follow-up question.
    # Finally, return:
    # - validated input status,
    # - aligned chapters,
    # - lesson plan text,
    # - refinement question
    # """,
    sub_agents=[
        input_validator_agent,
        post_validation_parallel_agent,
        formatter_agent
    ],
)
