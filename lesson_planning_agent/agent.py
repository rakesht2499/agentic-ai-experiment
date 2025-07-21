import os
import uuid
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL


# --- Input and Output Models --- #
class LessonPlanInput(BaseModel):
    standard: str = Field(..., description="e.g., '8' for Class 8")
    subject: str = Field(..., description="Subject like 'Science', 'Math'")
    chapters: List[str] = Field(..., description="List of chapter names")
    timeframe: str = Field(..., description="Timeframe like '2 weeks'")


class LessonPlanOutput(BaseModel):
    validated: bool = Field(..., description="Input validation result")
    aligned_chapters: List[str] = Field(..., description="Chapters aligned with NCERT syllabus")
    lesson_plan: str = Field(..., description="Generated lesson plan")
    refinement_question: str = Field(..., description="Follow-up question for refinement")


# --- Sub-Agents --- #

# 1. Input Validator Agent
input_validator_agent = LlmAgent(
    name="InputValidatorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Validate the lesson planning input. If any of 'standard', 'subject', 'chapters', or 'timeframe' is missing,
    return validated=False with a meaningful refinement_question asking for the missing parts.
    If all fields are present, return validated=True.
    """,
    output_schema=LessonPlanOutput
)

# 2. RAG Agent
rag_agent = LlmAgent(
    name="RagAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    You are connected to NCERT content. Compare the chapter list with official NCERT chapters for the given class and subject.
    Return only those chapters that are valid.
    """,
    output_schema=LessonPlanOutput
)

# 3. Planner Composer Agent
planner_composer_agent = LlmAgent(
    name="PlannerComposerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Generate a detailed lesson plan based on the chapters provided. Format as follows:

    Chapter: [Chapter Name]
    - Learning Objectives:
    - Key Concepts:
    - Suggested Activities:
    - Assessment Ideas:

    Repeat this for each chapter. Output should be a single string.
    """,
    output_schema=LessonPlanOutput
)

# 4. Refiner Agent
refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Ask a user-friendly follow-up question like:
    'Would you like to include hands-on experiments or group activities?'
    Put your response only inside the `refinement_question` field.
    """,
    output_schema=LessonPlanOutput
)

# --- Root Agent: Lesson Planner Orchestrator --- #
root_agent = LlmAgent(
    name="lesson_planner_agent",
    model=GEMINI_PRO_MODEL,
    description="Interactive lesson planning agent with NCERT alignment and multimodal guidance",
    instruction="""
    Step-by-step plan:
    1. Call InputValidatorAgent to check and confirm user inputs.
    2. If input is invalid, return refinement_question to user and stop.
    3. Call RagAgent to align chapters with NCERT syllabus.
    4. Call PlannerComposerAgent to generate the lesson plan.
    5. Call RefinerAgent to generate a follow-up question.
    Finally, return:
    - validated input status,
    - aligned chapters,
    - lesson plan text,
    - refinement question
    """,
    tools=[
        agent_tool.AgentTool(agent=input_validator_agent),
        agent_tool.AgentTool(agent=rag_agent),
        agent_tool.AgentTool(agent=planner_composer_agent),
        agent_tool.AgentTool(agent=refiner_agent)
    ],
    # output_schema=LessonPlanOutput
)
