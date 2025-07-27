import os
from typing import Literal, Optional
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import agent_tool

from common_agents.shared_rag_agent import shared_rag_agent, vector_rag_agent
from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL


class SyllabusPlannerInput(BaseModel):
    scope: Literal["Full Year", "Exam Window", "Custom Range"] = Field(..., description="Planning duration")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    subject: str = Field(..., description="Subject name, e.g., Science")
    standard: str = Field(..., description="Grade level, e.g., Class 8")


class SyllabusPlannerOutput(BaseModel):
    validated: bool = Field(..., description="Whether the input has been validated")
    topic_list: Optional[str] = Field("", description="List of NCERT topics")
    syllabus_calendar: Optional[str] = Field("", description="Week-wise or date-wise calendar")
    refinement_question: Optional[str] = Field("", description="Follow-up question for improvement")


# --- Sub-Agents --- #

# 1. Scope Clarifier Agent
scope_clarifier = LlmAgent(
    name="ScopeClarifierAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Check if the provided scope ('Full Year', 'Exam Window', or 'Custom Range') and dates are valid and sufficient.

    If incomplete:
      - validated = False
      - Ask what is missing via refinement_question.

    If complete:
      - validated = True
      - refinement_question = ""
    """,
    output_schema=SyllabusPlannerOutput
)

# 3. Calendar Mapper Agent
calendar_mapper = LlmAgent(
    name="CalendarMapperAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Create a structured week-wise or date-wise calendar plan between 'start_date' and 'end_date'.

    Use the list of topics to evenly distribute content.

    Format:
    - Week 1 (YYYY-MM-DD to YYYY-MM-DD): Topic 1, Topic 2
    - Week 2 (YYYY-MM-DD to YYYY-MM-DD): Topic 3, Topic 4
    ...

    No commentary. Ensure all topics are covered.
    """,
    output_schema=SyllabusPlannerOutput
)

# 4. Planner Refiner Agent
planner_refiner = LlmAgent(
    name="PlannerRefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Suggest ONE improvement or personalization follow-up for the syllabus plan.

    Examples:
    - Would you like to include revision weeks?
    - Should we add mock test checkpoints?
    - Want difficulty tagging for each topic?

    Only populate `refinement_question`.
    Leave other fields blank.
    """,
    output_schema=SyllabusPlannerOutput
)

processing_agent = SequentialAgent(
    name="processing_agent",
    sub_agents=[
        vector_rag_agent,
        calendar_mapper,
        planner_refiner
    ]
)

# --- Root Orchestrator Agent --- #
root_agent = LlmAgent(
    name="syllabus_planner_agent",
    model=GEMINI_PRO_MODEL,
    description="Calendar-based syllabus planner for NCERT classes",
    instruction="""
    Step-by-step behavior:

    1. Validate the scope and dates using ScopeClarifierAgent.
       - If validated=False, return refinement_question and stop.

    2. Fetch NCERT topic list using SharedRagAgent based on subject and standard.
       - SharedRagAgent returns structured output: {"subject": "Science", "class_": "Class 10", "content": "topic list..."}
       - Extract the topic list from the content field
       - If content field contains "RAG_RETRIEVAL_FAILED", use a fallback topic list for the subject

    3. Map the topic list to a date-wise plan using CalendarMapperAgent.
       - Pass the extracted topic list from SharedRagAgent's content field
       - Ensure balanced distribution.

    4. Ask a helpful refinement question using PlannerRefinerAgent.

    Final Output:
    - validated
    - topic_list (extracted from SharedRagAgent content field)
    - syllabus_calendar
    - refinement_question
    """,
    tools=[
        agent_tool.AgentTool(agent=scope_clarifier),
        agent_tool.AgentTool(agent=processing_agent)
    ]
)
