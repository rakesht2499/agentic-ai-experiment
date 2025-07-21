import os
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from models.constants import GEMINI_PRO_MODEL

# --- Environment Setup ---
os.environ["GOOGLE_CLOUD_PROJECT"] = "image-generation-sahayak"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# --- Input and Output Schemas --- #
class SyllabusPlannerInput(BaseModel):
    standard: str = Field(..., description="e.g., '10' for Class 10")
    subject: str = Field(..., description="e.g., 'Science'")
    scope: str = Field(..., description="'Full Year', 'Exam Window', or 'Custom Range'")
    start_date: str = Field(..., description="Start date for planning in YYYY-MM-DD")
    end_date: str = Field(..., description="End date for planning in YYYY-MM-DD")

class SyllabusPlannerOutput(BaseModel):
    validated: bool = Field(..., description="Whether input was valid and sufficient")
    topic_list: List[str] = Field(..., description="List of topics extracted from NCERT for planning")
    syllabus_calendar: str = Field(..., description="Calendar plan spread across weeks or dates")
    refinement_question: str = Field(..., description="Follow-up question to personalize or clarify planning")

# --- Sub-Agents --- #

# 1. Scope Clarifier Agent
scope_clarifier = LlmAgent(
    name="ScopeClarifierAgent",
    model=GEMINI_PRO_MODEL,
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

# 2. Topic Fetcher Agent (RAG-based)
rag_topic_fetcher = LlmAgent(
    name="RagAgent",
    model=GEMINI_PRO_MODEL,
    instruction="""
    Fetch NCERT topics based on standard and subject.
    Output must be a clean list of topic titles, one per line.
    Do NOT include explanations or commentary.
    """,
    output_schema=SyllabusPlannerOutput
)

# 3. Calendar Mapper Agent
calendar_mapper = LlmAgent(
    name="CalendarMapperAgent",
    model=GEMINI_PRO_MODEL,
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
    model=GEMINI_PRO_MODEL,
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

# --- Root Orchestrator Agent --- #
root_agent = LlmAgent(
    name="syllabus_planner_agent",
    model=GEMINI_PRO_MODEL,
    description="Calendar-based syllabus planner for NCERT classes",
    instruction="""
    Step-by-step behavior:

    1. Validate the scope and dates using ScopeClarifierAgent.
       - If validated=False, return refinement_question and stop.

    2. Fetch NCERT topic list using RagAgent based on subject and standard.

    3. Map the topic list to a date-wise plan using CalendarMapperAgent.
       - Ensure balanced distribution.

    4. Ask a helpful refinement question using PlannerRefinerAgent.

    Final Output:
    - validated
    - topic_list
    - syllabus_calendar
    - refinement_question
    """,
    tools=[
        agent_tool.AgentTool(agent=scope_clarifier),
        agent_tool.AgentTool(agent=rag_topic_fetcher),
        agent_tool.AgentTool(agent=calendar_mapper),
        agent_tool.AgentTool(agent=planner_refiner)
    ]
)
