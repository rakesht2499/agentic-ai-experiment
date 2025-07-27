import os
import uuid
from typing import List

from google.adk.tools import agent_tool
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent

from common_agents.shared_rag_agent import shared_rag_agent, clone_agent, vector_rag_agent
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

# 1) Clarifier / Input Validator
clarifier_agent = LlmAgent(
    name="ClarifierAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=input_validator_prompt,
)

# 2) Planner composer
planner_composer_agent = LlmAgent(
    name="PlannerComposerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=planner_composer_prompt,
)

# 3) Refiner
refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=refiner_prompt,
)

# 4) Final formatter
formatter_agent = LlmAgent(
    name="LessonPlanFormatterAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=lesson_formatter_prompt,
)

# --- Processing chain ---
processing_agent = SequentialAgent(
    name="ProcessingAgent",
    description="Interactive lesson planning agent with NCERT alignment and multimodal guidance",
    sub_agents=[
        clone_agent(vector_rag_agent, "lesson_plan"),
        planner_composer_agent,
        refiner_agent,
        formatter_agent
    ],
)

# --- Root Orchestrator (ONE-SHOT, max one clarification, still produce best-effort plan) ---
lesson_planning_agent = LlmAgent(
    name="LessonPlanningAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
You are a **ONE-SHOT** lesson planning orchestrator. You MUST:
- Call ClarifierAgent **once** (and only once).
- If inputs are incomplete, ask **at most one** clarifying question (in your own final response), but STILL generate and show a **best-effort plan** using defaults.
- Never loop or call any tool more than once.
- Never wait for the user reply to proceed. Always show something useful now.

---

### EXECUTION PLAN (STRICT)

1) Call **ClarifierAgent** with the raw user input.

2) Read its response:
   - If `needs_clarification = true`:
     - Compose **one** short clarifying question.
     - **Still proceed** to STEP 3 with the **best possible autofilled input** using sensible defaults (see "Defaulting Rules").
   - If `needs_clarification = false`:
     - Use the clarified input as-is and proceed to STEP 3.

3) Call **ProcessingAgent** once to build the full plan.

4) **Final Output** to the user must include:
   - (If asked) A single, short **“Clarifying question (optional)”** line at the top.
   - The **final formatted plan** returned by ProcessingAgent **verbatim** (no modification, no reformatting).
   - Optionally a short footer like: “If you confirm the missing fields, I’ll refine this further.”

---

### DEFAULTING RULES (Use ONLY when missing):
- `timeframe`: "2 weeks"
- `preferred_theme`: "" (omit)
- If `chapters` are missing: assume "entire syllabus overview" and generate a scaffolded weekly plan.
- If `subject` missing: infer from chapters if obvious, else default to "Science".
- If `standard` missing: default to "Class 6".
- If any field is ambiguous, state your assumption explicitly in a short bullet list BEFORE the plan (keep it under 3 bullets).
- Do NOT ask more than one clarifying question.

---

### CRITICAL RESTRICTIONS
- ❌ NEVER call ClarifierAgent after ProcessingAgent.
- ❌ NEVER call ProcessingAgent more than once.
- ❌ NEVER modify or beautify the ProcessingAgent output.
- ❌ NEVER loop or retry.
- ✅ ALWAYS return something useful in the same response.

If you break any of these, it is considered a critical failure.

""",
    tools=[
        agent_tool.AgentTool(agent=clarifier_agent),
        agent_tool.AgentTool(agent=processing_agent)
    ]
)

root_agent = lesson_planning_agent
