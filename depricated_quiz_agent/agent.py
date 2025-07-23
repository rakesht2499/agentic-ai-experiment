import os
from typing import List
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from models.constants import GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

from depricated_quiz_agent.prompts import instructions_for_question_input_validator

# --- Input and Output Schemas --- #
class QuestionGenerationInput(BaseModel):
    standard: str = Field(..., description="e.g., '10' for Class 10")
    subject: str = Field(..., description="e.g., 'Science'")
    chapters: List[str] = Field(..., description="List of chapter names from NCERT")
    question_type: str = Field(..., description="'MCQ', 'subjective', or 'mixed'")
    num_questions: int = Field(..., description="Total number of questions to generate")

class QuestionGenerationOutput(BaseModel):
    validated: bool = Field(..., description="Whether input was complete and valid")
    aligned_chapters: List[str] = Field(..., description="Chapters that match NCERT")
    generated_questions: str = Field(..., description="Generated questions only (no answers or solutions)")
    refinement_question: str = Field(..., description="Follow-up query to clarify or improve")

# --- Sub-Agents --- #

# 1. Input Validator Agent
question_input_validator = LlmAgent(
    name="QuestionInputValidatorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction=instructions_for_question_input_validator,
    output_schema=QuestionGenerationOutput
)

# 2. NCERT Chapter Aligner Agent
chapter_aligner = LlmAgent(
    name="ChapterAlignerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Cross-check the provided chapters against the official NCERT curriculum for the given subject and class.

    Return:
    - aligned_chapters: A filtered list containing only valid NCERT chapter names (maintain order).

    Do NOT include any explanation or additional commentary.
    """,
    output_schema=QuestionGenerationOutput
)

# 3. Core Question Generator Agent (non-repeating, high-quality)
core_question_generator = LlmAgent(
    name="CoreQuestionGeneratorAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Generate ONLY high-quality exam-style questions (NO answers, NO hints, NO solutions).

    Requirements:
    - Use 'question_type': 'MCQ', 'subjective', or 'mixed' appropriately.
    - Spread across Bloom’s taxonomy (recall, apply, analyze).
    - Ensure questions are NOT repeated or semantically duplicated.
    - No extra commentary.
    - Follow this format strictly:
        Q1. ...
        Q2. ...
        ...
    - Use age-appropriate tone for the specified class.
    - Total questions must equal 'num_questions'.

    Output must be ONLY the questions block.
    """,
    output_schema=QuestionGenerationOutput
)

# 4. Follow-up Refiner Agent
question_refiner = LlmAgent(
    name="QuestionRefinerAgent",
    model=GEMINI_FLASH_MODEL,
    instruction="""
    Suggest ONE relevant follow-up question to improve or personalize the question paper.

    Examples:
    - Would you like to specify difficulty levels?
    - Do you want marking scheme or CBSE pattern format?

    Only populate `refinement_question`. Leave other fields empty.
    """,
    output_schema=QuestionGenerationOutput
)

# --- Root Orchestrator Agent --- #
root_agent = LlmAgent(
    name="exam_generator_agent",
    model=GEMINI_PRO_MODEL,
    description="Exam question paper generator using NCERT-aligned chapters",
    instruction="""
    Step-by-step behavior:

    1. Invoke QuestionInputValidatorAgent.
       - If validated=False, return refinement_question and stop.

    2. If valid, run ChapterAlignerAgent to ensure chapters align with NCERT.

    3. Use CoreQuestionGeneratorAgent to generate ONLY exam questions.
       - Ensure no repetitions or semantic duplicates.
       - Respect count and question_type strictly.
       - Output format: Q1..., Q2..., no answers, no extra notes.

    4. Invoke QuestionRefinerAgent to provide a helpful follow-up suggestion.

    Final Output:
    - validated
    - aligned_chapters
    - generated_questions
    - refinement_question
    """,
    tools=[
        agent_tool.AgentTool(agent=question_input_validator),
        agent_tool.AgentTool(agent=chapter_aligner),
        agent_tool.AgentTool(agent=core_question_generator),
        agent_tool.AgentTool(agent=question_refiner)
    ],
)