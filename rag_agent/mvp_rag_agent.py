from typing import List

from google.adk.agents import LlmAgent

from models.constants import GEMINI_FLASH_MODEL
from pydantic import BaseModel, Field

rag_prompt = """
Given a list of chapters and one subject, for multiple grades (e.g., Class 3–5 or Class 3,8 and 10) or single grade (like 10th Class, 8th Standard or just 3rd), check the latest and updated NCERT syllabus.

Step-by-step instructions:

1. Match the chapters provided by the user against the official NCERT syllabus for the specified grades.
2. If some chapters do **not exist** in the syllabus for those classes:
    - Politely notify the user that the chapters are not part of the current NCERT syllabus for the mentioned grade(s).
    - Display the correct list of chapters for that subject and those classes in simple bullet points.
    - Example: "‘Carbon Cycle’ is not a listed chapter for Class 5 Science. Here is the official chapter list for Class 5 Science: ..."
3. Despite the mismatch, **do not stop** the process. Ask the user if the user still wants to explore the topic, provide a **school-friendly explanation** of that concept using simple, culturally-relevant examples.
4. Proceed with the chapters that were correctly matched for planning.
5. Deduplicate and clean the list of valid chapters.
6. Leave `lesson_plan` empty.

Only return in the output:
- `aligned_chapters`: list of valid NCERT-aligned chapters
- `refinement_question`: suggest if the user would like to choose from the official chapter list or continue with their original input
- `validated`: True
- `lesson_plan`: leave blank
"""

class LessonPlanOutput(BaseModel):
    class_name: str = Field(..., description="The class name")
    subject_name: str = Field(..., description="The subject name")
    
    chapters_name_list: List[str] = Field(..., description="NCERT-aligned and deduplicated chapter list")
    chapters_content


rag_agent = LlmAgent(
    name="rag_agent",
    model=GEMINI_FLASH_MODEL,
    instruction=rag_prompt,
    output_schema=LessonPlanOutput
)