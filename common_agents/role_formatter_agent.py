from abc import ABC, abstractmethod
from typing import Literal, override, List

from google.adk.agents import InvocationContext, LlmAgent
from google.adk.tools import BaseTool, ToolContext
from pydantic import BaseModel, Field

from models.constants import GEMINI_PRO_MODEL

class RoleFormatterInput(BaseModel):
    role: Literal["teacher", "student", "parent"] = Field(description="The role of the user")
    content: str = Field(description="The content to be formatted.")
    formatter_type: Literal["answer", "quiz"] = Field(description="The decider whether to use qna formatter or quiz formatter.")

class RoleFormatterOutput(BaseModel):
    formatter_content: str = Field(description="The formatted content to be formatted.")
    formatter_type: Literal["answer", "quiz"] = Field(description="The decider whether to use qna formatter or quiz formatter.")
    error_logs: List[str] = Field(description="A list of error logs to show in case of failure.")


class BaseContentFormatter(ABC):
    """Abstract base class for content-specific formatters"""
    
    @abstractmethod
    def format_for_teacher(self, content: str) -> str:
        pass
    
    @abstractmethod 
    def format_for_student(self, content: str) -> str:
        pass
        
    @abstractmethod
    def format_for_parent(self, content: str) -> str:
        pass

class AnswerFormatter(BaseContentFormatter):
    """Handles Answer content formatting using the exact prompts from answer_orchestrator_agent"""
    
    def __init__(self):
        # Create LLM agents for each role with exact prompts from reference
        self.teacher_agent = LlmAgent(
            name="AnswerTeacherFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
    You are an expert teacher assistant helping a teacher explain a textbook concept to students in class.
    
    Structure:
    1. Begin with a teacher-friendly opener like: "Here's how you can teach this:"
    2. Explain the concept in clear, simple, and structured steps.
    3. Add a relatable analogy using classroom or real-world objects.
    4. Suggest at least ONE of the following:
       - A chalkboard diagram idea
       - A simple classroom activity
       - A quick discussion question
    
    Constraints:
    - Keep tone professional but warm (not robotic).
    - Do not invent information. Stick to textbook facts.
    
    Wrap with:
    - ✅ A 1-line recap
    - 📍 "Based on Chapter <chapter name>"
    """
        )
        
        self.parent_agent = LlmAgent(
            name="AnswerParentFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
You are a supportive helper guiding a parent in explaining a textbook concept to their child.

Tone & Style:
- Use a kind, encouraging, empathetic tone.
- Assume the parent may not remember school concepts.
- Never use complex or academic language.

Structure:
1. Start with reassurance: "No worries! Here's how you can explain it at home:"
2. Simplify the concept in everyday words.
3. Use an example from daily life (cooking, walking, shopping, home chores).
4. Gently offer to translate: "Would you like this in your local language?"

Wrap with:
- ✅ "You did a great job explaining this!"
- Mention TranslatorAgent only if language ≠ English.
"""
        )
        
        self.student_agent = LlmAgent(
            name="AnswerStudentFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
You are a cheerful, encouraging tutor helping a student understand a textbook concept.

Tone:
- Friendly, fun, clear — like a favorite older sibling or coach.
- Encourage the student and make them feel confident.

Structure:
1. Begin with: "Let's learn this together!"
2. Explain in 2–3 steps using short, simple sentences.
3. Add a memory hook using a relatable object: "Think of it like a sponge..."
4. End with a quick quiz or check-in: 
   - "Want to try a quick question?"
   - "What do you think happens next?"

Wrap with:
- ✨ The memory hook
- 📘 "Based on Chapter <chapter>"
"""
        )
    
    def format_for_teacher(self, content: str) -> str:
        response = self.teacher_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content
    
    def format_for_student(self, content: str) -> str:
        response = self.student_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content
        
    def format_for_parent(self, content: str) -> str:
        response = self.parent_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content


class QuizFormatter(BaseContentFormatter):
    """Handles Quiz/Exam content formatting using the exact prompts from exam_generating_agent_new"""
    
    def __init__(self):
        # Create LLM agents for each role with exact prompts from reference
        self.teacher_agent = LlmAgent(
            name="QuizTeacherFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
You are formatting a quiz or exam for a teacher to print or assign in class.

1. Present questions in a clean numbered list.
2. Maintain proper structure with:
   - ✅ Marks per question
   - ✅ Section headers (if exam)
   - ✅ Blank space for answers
3. Do not simplify or answer the questions.
4. Preserve subject-specific terminology.

At the end, add:
📘 For: Class {{ input.class_ }}, Subject: {{ input.subject }}, Chapters: {{ chapter_list }}
"""
        )
        
        self.parent_agent = LlmAgent(
            name="QuizParentFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
You are helping a parent revise a quiz or exam with their child at home.

1. Reword questions in simple, parent-friendly language.
2. Add short encouraging tips after each question like:
   - "You've got this!"
   - "Let's recall what we saw in the diagram!"
3. Keep it warm and easy to read. Use everyday vocabulary.
4. Do not answer the questions.

At the end:
- ✅ "You did a great job guiding your child!"
- 📘 Based on: Class {{ input.class_ }}, Subject: {{ input.subject }}
"""
        )
        
        self.student_agent = LlmAgent(
            name="QuizStudentFormatter",
            model=GEMINI_PRO_MODEL,
            instruction="""
    You are a quiz master giving an interactive quiz directly to a student.
    
    🧠 Here's how you behave:
    - Present **only one** question at a time.
    - After each question, wait for the student's answer before sending the next.
    - Use friendly, encouraging language.
    - After 2–3 questions, check in:
       - "Need a break?"
       - "Want a hint?"
       - "You're doing amazing, keep going! ⚡"
    
    🤖 Important:
    - Do **not** give answers.
    - Keep track of which questions have been asked.
    - Never repeat a question.
    - At the end, say:
      ✅ "Great work!"
      📘 "Questions from Class {{ input.class_ }}, Subject: {{ input.subject }}, Chapters: {{ input.chapter_list }}"
    
    🎯 Example Interaction Flow:
    1. "Ready to test your knowledge on Heredity? Let's go! 🤓"
    2. "Q1: What is the term for the study of how traits are passed from one generation to the next?"
    ⏸️ *[Wait for response]*
    3. "Awesome! Let's try another one. ⚡"
    4. "Q2: When Gregor Mendel crossed a pure tall and pure dwarf pea plant..."
    
    Stay in this loop until all questions are asked.
    """
        )
    
    def format_for_teacher(self, content: str) -> str:
        response = self.teacher_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content
    
    def format_for_student(self, content: str) -> str:
        response = self.student_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content
        
    def format_for_parent(self, content: str) -> str:
        response = self.parent_agent.run(content)
        return response.content.parts[0].text if response.content and response.content.parts else content


class RoleFormatterAgent(BaseTool):
    """
    Shared agent for role-based content formatting using Strategy Pattern.
    Supports different content types with role-specific formatting.
    """
    
    def __init__(self):
        super().__init__(
            name="role_formatter_agent",
            description="Formats content based on user role and content type using Strategy Pattern"
        )
        
        # Initialize formatters using Strategy Pattern
        self.formatters = {
            "answer": AnswerFormatter(),
            "quiz": QuizFormatter(),
        }
    
    @override
    async def run_async(self, context: InvocationContext, tool_context: ToolContext) -> RoleFormatterOutput:
        input_data = RoleFormatterInput(**context.input.dict())
        # role = input_data["role"]
        content = input_data["content"]
        formatter_type = input_data["formatter_type"]
        role = context.session.state.get("role", "unknown")
        
        # Log the formatting operation
        logs = [f"Formatting {formatter_type} content for role: {role}"]
        
        try:
            # Get the appropriate formatter
            formatter = self.formatters.get(formatter_type)
            if not formatter:
                logs.append(f"Unknown formatter_type '{formatter_type}', using default formatting")
                return RoleFormatterOutput(formatter_content=content, formatter_type=formatter_type, error_logs=logs)
            
            # Get the role-specific method
            method_name = f"format_for_{role}"
            format_method = getattr(formatter, method_name, None)
            
            if not format_method:
                logs.append(f"Unknown role '{role}', using default formatting")
                return RoleFormatterOutput(formatter_content=content, formatter_type=formatter_type, error_logs=logs)
            
            # Apply formatting
            formatted_content = format_method(content)
            logs.append(f"Content formatted successfully for {role} using {formatter_type} formatter")

            return RoleFormatterOutput(formatter_content=formatted_content, formatter_type=formatter_type, error_logs=[])
        except Exception as e:
            logs.append(f"Error during formatting: {str(e)}")
            return RoleFormatterOutput(formatter_content=content, formatter_type=formatter_type, error_logs=logs)

# Create the singleton instance for reuse
role_formatter_agent = RoleFormatterAgent()


# Usage example for orchestrators:
"""
# In AnswerOrchestratorAgent or QuizPrepOrchestratorAgent:

from common_agents.role_formatter_agent import role_formatter_agent

# Add to orchestrator's tools list:
tools=[
    # ... other tools
    role_formatter_agent,
]

# Usage examples:
# For Answer content:
# {"role": "teacher", "content": "RAG answer...", "content_type": "answer"}

# For Quiz content:
# {"role": "student", "content": "Generated quiz...", "content_type": "quiz"}

# Easy to extend with new content types:
# Just add new formatter classes that implement BaseContentFormatter
""" 