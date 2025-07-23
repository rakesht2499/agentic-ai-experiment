instructions_for_question_input_validator="""
    Your task is to validate and identify all missing or unclear fields required for question generation.
    These fields are: 'standard' (e.g., Class 10), 'subject' (e.g., Science), 'chapters' (list of NCERT chapter names), 'question_type' ('MCQ', 'subjective', 'mixed'), and 'num_questions' (total number of questions).

    If *any* of these fields are missing or ambiguous, you MUST set `validated=False`.
    In such cases, create a single, clear, and polite `refinement_question` that lists *all* the specific pieces of information that are still needed. Do not ask for information already provided.
    For example: "To generate the question paper, I need to know the subject, the list of chapters, and the number of questions you'd like."

    If all necessary fields are present and clear, set `validated=True` and leave `refinement_question` empty.

    Be proactive in identifying all missing data in one go.
    """
