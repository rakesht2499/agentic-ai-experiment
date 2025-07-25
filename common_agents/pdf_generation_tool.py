from fpdf import FPDF
import os


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, self.title, ln=1, align="C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.ln(10)
        self.cell(0, 10, title, ln=1)

    def chapter_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, text)
        self.ln()

    def add_question_list(self, questions):
        self.set_font("Arial", "", 11)
        for i, q in enumerate(questions, start=1):
            self.multi_cell(0, 8, f"{i}. {q['question']}")
            self.ln(1)

    def add_answer_list(self, questions):
        self.set_font("Arial", "", 11)
        for i, q in enumerate(questions, start=1):
            self.multi_cell(0, 8, f"{i}. {q['answer']}")
            self.ln(1)


def generate_pdf_with_answers(questions, class_, subject, mode, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{subject.lower()}_{class_.lower().replace(' ', '')}_{mode.lower()}.pdf"
    output_path = os.path.join(output_dir, filename)

    pdf = PDF()
    pdf.set_title("Quiz / Exam Paper")
    pdf.add_page()

    # Page 1: Questions
    pdf.chapter_title(f"{subject} - {class_} - {mode.capitalize()}")
    pdf.chapter_title("Quiz Questions")
    pdf.add_question_list(questions)

    # Page 2: Answer Key
    pdf.add_page()
    pdf.chapter_title("Answer Key")
    pdf.add_answer_list(questions)

    pdf.output(output_path)

    return output_path
