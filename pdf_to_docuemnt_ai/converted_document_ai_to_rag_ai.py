import json
import os
import uuid

INPUT_FOLDER = "output"  # local folder after download
OUTPUT_FILE = "converted_vertex_rag_docs.json"

def extract_title(text):
    # Very naive title extractor — improve later with regex if needed
    lines = text.split("\n")
    for line in lines:
        if "chapter" in line.lower():
            return line.strip().title()
    return "Untitled Chapter"

def split_text(text, max_chars=2000):
    # Split raw text into chunks of ~2000 characters (RAG best practices)
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

def convert_docai_to_rag_format(input_folder):
    rag_docs = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                try:
                    doc = json.load(f)
                    full_text = doc.get("text", "")
                    chapter_title = extract_title(full_text)
                    chapter_number = chapter_title.split(" ")[1] if " " in chapter_title else "Unknown"

                    content_chunks = split_text(full_text)
                    for i, chunk in enumerate(content_chunks):
                        rag_docs.append({
                            "id": f"cbse_class10_science_ch{chapter_number}_chunk{i+1}",
                            "source": "CBSE_Class10_Science",
                            "title": chapter_title,
                            "content": chunk,
                            "metadata": {
                                "chapter": f"Chapter {chapter_number}",
                                "subject": "Science",
                                "board": "CBSE",
                                "class": "10"
                            }
                        })
                except Exception as e:
                    print(f"❌ Failed to process {file}: {e}")

    return rag_docs

if __name__ == "__main__":
    print("🚧 Converting raw DocAI JSON to RAG format...")
    documents = convert_docai_to_rag_format(INPUT_FOLDER)
    with open(OUTPUT_FILE, "w") as out_file:
        json.dump(documents, out_file, indent=2)
    print(f"✅ Converted {len(documents)} chunks and saved to {OUTPUT_FILE}")
