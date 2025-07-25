import json
import os
import re
import tempfile
from typing import List

from google.adk.agents import LlmAgent
from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from models.constants import GEMINI_PRO_MODEL

INPUT_FOLDER = "output"
LOCAL_OUTPUT_FILE = "class10_science.jsonl"
GCS_OUTPUT_URI = f"gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/science/{LOCAL_OUTPUT_FILE}"
PROJECT_ID = "rag-engine-vertex-ai-project"

# ✅ NEW CUSTOM SEMANTIC SPLITTERS

def custom_segment_text(text):
    segments = re.split(r"(Activity\s+\d+\.\d+|Q U E S T I O N S|EXERCISES|Step [IVXL]+|Fig\.\s*\d+\.\d+|^\d+\.\d+.*?)\n", text, flags=re.MULTILINE)
    structured = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments):
            segment = segments[i] + "\n" + segments[i + 1]
            structured.append(segment.strip())
            i += 2
        else:
            structured.append(segments[i].strip())
            i += 1
    return structured

def split_with_overlap(text, chunk_size=2000, overlap=256):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def split_text(text):
    segments = custom_segment_text(text)
    all_chunks = []
    for segment in segments:
        chunks = split_with_overlap(segment)
        all_chunks.extend(chunks)
    return all_chunks

# ✅ UNCHANGED FROM HERE ON (EXCEPT chunking uses above split_text)

def extract_title_and_chapter(text, filename):
    chapter_pattern = re.compile(r"chapter\s*(\d+)([^\n]*)", re.IGNORECASE)
    match = chapter_pattern.search(text)
    if match:
        chapter_number = match.group(1).strip()
        rest = match.group(2).strip()
        title = f"Chapter {chapter_number} - {rest}" if rest else f"Chapter {chapter_number}"
        return title.strip(), chapter_number

    file_name_clean = os.path.basename(filename).lower()
    match_from_file = re.search(r"chapter[\-_\s]?(\d+)", file_name_clean)
    if match_from_file:
        chapter_number = match_from_file.group(1).strip()
        title = f"Chapter {chapter_number}"
        return title, chapter_number

    return "Untitled Chapter", "Unknown"


def convert_docai_to_rag_format(input_folder, output_file, temp_content_dir):
    processed_chunks_count = 0
    with open(output_file, "w") as out_f:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if not file.endswith(".json"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        doc = json.load(f)
                        full_text = doc.get("text", "")

                        if not full_text.strip():
                            print(f"Skipping empty document: {file}")
                            continue

                        chapter_title, chapter_number = extract_title_and_chapter(full_text, file)

                        content_chunks = split_text(full_text)

                        for i, chunk in enumerate(content_chunks):
                            chunk_id = f"cbse_class10_science_ch{chapter_number}_chunk{i + 1}"
                            content_filename = f"{chunk_id}.txt"
                            local_content_path = os.path.join(temp_content_dir, content_filename)

                            with open(local_content_path, "w", encoding="utf-8") as content_f:
                                content_f.write(chunk)

                            rag_doc = {
                                "id": chunk_id,
                                "source": "CBSE_Class10_science",
                                "title": chapter_title,
                                "content": chunk,
                                "metadata": {
                                    "chapter": f"Chapter {chapter_number}",
                                    "subject": "Science",
                                    "board": "CBSE",
                                    "class": "10"
                                }
                            }
                            out_f.write(json.dumps(rag_doc) + "\n")
                            processed_chunks_count += 1
                    except Exception as e:
                        print(f"❌ Failed to process {file}: {e}")
    return processed_chunks_count


def upload_to_gcs(local_file: str, gcs_uri: str) -> None:
    try:
        bucket_name = gcs_uri.split("/")[2]
        blob_path_jsonl = "/".join(gcs_uri.split("/")[3:])

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        blob_jsonl = bucket.blob(blob_path_jsonl)
        blob_jsonl.upload_from_filename(local_file)
        print(f"✅ Uploaded {local_file} to {gcs_uri}")

        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}")
        return False


if __name__ == "__main__":
    print("🚧 Converting raw DocAI JSON to Vertex AI RAG Engine format...")

    with tempfile.TemporaryDirectory() as temp_content_dir:
        print(f"Created temporary directory for content chunks: {temp_content_dir}")

        num_chunks = convert_docai_to_rag_format(INPUT_FOLDER, LOCAL_OUTPUT_FILE, temp_content_dir)

        print(f"✅ Converted {num_chunks} chunks and saved locally to {LOCAL_OUTPUT_FILE}")

        print(f"📤 Uploading JSONL and content files to GCS...")
        if upload_to_gcs(LOCAL_OUTPUT_FILE, GCS_OUTPUT_URI):
            print(f"🎉 File ready for Agent 3 at: {GCS_OUTPUT_URI}")
            print(f"Content chunks are located under: {os.path.dirname(GCS_OUTPUT_URI)}/content_chunks/")
        else:
            print(f"⚠️ Upload failed. You'll need to manually upload {LOCAL_OUTPUT_FILE} and the contents of {temp_content_dir} to GCS.")
