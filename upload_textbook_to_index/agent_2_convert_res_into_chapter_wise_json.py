import json
import os
import re
import tempfile
from google.cloud import storage

INPUT_FOLDER = "output"  # local folder after download
LOCAL_OUTPUT_FILE = "class10_english.jsonl"  # Changed to .jsonl
GCS_OUTPUT_URI = f"gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/english/{LOCAL_OUTPUT_FILE}"  # Changed to .jsonl
PROJECT_ID = "rag-engine-vertex-ai-project"


def extract_title_and_chapter(text, filename):
    # Try to extract chapter number and title from text using regex
    chapter_pattern = re.compile(r"chapter\s*(\d+)([^\n]*)", re.IGNORECASE)
    match = chapter_pattern.search(text)
    if match:
        chapter_number = match.group(1).strip()
        rest = match.group(2).strip()
        title = f"Chapter {chapter_number} - {rest}" if rest else f"Chapter {chapter_number}"
        return title.strip(), chapter_number

    # Fallback: try to extract chapter number from filename
    file_name_clean = os.path.basename(filename).lower()
    match_from_file = re.search(r"chapter[\-_\s]?(\d+)", file_name_clean)
    if match_from_file:
        chapter_number = match_from_file.group(1).strip()
        title = f"Chapter {chapter_number}"
        return title, chapter_number

    return "Untitled Chapter", "Unknown"


def split_text(text, max_chars=2000):
    # Split raw text into chunks of ~2000 characters (RAG best practices)
    chunks = []
    current = ""
    # Ensure no empty lines are processed as valid content
    lines = [line for line in text.split("\n") if line.strip()]
    for line in lines:
        # Add 1 for the newline character that will be added back
        if len(current) + len(line) + 1 < max_chars:
            current += line + "\n"
        else:
            if current.strip():  # Only add non-empty chunks
                chunks.append(current.strip())
            current = line + "\n"
    if current.strip():  # Add the last non-empty chunk
        chunks.append(current.strip())
    return chunks


def convert_docai_to_rag_format(input_folder, output_file, temp_content_dir):
    """
    Converts DocAI JSON outputs into Vertex AI RAG Engine compatible JSONL format.
    Each text chunk is saved as a separate .txt file, and the JSONL file
    references these text files via a 'uri' field.
    """
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

                        # Skip if text is empty to avoid creating empty chunks
                        if not full_text.strip():
                            print(f"Skipping empty document: {file}")
                            continue

                        chapter_title, chapter_number = extract_title_and_chapter(full_text, file)

                        content_chunks = split_text(full_text)

                        for i, chunk in enumerate(content_chunks):
                            # Create a unique ID for the chunk
                            chunk_id = f"cbse_class10_english_ch{chapter_number}_chunk{i + 1}"
                            content_filename = f"{chunk_id}.txt"
                            local_content_path = os.path.join(temp_content_dir, content_filename)

                            # Save the chunk to a local .txt file
                            with open(local_content_path, "w", encoding="utf-8") as content_f:
                                content_f.write(chunk)

                            rag_doc = {
                                "id": f"cbse_class10_english_ch{chapter_number}_chunk{i+1}",
                                "source": "CBSE_Class10_English",
                                "title": chapter_title,
                                "content": chunk,
                                "metadata": {
                                    "chapter": f"Chapter {chapter_number}",
                                    "subject": "English",
                                    "board": "CBSE",
                                    "class": "10"
                                }
                            }
                            out_f.write(json.dumps(rag_doc) + "\n")
                            processed_chunks_count += 1
                    except Exception as e:
                        print(f"❌ Failed to process {file}: {e}")
    return processed_chunks_count


def upload_to_gcs(local_file, gcs_uri):
    """Upload the processed JSONL file and content .txt files to GCS"""
    try:
        # Parse GCS URI for the main JSONL file
        bucket_name = gcs_uri.split("/")[2]
        blob_path_jsonl = "/".join(gcs_uri.split("/")[3:])

        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)

        # 1. Upload the main JSONL file
        blob_jsonl = bucket.blob(blob_path_jsonl)
        blob_jsonl.upload_from_filename(local_file)
        print(f"✅ Uploaded {local_file} to {gcs_uri}")

        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}")
        return False


if __name__ == "__main__":
    print("🚧 Converting raw DocAI JSON to Vertex AI RAG Engine format...")

    # Create a temporary directory for content chunks
    with tempfile.TemporaryDirectory() as temp_content_dir:
        print(f"Created temporary directory for content chunks: {temp_content_dir}")

        # Convert and save locally
        num_chunks = convert_docai_to_rag_format(INPUT_FOLDER, LOCAL_OUTPUT_FILE, temp_content_dir)

        print(f"✅ Converted {num_chunks} chunks and saved locally to {LOCAL_OUTPUT_FILE}")

        # Upload to GCS with the filename Agent 3 expects
        print(f"📤 Uploading JSONL and content files to GCS...")
        if upload_to_gcs(LOCAL_OUTPUT_FILE, GCS_OUTPUT_URI):
            print(f"🎉 File ready for Agent 3 at: {GCS_OUTPUT_URI}")
            print(f"Content chunks are located under: {os.path.dirname(GCS_OUTPUT_URI)}/content_chunks/")
        else:
            print(
                f"⚠️ Upload failed. You'll need to manually upload {LOCAL_OUTPUT_FILE} and the contents of {temp_content_dir} to GCS.")

