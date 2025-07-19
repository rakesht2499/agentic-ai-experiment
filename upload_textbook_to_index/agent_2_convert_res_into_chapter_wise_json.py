import json
import os
import re
from google.cloud import storage

INPUT_FOLDER = "output"  # local folder after download
LOCAL_OUTPUT_FILE = "converted_vertex_rag_docs.json"
GCS_OUTPUT_URI = "gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/science/class10_science.json"
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
                    chapter_title, chapter_number = extract_title_and_chapter(full_text, file)

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

def upload_to_gcs(local_file, gcs_uri):
    """Upload the processed file to GCS"""
    try:
        # Parse GCS URI
        bucket_name = gcs_uri.split("/")[2]
        blob_path = "/".join(gcs_uri.split("/")[3:])
        
        # Initialize storage client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Upload file
        blob.upload_from_filename(local_file)
        print(f"✅ Uploaded {local_file} to {gcs_uri}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload to GCS: {e}")
        return False

if __name__ == "__main__":
    print("🚧 Converting raw DocAI JSON to RAG format...")
    documents = convert_docai_to_rag_format(INPUT_FOLDER)
    
    # Save locally first
    with open(LOCAL_OUTPUT_FILE, "w") as out_file:
        json.dump(documents, out_file, indent=2)
    print(f"✅ Converted {len(documents)} chunks and saved locally to {LOCAL_OUTPUT_FILE}")
    
    # Upload to GCS with the filename Agent 3 expects
    print(f"📤 Uploading to GCS...")
    if upload_to_gcs(LOCAL_OUTPUT_FILE, GCS_OUTPUT_URI):
        print(f"🎉 File ready for Agent 3 at: {GCS_OUTPUT_URI}")
    else:
        print(f"⚠️ Upload failed. You'll need to manually upload {LOCAL_OUTPUT_FILE} to GCS.")
