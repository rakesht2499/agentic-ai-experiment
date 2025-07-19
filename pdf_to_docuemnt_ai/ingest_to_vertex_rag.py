import vertexai
from vertexai.preview import rag
from google.cloud import aiplatform
import uuid
import json
import os
from dotenv import load_dotenv
import tempfile

def convert_to_structured_files(structured_docs, temp_dir):
    """Convert JSON documents to individual JSON files for ingestion"""
    file_paths = []
    
    for i, doc in enumerate(structured_docs):
        # Create a JSON file for each document to preserve structure
        safe_id = doc['id'].replace('/', '_').replace(' ', '_')
        file_path = os.path.join(temp_dir, f"{safe_id}.json")
        
        # Create individual document JSON with enhanced structure
        enhanced_doc = {
            "document_id": doc['id'],
            "title": doc.get('title', 'Untitled'),
            "content": doc['content'],
            "source": doc.get('source', ''),
            "metadata": {
                "chapter": doc.get('metadata', {}).get('chapter', ''),
                "subject": doc.get('metadata', {}).get('subject', ''),
                "board": doc.get('metadata', {}).get('board', ''),
                "class": doc.get('metadata', {}).get('class', ''),
                "document_type": "textbook_content",
                "content_length": len(doc['content']),
                "chunk_index": i
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_doc, f, indent=2, ensure_ascii=False)
        
        file_paths.append(file_path)
        print(f"📄 Created JSON: {os.path.basename(file_path)}")
    
    return file_paths

if __name__ == "__main__":
    load_dotenv()
    
    # === CONFIG ===
    PROJECT_ID = "rag-engine-vertex-ai-project"
    REGION = "us-central1"  # or your GCP region
    CORPUS_DISPLAY_NAME = "CBSE Class 10 Science Textbook"  # Human readable name
    INPUT_JSON_PATH = "converted_vertex_rag_docs.json"

    # === INIT VERTEX ===
    vertexai.init(project=PROJECT_ID, location=REGION)

    print(f"🔧 Project: {PROJECT_ID}")
    print(f"🌍 Region: {REGION}")
    print(f"📚 Corpus: {CORPUS_DISPLAY_NAME}")
    
    # === LOAD DATA ===
    print(f"📖 Loading documents from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, "r", encoding='utf-8') as f:
        structured_docs = json.load(f)
    
    print(f"📊 Found {len(structured_docs)} documents")

    try:
        # === CREATE OR GET CORPUS ===
        print("\n🏗️ Creating/getting RAG corpus...")
        try:
            # Try to create a new corpus
            corpus = rag.create_corpus(
                display_name=CORPUS_DISPLAY_NAME,
                description="CBSE Class 10 Science textbook content extracted from PDFs"
            )
            print(f"✅ Created new corpus: {corpus.name}")
        except Exception as e:
            # If corpus already exists, list and find it
            print(f"ℹ️ Corpus may already exist: {e}")
            corpora = rag.list_corpora()
            
            matching_corpus = None
            for c in corpora:
                if c.display_name == CORPUS_DISPLAY_NAME:
                    matching_corpus = c
                    break
            
            if matching_corpus:
                corpus = matching_corpus
                print(f"✅ Using existing corpus: {corpus.name}")
            else:
                print("❌ Could not create or find corpus")
                raise

        # === CONVERT TO STRUCTURED JSON FILES ===
        print("\n📝 Converting documents to structured JSON files...")
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = convert_to_structured_files(structured_docs, temp_dir)
            
            # === UPLOAD FILES TO GOOGLE CLOUD STORAGE FIRST ===
            print(f"\n💼 Uploading {len(file_paths)} files to Google Cloud Storage...")
            
            # Create a bucket for temporary storage
            bucket_name = f"{PROJECT_ID.replace('-', '_')}_temp_rag_files"
            
            from google.cloud import storage as gcs_storage
            gcs_client = gcs_storage.Client(project=PROJECT_ID)
            
            try:
                bucket = gcs_client.bucket(bucket_name)
                if not bucket.exists():
                    bucket = gcs_client.create_bucket(bucket_name, location=REGION)
                    print(f"💼 Created GCS bucket: {bucket_name}")
                else:
                    print(f"💼 Using existing GCS bucket: {bucket_name}")
            except Exception as e:
                print(f"❌ Error with bucket: {e}")
                # Try with a different bucket name
                bucket_name = f"temp-rag-files-{uuid.uuid4().hex[:8]}"
                bucket = gcs_client.create_bucket(bucket_name, location=REGION)
                print(f"💼 Created GCS bucket: {bucket_name}")
            
            # Upload files to GCS
            gcs_paths = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                gcs_path = f"gs://{bucket_name}/structured_docs/{filename}"
                
                blob = bucket.blob(f"structured_docs/{filename}")
                blob.upload_from_filename(file_path)
                gcs_paths.append(gcs_path)
                print(f"✅ Uploaded: {filename}")
            
            # === IMPORT FILES FROM GCS TO VERTEX RAG ===
            print(f"\n📥 Importing {len(gcs_paths)} files from GCS to Vertex RAG...")
            
            # Import files in batches to avoid timeouts
            batch_size = 10
            successful_uploads = 0
            
            for i in range(0, len(gcs_paths), batch_size):
                batch = gcs_paths[i:i+batch_size]
                print(f"📦 Importing batch {i//batch_size + 1}/{(len(gcs_paths)-1)//batch_size + 1} ({len(batch)} files)...")
                
                try:
                    response = rag.import_files(
                        corpus_name=corpus.name,
                        paths=batch,
                        chunk_size=1024,  # Adjust as needed
                        chunk_overlap=200  # Adjust as needed
                    )
                    print(f"✅ Batch imported successfully")
                    successful_uploads += len(batch)
                except Exception as e:
                    print(f"❌ Error importing batch: {e}")
                    continue
            
            print(f"\n🎉 Successfully imported {successful_uploads}/{len(gcs_paths)} files")

        print("\n🎉 Ingestion completed!")
        print(f"📋 Corpus Name: {corpus.name}")
        print(f"📊 Total Documents: {len(structured_docs)}")
        
        # === LIST FILES IN CORPUS ===
        print("\n📁 Files in corpus:")
        try:
            files = rag.list_files(corpus_name=corpus.name)
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file.display_name}")
        except Exception as e:
            print(f"ℹ️ Could not list files: {e}")
            
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
