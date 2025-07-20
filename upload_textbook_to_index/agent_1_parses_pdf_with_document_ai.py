from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import os
from dotenv import load_dotenv

# ⚙️ CONFIGURATION
PROJECT_ID = "rag-engine-vertex-ai-project"
LOCATION = "us-east1"  # or your processor’s region
PROCESSOR_ID = "dcb5c77d0e43d45b"
INPUT_GCS_URI = "gs://shahayak-agentic-ai-gpl-muskeeters/pdf/cbse/class10/english"
OUTPUT_GCS_URI = "gs://shahayak-agentic-ai-gpl-muskeeters/json/cbse/class10/english"

# Initialize clients
storage_client = storage.Client(project=PROJECT_ID)
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-documentai.googleapis.com"}
)

processor_name = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

def batch_process():
    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_documents=documentai.BatchDocumentsInputConfig(
            gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=INPUT_GCS_URI)
        ),
        document_output_config=documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=OUTPUT_GCS_URI
            )
        ),
        skip_human_review=True,
    )
    operation = docai_client.batch_process_documents(request)
    operation.result(timeout=300)
    print("✅ Document AI processing complete.")

def download_output():
    _, prefix = OUTPUT_GCS_URI.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(OUTPUT_GCS_URI.split("/", 3)[2])
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith(".json"):
            local_path = os.path.join("output", os.path.basename(blob.name))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print("📥 Downloaded", local_path)

def test_connection():
    """Test if we can connect to Document AI and list processors"""
    # Common Document AI locations to try
    locations_to_try = ["us", "us-central1", "us-east1", "us-west1", "eu", "asia"]
    
    print(f"🔍 Testing connection to Document AI...")
    print(f"📋 Project ID: {PROJECT_ID}")
    print(f"🔧 Processor ID: {PROCESSOR_ID}")
    
    for location in locations_to_try:
        try:
            print(f"\n🌍 Trying location: {location}")
            
            # Create client for this location
            temp_client = documentai.DocumentProcessorServiceClient(
                client_options={"api_endpoint": f"{location}-documentai.googleapis.com"}
            )
            
            # Try to list processors in this location
            parent = temp_client.common_location_path(PROJECT_ID, location)
            processors = temp_client.list_processors(parent=parent)
            
            processors_list = list(processors)
            if processors_list:
                print(f"✅ Found {len(processors_list)} processor(s) in {location}:")
                for processor in processors_list:
                    print(f"  - ID: {processor.name.split('/')[-1]}")
                    print(f"    Name: {processor.display_name}")
                    print(f"    Type: {processor.type_}")
                    print(f"    Full path: {processor.name}")
                    print()
                
                # Check if our specific processor is in this location
                temp_processor_name = temp_client.processor_path(PROJECT_ID, location, PROCESSOR_ID)
                try:
                    processor = temp_client.get_processor(name=temp_processor_name)
                    print(f"🎯 Found our target processor in {location}!")
                    print(f"   Name: {processor.display_name}")
                    print(f"   Type: {processor.type_}")
                    
                    # Update global variables
                    global LOCATION, docai_client, processor_name
                    LOCATION = location
                    docai_client = temp_client
                    processor_name = temp_processor_name
                    return True
                except Exception:
                    print(f"⚠️ Our target processor {PROCESSOR_ID} not found in {location}")
            else:
                print(f"ℹ️ No processors found in {location}")
                
        except Exception as e:
            print(f"❌ Could not connect to {location}: {e}")
            continue
    
    print("\n❌ Could not find the processor in any location")
    return False

if __name__ == "__main__":
    load_dotenv()
    
    # Test connection first
    if test_connection():
        print("\n🚀 Starting batch processing...")
        batch_process()
        download_output()
    else:
        print("\n⚠️ Please check your processor configuration and try again.")
