import os

import nltk
import vertexai
from dotenv import load_dotenv
from vertexai import agent_engines

from shahayak_agent.agent import root_agent

if __name__ == "__main__":
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )

    remote_app = agent_engines.create(
        display_name="Shahayak 2.0 - test 1",
        agent_engine=root_agent,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]==1.104.0",
            "google-adk==1.7.0",
            "python-dotenv==1.1.0",
            "requests>=2.32.4",
            "pydantic==2.11.3",
            "absl-py==2.1.0",
            "cloudpickle==3.0.0",
            "llama-index==0.10.39",
            "deprecated>=1.2.14",
            "nltk>=3.8.1"
        ],
        extra_packages=[
            "./shahayak_agent",
            "./q_and_a_agent",
            "./diagram_generating_agent",
            "./image_generating_agent",
            "./lesson_planning_agent",
            "./rag_agent",
            "upload_textbook_to_index",
        ],
    )