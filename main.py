import os

import vertexai
from dotenv import load_dotenv
from vertexai import agent_engines

from request_processor_agent.agent import root_agent

if __name__ == "__main__":
    load_dotenv()

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")

    agent = agent_engines.get("projects/rag-engine-vertex-ai-project/locations/us-central1/reasoningEngines/8302482670080753664")
    print(agent.operation_schemas())
    print(agent.stream_query(input="I am a teacher, can you explain heredity from class 10 scince?"))
    #
    # try:
    #     nltk.data.find("tokenizers/punkt")
    # except LookupError:
    #     nltk.download("punkt")
    #
    # vertexai.init(
    #     project=project_id,
    #     location=location,
    #     staging_bucket=bucket,
    # )
    #
    # remote_app = agent_engines.create(
    #     display_name="Shahayak 2.0 - test 1",
    #     agent_engine=root_agent,
    #     requirements=[
    #         "google-cloud-aiplatform[adk,agent_engines]==1.104.0",
    #         "google-adk==1.7.0",
    #         "python-dotenv==1.1.0",
    #         "requests>=2.32.4",
    #         "pydantic==2.11.3",
    #         "absl-py==2.1.0",
    #         "cloudpickle==3.0.0",
    #         "deprecated>=1.2.14",
    #     ],
    #     extra_packages=[
    #         "./request_processor_agent",
    #         "./answer_orchastrator_agent",
    #         "./diagram_generating_agent",
    #         "./image_generating_agent",
    #         "./lesson_planning_agent",
    #         "./quiz_generating_agent_new",
    #         "./common_agents",
    #         "./syllabus_planning_agent",
    #         "./models"
    #     ],
    # )