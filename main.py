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
    #
    # agent = agent_engines.get("projects/rag-engine-vertex-ai-project/locations/us-central1/reasoningEngines/8302482670080753664")
    # print(agent.operation_schemas())
    # print(agent.stream_query(input="I am a teacher, can you explain heredity from class 10 scince?"))
    #
    # try:
    #     nltk.data.find("tokenizers/punkt")
    # except LookupError:
    #     nltk.download("punkt")
    #
    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )

    remote_app = agent_engines.create(
        display_name="Shahayak 2.0",
        agent_engine=root_agent,
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]==1.104.0",
            "google-adk==1.7.0",
            "python-dotenv==1.1.0",
            "requests>=2.32.4",
            "pydantic==2.11.3",
            "absl-py==2.1.0",
            "cloudpickle==3.0.0",
            "deprecated>=1.2.14",
            "networkx >= 3.0",
            "Pillow >= 10.0.0"
        ],
        extra_packages=[
            "./answer_orchastrator_agent",
            "./common_agents",
            "./diagram_generating_agent",
            "./image_generating_agent",
            "./kg_quiz_generating_agent_new",
            "./knowledge_graph",
            "./lesson_planning_agent",
            "./models",
            "./quiz_generating_agent_new",
            "./request_processor_agent",
            "./syllabus_planning_agent",
        ],
    )