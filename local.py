from fastapi import FastAPI
from vertexai.preview import reasoning_engines
from shahayak_agent.agent import root_agent  # your AdkAgent definition

# Wrap your agent using AdkFastApiApp
app = reasoning_engines.AdkFastApiApp(agent=root_agent)
