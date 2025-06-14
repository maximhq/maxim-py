"""Example cookbook demonstrating Maxim tracing with an Agno agent."""

from __future__ import annotations

from maxim.logger import Logger
from maxim.logger.agno import MaximAgnoClient

# Agno imports
from agno.agent.agent import Agent
from agno.models.ollama import Ollama


# Initialize Maxim logger (replace placeholders with your Maxim credentials)
logger = Logger(
    {"id": "YOUR_LOG_REPO_ID"},
    api_key="YOUR_MAXIM_API_KEY",
    base_url="https://app.getmaxim.ai",
)

# Enable tracing for Agno
MaximAgnoClient(logger)

# Create an Agno agent using any supported model
agent = Agent(model=Ollama())

# Run the agent. The run will automatically be logged to Maxim.
response = agent.run(message="Hello from Agno!", generation_name="hello")
print(response)
