import json
import logging
import os
import unittest
from uuid import uuid4

from openai import AzureOpenAI

from .. import Config, Maxim
from ..logger import GenerationConfig, LoggerConfig, TraceConfig

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "beta"

openaiKey = data["openAIKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]

logging.basicConfig(level=logging.INFO)

class TestLoggingUsingAzureOpenAI(unittest.TestCase):

    def setUp(self) -> None:
        config = Config(api_key=apiKey, base_url=baseUrl, debug=True)
        self.client = AzureOpenAI(
            api_version="2023-07-01-preview",
            api_key="8fdb3849a79d4bb99c907d42aa8e36e7",
            azure_endpoint="https://finetunemaxim.openai.azure.com/",
        )
        self.maxim = Maxim(config)

    def test_logging_using_azure_openai_text_completion(self):
        logger = self.maxim.logger(LoggerConfig(id=repoId))
        trace_id = str(uuid4())
        trace = logger.trace(TraceConfig(id=trace_id, name="text-completion-trace"))
        generation = trace.generation(
            GenerationConfig(
                id=str(uuid4()),
                model="text-davinci-002",
                provider="azure",
                model_parameters={"temperature": 0.7, "max_tokens": 100},
            )
        )

        completion = self.client.completions.create(
            model="text-davinci-002",
            prompt="Translate the following English text to French: 'Hello, how are you?'",
            max_tokens=100,
            temperature=0.7,
        )

        print(completion)
        generation.result(completion)
        trace.end()

    def test_logging_using_azure_openai_text_completion_with_stop_sequence(self):
        logger = self.maxim.logger(LoggerConfig(id=repoId))
        trace_id = str(uuid4())
        trace = logger.trace(
            TraceConfig(id=trace_id, name="text-completion-stop-sequence-trace")
        )
        generation = trace.generation(
            GenerationConfig(
                id=str(uuid4()),
                model="text-davinci-002",
                provider="azure",
                model_parameters={
                    "temperature": 0.5,
                    "max_tokens": 50,
                    "stop": [".", "\n"],
                },
            )
        )

        completion = self.client.completions.create(
            model="text-davinci-002",
            prompt="Write a short sentence about artificial intelligence",
            max_tokens=50,
            temperature=0.5,
            stop=[".", "\n"],
        )

        print(completion)
        generation.result(completion)
        trace.end()

    def test_logging_using_azure_openai_with_tool_call(self):
        logger = self.maxim.logger(LoggerConfig(id=repoId))
        trace_id = str(uuid4())
        trace = logger.trace(TraceConfig(id=trace_id, name="tool-call-trace"))
        generation = trace.generation(
            GenerationConfig(
                id=str(uuid4()),
                model="gpt-4o",
                provider="azure",
                messages=[
                    {"role": "user", "content": "What's the weather in New York?"}
                ],
                model_parameters={"temperature": 0.7},
            )
        )
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in New York?",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. New York, NY",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )
        print(completion)
        generation.result(completion)
        trace.end()

    def tearDown(self) -> None:
        self.maxim.cleanup()
        return super().tearDown()
