import json
import logging
import os
import unittest
from uuid import uuid4

from openai import AzureOpenAI

from .. import Config, Maxim
from ..logger import GenerationConfig, LoggerConfig, TraceConfig
from ..tests.mock_writer import inject_mock_writer

# Use environment variables instead of hardcoded config file
logging.basicConfig(level=logging.INFO)
env = "beta"

openaiKey = os.getenv("OPENAI_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")
azureApiKey = os.getenv("AZURE_OPENAI_API_KEY")
azureEndpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

logging.basicConfig(level=logging.INFO)

class TestLoggingUsingAzureOpenAI(unittest.TestCase):

    def setUp(self) -> None:
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)

        # Skip tests if Azure credentials are not available
        if not azureApiKey or not azureEndpoint:
            self.skipTest("Azure OpenAI credentials not available")

        self.client = AzureOpenAI(
            api_version="2023-07-01-preview",
            api_key=azureApiKey,
            azure_endpoint=azureEndpoint,
        )

    def test_logging_using_azure_openai_text_completion(self):
        trace_id = str(uuid4())
        trace = self.logger.trace(
            TraceConfig(id=trace_id, name="text-completion-trace")
        )
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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Assert that we have exactly 1 add-generation log
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)

        # Assert that we have exactly 1 result log on generation
        self.mock_writer.assert_entity_action_count("generation", "result", 1)

        # Assert that we have exactly 1 trace create log
        self.mock_writer.assert_entity_action_count("trace", "create", 1)

        # Assert that we have exactly 1 trace end log
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_logging_using_azure_openai_text_completion_with_stop_sequence(self):
        trace_id = str(uuid4())
        trace = self.logger.trace(
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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Assert that we have exactly 1 add-generation log
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)

        # Assert that we have exactly 1 result log on generation
        self.mock_writer.assert_entity_action_count("generation", "result", 1)

        # Assert that we have exactly 1 trace create log
        self.mock_writer.assert_entity_action_count("trace", "create", 1)

        # Assert that we have exactly 1 trace end log
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_logging_using_azure_openai_with_tool_call(self):
        trace_id = str(uuid4())
        trace = self.logger.trace(TraceConfig(id=trace_id, name="tool-call-trace"))
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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Assert that we have exactly 1 add-generation log
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)

        # Assert that we have exactly 1 result log on generation
        self.mock_writer.assert_entity_action_count("generation", "result", 1)

        # Assert that we have exactly 1 trace create log
        self.mock_writer.assert_entity_action_count("trace", "create", 1)

        # Assert that we have exactly 1 trace end log
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def tearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()
        self.maxim.cleanup()
        return super().tearDown()
