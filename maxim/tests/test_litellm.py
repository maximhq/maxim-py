import asyncio
import json
import logging
import os
import unittest
from uuid import uuid4
import time

import litellm
from litellm import acompletion, completion

from maxim import Config, Maxim
from maxim.logger import LoggerConfig, TraceConfig
from maxim.logger.litellm.tracer import MaximLiteLLMTracer
from maxim.tests.mock_writer import inject_mock_writer

logging.basicConfig(level=logging.INFO)

awsAccessKeyId = os.getenv("BEDROCK_ACCESS_KEY_ID")
awsAccessKeySecret = os.getenv("BEDROCK_SECRET_ACCESS_KEY")
azureOpenAIBaseUrl = os.getenv("AZURE_OPENAI_BASE_URL")
azureOpenAIKey = os.getenv("AZURE_OPENAI_KEY")
openAIKey = os.getenv("OPENAI_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")
anthropicApiKey = os.getenv("ANTHROPIC_API_KEY")


class TestLiteLLM(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Clear any existing LiteLLM callbacks before setting up new ones
        litellm.callbacks = []

        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)
        callback = MaximLiteLLMTracer(self.logger)
        # Clear any leftover container state
        callback.containers.clear()
        litellm.callbacks = [callback]

    def test_openai(self) -> None:
        if not openAIKey:
            self.skipTest("OpenAI API key not available")

        response = completion(
            model="openai/gpt-4o",
            api_key=openAIKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            temperature=0.7,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        )
        print(response)

        # Give LiteLLM callback time to complete
        time.sleep(0.5)

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

    def test_openai_with_external_trace(self) -> None:
        if not openAIKey:
            self.skipTest("OpenAI API key not available")

        trace_id = str(uuid4())
        trace = self.logger.trace(TraceConfig(id=trace_id, name="external_trace"))

        response = completion(
            model="openai/gpt-4o",
            api_key=openAIKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            temperature=0.7,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            metadata={"maxim": {"trace_id": trace_id, "span_name": "lite-llm call"}},
        )
        trace.end()
        print(response)

        # Give LiteLLM callback time to complete
        time.sleep(0.5)

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # With external trace, we expect 1 trace create (external) and 1 span create (litellm)
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-span", 1)

        # Assert that we have exactly 1 add-generation log (on the span)
        self.mock_writer.assert_entity_action_count("span", "add-generation", 1)

        # Assert that we have exactly 1 result log on generation
        self.mock_writer.assert_entity_action_count("generation", "result", 1)

        # Assert that we have exactly 1 trace end and 1 span end
        self.mock_writer.assert_entity_action_count("trace", "end", 1)
        self.mock_writer.assert_entity_action_count("span", "end", 1)

    def test_anthropic(self) -> None:
        if not anthropicApiKey:
            self.skipTest("Anthropic API key not available")

        callback = MaximLiteLLMTracer(self.logger)
        litellm.callbacks = [callback]
        response = completion(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key=anthropicApiKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        print(response)

        # Give LiteLLM callback time to complete
        time.sleep(0.5)

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

        # Clear LiteLLM callbacks to ensure test isolation
        litellm.callbacks = []


class TestLiteLLMAsync(unittest.IsolatedAsyncioTestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    async def asyncSetUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        # Clear any existing LiteLLM callbacks before setting up new ones
        litellm.callbacks = []

        self.maxim = Maxim()
        self.logger = self.maxim.logger()
        self.mock_writer = inject_mock_writer(self.logger)
        callback = MaximLiteLLMTracer(self.logger)
        # Clear any leftover container state
        callback.containers.clear()
        litellm.callbacks = [callback]

    async def test_openai_async(self) -> None:
        if not openAIKey:
            self.skipTest("OpenAI API key not available")

        response = await acompletion(
            model="openai/gpt-4o",
            api_key=openAIKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        print(response)
        # Remove the long sleep for faster testing
        # await asyncio.sleep(10)

        # Give LiteLLM callback time to complete
        time.sleep(0.5)

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

    async def asyncTearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()
        self.maxim.cleanup()

        # Clear LiteLLM callbacks to ensure test isolation
        litellm.callbacks = []
