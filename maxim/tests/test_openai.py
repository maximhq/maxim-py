import json
import logging
import os
import unittest
from uuid import uuid4

from openai import OpenAI
from openai.types.chat import ChatCompletion

from maxim import Config, Maxim
from maxim.logger import (
    GenerationConfig,
    GenerationRequestMessage,
    LoggerConfig,
    TraceConfig,
)
from maxim.logger.openai import MaximOpenAIClient
from maxim.tests.mock_writer import inject_mock_writer
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)

openaiApiKey = os.getenv("OPENAI_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL")
repoId = os.getenv("MAXIM_LOG_REPO_ID")


class TestAsyncOpenAI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim({ "base_url": baseUrl, "debug": True }).logger()
        # self.mock_writer = inject_mock_writer(self.logger)

    async def test_async_chat_completions(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        ).aio
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how ML works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    async def test_async_chat_completions_stream(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        ).aio
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how Deep learning works"}],
            model="gpt-4o",
            max_tokens=1000,
            stream=True,
        )
        async for chunk in response:
            print(chunk.choices[0].delta.content or "", end="")

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    async def asyncTearDown(self) -> None:
        # Print final summary for debugging
        # self.mock_writer.print_logs_summary()
        #
        # # Cleanup the mock writer
        # self.mock_writer.cleanup()
        self.logger.flush();


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim({ "base_url": baseUrl }).logger()
        # self.mock_writer = inject_mock_writer(self.logger)

    def test_chat_completions(self):
        client = OpenAI(api_key=openaiApiKey)
        # trace = self.logger.trace(TraceConfig(id=str(uuid4())))
        # config = GenerationConfig(
        #     id=str(uuid4()),
        #     model="gpt-4o",
        #     provider="openai",
        #     model_parameters={"max_tokens": 1000},
        #     messages=[
        #         GenerationRequestMessage(role="user", content="Explain how  works")
        #     ],
        # )
        # generation = trace.generation(config)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how F1 works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_chat_completions_using_wrapper(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_chat_completions_stream_using_wrapper(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        )
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how Nascar works"}],
            model="gpt-4o",
            max_tokens=1000,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in response:
            # print(event)
            pass

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_chat_completions_with_tool_calls(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        )

        # Define the tools
        tools = [
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
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "What's the weather like in New York?"}],
            model="gpt-4o",
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )
        print("Response: ", response)

        # Flush the logger and verify logging
        self.logger.flush()
        # self.mock_writer.print_logs_summary()
        #
        # # Assert that we have exactly 1 add-generation log
        # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        #
        # # Assert that we have exactly 1 result log on generation
        # self.mock_writer.assert_entity_action_count("generation", "result", 1)
        #
        # # Assert that we have exactly 1 trace create log
        # self.mock_writer.assert_entity_action_count("trace", "create", 1)
        #
        # # Assert that we have exactly 1 trace end log
        # self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def tearDown(self) -> None:
        # Print final summary for debugging
        # self.mock_writer.print_logs_summary()
        #
        # # Cleanup the mock writer
        # self.mock_writer.cleanup()
        return super().tearDown()
