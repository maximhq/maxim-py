import json
import logging
import os
import unittest

import dotenv
from openai import OpenAI

from maxim import Maxim
from maxim.logger.openai import MaximOpenAIClient

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
        self.logger = Maxim({"api_key": apiKey, "base_url": baseUrl}).logger()

    async def test_async_chat_completions(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how ML works"}],
            model="gpt-4o",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "test": "test",
                    "test2": "test2",
                    "test3": "test3",
                    "test4": "test4",
                    "test5": "test5",
                    "test6": "test6",
                    "test7": "test7",
                    "test8": "test8",
                    "test9": "test9",
                    "test10": "test10",
                }),
            },
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()

    async def test_async_chat_completions_stream(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how Deep learning works"}],
            model="gpt-4o",
            max_tokens=1000,
            stream=True,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "stream_test": "stream_test",
                    "stream_test2": "stream_test2",
                    "stream_test3": "stream_test3",
                    "stream_test4": "stream_test4",
                    "stream_test5": "stream_test5",
                }),
                "x-maxim-generation-name": "test_async_chat_completions_stream",
                "x-maxim-trace-id": "8e9b88ca-10ba-4206-b287-8d4aea73719f",
            },
        )

        # Collect all chunks
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            if (
                chunk.choices
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content
            ):
                print(chunk.choices[0].delta.content, end="", flush=True)

        # Verify we got some content
        self.assertTrue(len(chunks) > 0, "No chunks received from the stream")

        # Flush the logger and verify logging
        self.logger.flush()

    async def test_async_chat_completions_with_functions(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio

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
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            model="gpt-4o",
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )
        print("Response: ", response)
        self.logger.flush()

    async def test_async_chat_completions_with_system_message(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio

        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant who always writes code in Python.",
            },
            {
                "role": "user",
                "content": "Write a function to calculate fibonacci numbers",
            },
        ]

        response = await client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": {
                    "test": "test",
                },
            },
        )
        print("Response: ", response)
        self.logger.flush()

    async def test_async_chat_completions_error_handling(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio

        # Test invalid model
        with self.assertRaises(Exception):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="invalid-model",
                max_tokens=1000,
            )

        # Test invalid message format
        with self.assertRaises(Exception):
            await client.chat.completions.create(
                messages=[{"invalid_role": "user", "content": "Hello"}],
                model="gpt-4o",
                max_tokens=1000,
            )

        self.logger.flush()

    async def test_async_chat_completions_stream_with_functions(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger).aio

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
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            model="gpt-4o",
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
            stream=True,
        )

        # Collect all chunks
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            if (
                chunk.choices
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content
            ):
                print(chunk.choices[0].delta.content, end="", flush=True)

        # Verify we got some content
        self.assertTrue(len(chunks) > 0, "No chunks received from the stream")
        self.logger.flush()

    async def asyncTearDown(self) -> None:
        self.logger.flush()


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()

    def test_chat_completions(self):
        client = OpenAI(api_key=openaiApiKey)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how F1 works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()

    def test_chat_completions_using_wrapper(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "sync_test": "sync_test",
                    "sync_test3": "sync_test3",
                    "sync_test4": "sync_test4",
                    "sync_test5": "sync_test5",
                }),
                "x-maxim-trace-id": "8e9b88ca-10ba-4206-b287-8d4aea73719r",
                "x-maxim-generation-name": "test_chat_completions_using_wrapper",
            },
        )
        print(response)

        # Flush the logger and verify logging
        self.logger.flush()

    def test_chat_completions_stream_using_wrapper(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how Nascar works"}],
            model="gpt-4o",
            max_tokens=1000,
            stream=True,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "sync_stream_test": "sync_stream_test",
                    "sync_stream_test3": "sync_stream_test3",
                    "sync_stream_test4": "sync_stream_test4",
                    "sync_stream_test5": "sync_stream_test5",
                }),
                "x-maxim-trace-id": "8e9b88ca-10ba-4206-b287-8d4aea7371rr",
                "x-maxim-generation-name": "test_chat_completions_stream_using_wrapper",
            },
        )
        for event in response:
            print(event)

        # Flush the logger and verify logging
        self.logger.flush()

    def test_chat_completions_with_tool_calls(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

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
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            model="gpt-4o",
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )
        print("Response: ", response)

        # Flush the logger and verify logging
        self.logger.flush()

    def test_chat_completions_with_multiple_functions(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        # Define multiple tools
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
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_restaurant",
                    "description": "Find a restaurant in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            },
                            "cuisine": {
                                "type": "string",
                                "description": "Type of cuisine",
                            },
                        },
                        "required": ["location", "cuisine"],
                    },
                },
            },
        ]

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Find me an Italian restaurant in New York and tell me if I need an umbrella",
                }
            ],
            model="gpt-4o",
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )
        print("Response: ", response)
        self.logger.flush()

    def test_chat_completions_with_system_message(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful coding assistant who always writes code in Python.",
            },
            {
                "role": "user",
                "content": "Write a function to calculate fibonacci numbers",
            },
        ]

        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            max_tokens=1000,
        )
        print("Response: ", response)
        self.logger.flush()

    def test_chat_completions_with_response_format(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with name: John Doe, age: 30",
                }
            ],
            model="gpt-4o",
            response_format={"type": "json_object"},
            max_tokens=1000,
        )
        print("Response: ", response)
        self.logger.flush()

    def test_chat_completions_with_temperature(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        # Test with different temperature settings
        response_creative = client.chat.completions.create(
            messages=[{"role": "user", "content": "Write a short story about a robot"}],
            model="gpt-4o",
            temperature=1.0,  # More creative
            max_tokens=1000,
        )
        print("Creative Response: ", response_creative)

        response_focused = client.chat.completions.create(
            messages=[{"role": "user", "content": "Write a short story about a robot"}],
            model="gpt-4o",
            temperature=0.2,  # More focused
            max_tokens=1000,
        )
        print("Focused Response: ", response_focused)
        self.logger.flush()

    def test_chat_completions_error_handling(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        # Test invalid model
        with self.assertRaises(Exception):
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="invalid-model",
                max_tokens=1000,
            )

        # Test invalid message format
        with self.assertRaises(Exception):
            client.chat.completions.create(
                messages=[{"invalid_role": "user", "content": "Hello"}],
                model="gpt-4o",
                max_tokens=1000,
            )

        # Test empty messages
        with self.assertRaises(Exception):
            client.chat.completions.create(
                messages=[],
                model="gpt-4o",
                max_tokens=1000,
            )

        self.logger.flush()

    def test_chat_completions_with_seed(self):
        client = MaximOpenAIClient(OpenAI(api_key=openaiApiKey), logger=self.logger)

        # Make two identical requests with the same seed
        seed_value = 123
        response1 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Generate a random number between 1 and 100",
                }
            ],
            model="gpt-4o",
            seed=seed_value,
            max_tokens=1000,
        )

        response2 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Generate a random number between 1 and 100",
                }
            ],
            model="gpt-4o",
            seed=seed_value,
            max_tokens=1000,
        )

        print("Response 1: ", response1)
        print("Response 2: ", response2)
        self.logger.flush()

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
