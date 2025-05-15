import asyncio
import json
import logging
import os
import unittest
from uuid import uuid4

import litellm
import pytest
from aiounittest import AsyncTestCase
from litellm import acompletion, completion

from maxim import Config, Maxim
from maxim.logger import LoggerConfig, TraceConfig
from maxim.logger.litellm.tracer import MaximLiteLLMTracer

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "beta"

awsAccessKeyId = data["bedrockAccessKey"]
awsAccessKeySecret = data["bedrockSecretKey"]
azureOpenAIBaseUrl = data["azureOpenAIBaseUrl"]
azureOpenAIKey = data["azureOpenAIKey"]
openAIKey = data["openAIKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]
anthropicApiKey = data["anthropicApiKey"]


maxim = Maxim(
    Config(api_key=apiKey, base_url=baseUrl, debug=True, raise_exceptions=True)
)
logger = maxim.logger(LoggerConfig(id=repoId))
callback = MaximLiteLLMTracer(logger)
litellm.callbacks = [callback]

class TestLiteLLM(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self):
        pass

    def test_openai(self) -> None:
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
        
    
    def test_embedding_call(self) -> None:
        response = litellm.embedding(
            model="text-embedding-ada-002",
            input=["Hello world", "Test embedding"],
            api_key=openAIKey
        )
        print(response)

    
    def test_openai_with_external_trace(self) -> None:
        trace_id = str(uuid4())
        trace = logger.trace(TraceConfig(id=trace_id, name="external_trace"))

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

    def test_anthropic(self) -> None:
        callback = MaximLiteLLMTracer(logger)
        litellm.callbacks = [callback]
        response = completion(
            model="anthropic/claude-3-5-sonnet",
            api_key=anthropicApiKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        print(response)

    def tearDown(self) -> None:
        maxim.cleanup()


class TestLiteLLMAsync(AsyncTestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self):
        pass

    @pytest.mark.asyncio
    async def test_openai_async(self) -> None:
        response = await acompletion(
            model="openai/gpt-4o",
            api_key=openAIKey,
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        print(response)
        await asyncio.sleep(10)

    def tearDown(self) -> None:
        maxim.cleanup()
