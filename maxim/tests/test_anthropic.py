import json
import logging
import os
import unittest
from uuid import uuid4

from anthropic import Anthropic
from anthropic.types import Message, MessageParam

from maxim import Config, Maxim
from maxim.logger import (
    GenerationConfig,
    GenerationRequestMessage,
    LoggerConfig,
    TraceConfig,
)
from maxim.logger.anthropic import MaximAnthropicClient,MaximAnthropicAsyncClient

# Load test config
with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.DEBUG)
env = "dev"

anthropicApiKey = data["anthropicApiKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]


class TestAsyncAnthropic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = Maxim(Config(api_key=apiKey, base_url=baseUrl)).logger(
            LoggerConfig(id=repoId)
        )

    async def test_async_messages(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        ).aio
        response = await client.messages(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        print(response)

    async def test_async_messages_stream(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        ).aio
        response = client.messages_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        async for chunk in response:
            print(chunk or "", end="")

    async def asyncTearDown(self) -> None:
        self.logger.flush()


class TestAnthropic(unittest.TestCase):
    def setUp(self):
        self.logger = Maxim(Config(api_key=apiKey, base_url=baseUrl)).logger(
            LoggerConfig(id=repoId)
        )

    def test_messages(self):
        client = Anthropic(api_key=anthropicApiKey)
        trace = self.logger.trace(TraceConfig(id=str(uuid4())))
        config = GenerationConfig(
            id=str(uuid4()),
            model="claude-3-5-sonnet-latest",
            provider="anthropic",
            model_parameters={"max_tokens": 1000},
            messages=[
                GenerationRequestMessage(role="user", content="Explain how AI works")
            ],
        )
        generation = trace.generation(config)
        response = client.messages.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        generation.result(response)
        print(response)

    def test_messages_using_wrapper(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        print(response)

    def test_messages_stream_using_wrapper(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        for event in response:
            pass
    def test_messages_with_system_prompt(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages(
            system="You are a helpful coding assistant",
            messages=[
                {"role": "user", "content": "Write a simple Python function to calculate factorial"},
            ],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )
        print(response)

    def tearDown(self) -> None:
        self.logger.flush()
        return super().tearDown()