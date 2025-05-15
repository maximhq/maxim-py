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


# Load test config
with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.DEBUG)
env = "dev"

openaiApiKey = data["openAIKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]


class TestAsyncOpenAI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = Maxim(Config(api_key=apiKey, base_url=baseUrl)).logger(
            LoggerConfig(id=repoId)
        )

    async def test_async_chat_completions(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        ).aio
        response = await client.chat_completions(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

    async def test_async_chat_completions_stream(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger 
        ).aio
        response = client.chat_completions_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        async for chunk in response:
            print(chunk or "", end="")

    async def asyncTearDown(self) -> None:
        self.logger.flush()


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.logger = Maxim(Config(api_key=apiKey, base_url=baseUrl)).logger(
            LoggerConfig(id=repoId)
        )

    def test_chat_completions(self):
        client = OpenAI(api_key=openaiApiKey)
        trace = self.logger.trace(TraceConfig(id=str(uuid4())))
        config = GenerationConfig(
            id=str(uuid4()),
            model="gpt-4o",
            provider="openai",
            model_parameters={"max_tokens": 1000},
            messages=[
                GenerationRequestMessage(role="user", content="Explain how AI works")
            ],
        )
        generation = trace.generation(config)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        generation.result(response)
        print(response)

    def test_chat_completions_using_wrapper(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        )
        response = client.chat_completions(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        print(response)

    def test_chat_completions_stream_using_wrapper(self):
        client = MaximOpenAIClient(
            OpenAI(api_key=openaiApiKey), logger=self.logger
        )
        response = client.chat_completions_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="gpt-4o",
            max_tokens=1000,
        )
        for event in response:
            print(event)
            pass


    def tearDown(self) -> None:
        self.logger.flush()
        return super().tearDown()