import os
import unittest
from uuid import uuid4

import dotenv
from google import genai
from google.genai.types import Content, Part

from maxim import Maxim
from maxim.logger.gemini import MaximGeminiClient

dotenv.load_dotenv()

geminiApiKey = os.getenv("GEMINI_API_KEY")


class TestAsyncGemini(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logger = Maxim().logger()

    async def test_async_generate_content(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        ).aio
        response = await client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
            config={
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        print(response)

    async def asyncTearDown(self) -> None:
        self.logger.flush()


class TestGemini(unittest.TestCase):
    def setUp(self):
        self.logger = Maxim().logger()

    def test_generate_content(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
        )
        print(response)

    def test_generate_content_streamed_using_wrapper(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )
        response = client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
            config={
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        for chunk in response:
            print(chunk.text)

    def test_generate_content_with_tool_call_using_wrapper(self):
        def get_current_weather(location: str) -> str:
            """Get the current whether in a given location.

            Args:
                location: required, The city and state, e.g. San Franciso, CA
                unit: celsius or fahrenheit
            """
            print(f"Called with: {location=}")
            return "23C"

        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Whats the weather of pune today?",
            config={
                "tools": [get_current_weather],
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        print(response)

    def test_generate_content_using_wrapper(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
            config={
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        print(response)

    def test_chat_create(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )

        chat = client.chats.create(
            model="gemini-2.0-flash",
            config={"system_instruction": "You are a helpful assistant"},
            history=[
                Content(
                    role="user",
                    parts=[Part(text="there is an emplyee called Akshay Deo")],
                )
            ],
        )
        response = chat.send_message("test")
        print(response)

    def test_generate_chat_create_using_wrapper(self):
        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config={"system_instruction": "You are a helpful assistant"},
            history=[
                Content(
                    role="user",
                    parts=[Part(text="there is an emplyee called Akshay Deo")],
                )
            ],
        )
        response = chat.send_message("test")
        print(response)

    def test_generate_chat_create_with_tool_call_using_wrapper(self):
        session_id = str(uuid4())

        def get_current_weather(location: str) -> str:
            """Get the current whether in a given location.

            Args:
                location: required, The city and state, e.g. San Franciso, CA
                unit: celsius or fahrenheit
            """
            print(f"Called with: {location=}")
            return "23C"

        client = MaximGeminiClient(
            genai.Client(api_key=geminiApiKey), logger=self.logger
        )
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config={
                "system_instruction": "You are a helpful assistant",
                "tools": [get_current_weather],
            },
            history=[
                Content(
                    role="user",
                    parts=[Part(text="Hows weather of SF looks today?")],
                )
            ],
            session_id=session_id,
        )
        response = chat.send_message("test")
        print(response)

    def tearDown(self) -> None:
        self.logger.flush()
        return super().tearDown()
