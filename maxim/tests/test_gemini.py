import os
import unittest
from uuid import uuid4

import dotenv
from google.genai import Client
from google.genai.types import Content, Part

from maxim import Maxim
from maxim.logger.gemini import MaximGeminiClient
from maxim.tests.mock_writer import inject_mock_writer

dotenv.load_dotenv()

geminiApiKey = os.getenv("GEMINI_API_KEY")


class TestAsyncGemini(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    async def test_async_generate_content(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger).aio
        response = await client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
            config={
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        print(response)

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


class TestGemini(unittest.TestCase):
    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def test_generate_content(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
        )
        print(response)

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

    def test_generate_content_streamed_using_wrapper(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)
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

    def test_generate_content_with_tool_call_using_wrapper(self):
        def get_current_weather(location: str) -> str:
            """Get the current whether in a given location.

            Args:
                location: required, The city and state, e.g. San Franciso, CA
                unit: celsius or fahrenheit
            """
            print(f"Called with: {location=}")
            return "23C"

        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)
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

    def test_generate_content_using_wrapper(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Explain how AI works",
            config={
                "system_instruction": "You are a helpful assisatant",
                "temperature": 0.8,
            },
        )
        print(response)

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

    def test_chat_create(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)

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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Chat creates a session first, then sends a message
        # So we expect session create + message generation logs
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def test_generate_chat_create_using_wrapper(self):
        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)
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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Chat creates a session first, then sends a message
        # So we expect session create + message generation logs
        # So we expect session create + message generation logs
        self.mock_writer.assert_entity_action_count("session", "create", 1)
        # So we expect session create + message generation logs
        self.mock_writer.assert_entity_action_count("session", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

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

        client = MaximGeminiClient(Client(api_key=geminiApiKey), logger=self.logger)
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

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Chat creates a session first, then sends a message
        # So we expect session create + message generation logs
        self.mock_writer.assert_entity_action_count("session", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def tearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()
        return super().tearDown()
