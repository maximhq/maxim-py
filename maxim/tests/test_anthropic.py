import json
import dotenv
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from anthropic import Anthropic, MessageStreamManager
from anthropic.types import Message, MessageParam

from maxim import Maxim
from maxim.logger.anthropic import MaximAnthropicClient
from maxim.tests.mock_writer import inject_mock_writer


dotenv.load_dotenv()


anthropicApiKey = os.getenv("ANTHROPIC_API_KEY")
apiKey = os.getenv("MAXIM_API_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
repoId = os.getenv("MAXIM_LOG_REPO_ID")


class TestAnthropicWithMockWriter(unittest.TestCase):
    """Test class demonstrating how to use MockLogWriter for verification."""

    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        # Create logger and patch its writer
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def test_messages_with_mock_writer_verification(self):
        """Test that demonstrates verifying logged commands with mock writer."""
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        # Make the API call
        response = client.messages.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-opus-4-5-20251101",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "test": "test",
                    "test2": "test2",
                    "test3": "test3",
                    "test4": "test4",
                    "test5": "test5",
                }),
            },
        )

        # Flush the logger to ensure all logs are processed
        self.logger.flush()

        # Print logs summary for debugging
        self.mock_writer.print_logs_summary()

        # Assert that we have at least one log
        all_logs = self.mock_writer.get_all_logs()
        self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

        # Assert that we have exactly 1 add-generation log
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)

        # Assert that we have exactly 1 result log on generation
        self.mock_writer.assert_entity_action_count("generation", "result", 1)

        # Assert that we have exactly 1 trace create log
        self.mock_writer.assert_entity_action_count("trace", "create", 1)

        # Assert that we have exactly 1 trace end log
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

        # Verify that flush was called
        self.assertGreater(
            self.mock_writer.flush_count, 0, "Expected flush to be called"
        )

    def test_stream_with_mock_writer_verification(self):
        """Test streaming with mock writer verification."""
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )

        # Clear any existing logs
        self.mock_writer.clear_logs()

        # Make the streaming API call and exhaust the stream
        response = client.messages.create_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-opus-4-5-20251101",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "stream_test": "stream_test",
                    "stream_test2": "stream_test2",
                    "stream_test3": "stream_test3",
                    "stream_test4": "stream_test4",
                    "stream_test5": "stream_test5",
                }),
            },
        )

        self.assertIsInstance(response, MessageStreamManager)
        with response as stream:
            for event in stream:
                pass

        # Flush the logger
        self.logger.flush()

        # Print logs summary for debugging
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
        return super().tearDown()


class TestAnthropic(unittest.TestCase):
    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()

    def test_messages_using_wrapper(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages.create(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-opus-4-5-20251101",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "test": "test",
                    "test2": "test2",
                    "test3": "test3",
                    "test4": "test4",
                    "test5": "test5",
                }),
            },
        )
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, "content"))
        self.assertTrue(isinstance(response.content, list))
        self.assertTrue(len(response.content) > 0)
        self.logger.flush()

    def test_messages_stream_using_wrapper(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages.create_stream(
            messages=[{"role": "user", "content": "Explain how AI works"}],
            model="claude-opus-4-5-20251101",
            max_tokens=1000,
            extra_headers={
                "x-maxim-trace-tags": json.dumps({
                    "stream_test": "stream_test",
                    "stream_test3": "stream_test3",
                    "stream_test4": "stream_test4",
                    "stream_test5": "stream_test5",
                }),
            },
        )

        # Verify response is a MessageStreamManager
        self.assertIsInstance(response, MessageStreamManager)

        # Consume the stream
        event_count = 0
        for event in response:
            event_count += 1

        # Verify we received events
        self.assertGreater(event_count, 0, "Expected to receive streaming events")

        # Flush the logger and verify logging
        self.logger.flush()

    def test_messages_with_system_prompt(self):
        client = MaximAnthropicClient(
            Anthropic(api_key=anthropicApiKey), logger=self.logger
        )
        response = client.messages.create(
            system="You are a helpful coding assistant",
            messages=[
                {
                    "role": "user",
                    "content": "Write a simple Python function to calculate factorial",
                },
            ],
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
        )

        # Verify the response structure
        self.assertIsInstance(response, Message)
        self.assertTrue(hasattr(response, "content"))
        self.assertTrue(isinstance(response.content, list))
        self.assertTrue(len(response.content) > 0)

        # Verify the response contains code (since we asked for a function)
        response_text = ""
        for content_block in response.content:
            if hasattr(content_block, "text"):
                response_text += content_block.text

        self.assertIn(
            "def",
            response_text.lower(),
            "Expected response to contain a function definition",
        )
        self.assertIn(
            "factorial", response_text.lower(), "Expected response to mention factorial"
        )

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def tearDown(self) -> None:
        return super().tearDown()
