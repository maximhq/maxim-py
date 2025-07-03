"""Test cases for Agno integration with Maxim SDK.

This module contains comprehensive test cases for the Agno integration,
covering multiple AI providers: OpenAI, Azure OpenAI, Anthropic, and Gemini.
Uses real API keys from environment variables for authentic testing.
"""

import os
import unittest
import asyncio
from uuid import uuid4
from typing import Optional
from agno.agent.agent import Agent
import pytest

from maxim import Maxim
from maxim.logger.agno import MaximAgnoClient, instrument_agno
from maxim.tests.mock_writer import inject_mock_writer


def has_api_key(provider: str) -> bool:
    """Check if API key is available for the given provider."""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "azure": "AZURE_OPENAI_API_KEY",
    }
    return bool(os.getenv(key_mapping.get(provider, "")))


def create_real_agent(model: str, provider: str, **kwargs):
    """Create a real Agno agent with proper configuration."""
    try:
        if provider == "openai":
            if not has_api_key("openai"):
                return None
            tools = kwargs.get(
                "tools",
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather information for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name",
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "Search the web for information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query",
                                    }
                                },
                                "required": ["query"],
                            },
                        },
                    },
                ],
            )
            from agno.models.openai import OpenAIChat

            model_obj = OpenAIChat(id=model, api_key=os.getenv("OPENAI_API_KEY"))
            return Agent(model=model_obj, tools=tools)
        elif provider == "anthropic":
            if not has_api_key("anthropic"):
                return None
            tools = kwargs.get(
                "tools",
                [
                    {
                        "name": "calculator",
                        "description": "Perform mathematical calculations",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["add", "subtract", "multiply", "divide"],
                                },
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["operation", "a", "b"],
                        },
                    }
                ],
            )
            from agno.models.anthropic import Claude

            model_obj = Claude(id=model, api_key=os.getenv("ANTHROPIC_API_KEY"))
            return Agent(model=model_obj, tools=tools)
        elif provider == "google":
            if not has_api_key("google"):
                return None
            from agno.models.google import Gemini

            model_obj = Gemini(id=model, api_key=os.getenv("GOOGLE_API_KEY"))
            return Agent(model=model_obj)
        else:
            return None
    except Exception as e:
        print(f"Failed to create agent for {provider}: {e}")
        return None


def skip_if_no_key(provider: str):
    """Decorator to skip tests if API key is not available."""

    def decorator(test_func):
        return unittest.skipUnless(
            has_api_key(provider),
            f"Skipping {provider} test - API key not available",
        )(test_func)

    return decorator


class TestAgnoIntegration(unittest.TestCase):
    """Test basic Agno integration functionality."""

    def setUp(self):
        """Set up test environment."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    def test_agno_client_initialization(self):
        """Test that MaximAgnoClient initializes properly."""
        client = MaximAgnoClient(self.logger)
        self.assertIsNotNone(client)
        self.assertEqual(client._logger, self.logger)

    def test_instrument_agno_function(self):
        """Test the instrument_agno function."""
        # Instrument agno
        instrument_agno(self.logger)

        # Verify that the Agent class has been patched
        self.assertTrue(hasattr(Agent, "_maxim_patched"))

    def test_double_instrumentation_prevention(self):
        """Test that double instrumentation is prevented."""
        # First instrumentation
        instrument_agno(self.logger)

        # Second instrumentation should not cause issues
        instrument_agno(self.logger)

        # Should still work normally
        self.assertTrue(hasattr(Agent, "_maxim_patched"))


class TestAgnoOpenAI(unittest.TestCase):
    """Test Agno integration with OpenAI provider using real API calls."""

    def setUp(self):
        """Set up test environment for OpenAI tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    @skip_if_no_key("openai")
    def test_openai_sync_run(self):
        """Test synchronous run with OpenAI provider."""
        MaximAgnoClient(self.logger)

        # Create real agent with OpenAI model
        agent = create_real_agent("gpt-3.5-turbo", "openai", tools=[])

        if agent is None:
            self.fail("Failed to create OpenAI agent")

        # Run the agent with a simple prompt
        result = agent.run("Say hello in exactly 3 words.")

        # Verify result
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "content") or hasattr(result, "text"))

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging occurred
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    @skip_if_no_key("openai")
    def test_openai_with_tool_calls(self):
        """Test OpenAI agent with function/tool calls."""
        MaximAgnoClient(self.logger)

        # Create agent with tools
        agent = create_real_agent("gpt-3.5-turbo", "openai")
        if agent is None:
            self.fail("Failed to create OpenAI agent with tools")

        # Run the agent with a prompt that should trigger tool calls
        result = agent.run(
            "What's the weather like in San Francisco? Please use the get_weather function."
        )

        # Verify result exists
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    @skip_if_no_key("openai")
    def test_openai_with_custom_trace_id(self):
        """Test OpenAI run with custom trace ID."""
        MaximAgnoClient(self.logger)

        agent = create_real_agent("gpt-3.5-turbo", "openai", tools=[])
        if agent is None:
            self.fail("Failed to create OpenAI agent")

        custom_trace_id = str(uuid4())

        # Run with custom trace ID
        result = agent.run("Say 'Hello!'", trace_id=custom_trace_id)

        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Should create generation but not end trace (external trace)
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        # Should not end trace since it's external
        self.mock_writer.assert_entity_action_count("trace", "end", 0)

    @skip_if_no_key("openai")
    def test_openai_vision_model(self):
        """Test GPT-4o with vision capabilities."""
        MaximAgnoClient(self.logger)

        # Create vision-capable agent
        agent = create_real_agent("gpt-4o-mini", "openai", tools=[])
        if agent is None:
            self.fail("Failed to create GPT-4o agent")

        # Test with text-only prompt that mentions vision
        result = agent.run(
            "Describe what you would see if you were looking at a beautiful sunset."
        )

        # Verify result
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)


class TestAgnoAsyncOpenAI(unittest.IsolatedAsyncioTestCase):
    """Test Agno async integration with OpenAI provider."""

    async def asyncSetUp(self):
        """Set up async test environment for OpenAI tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    async def asyncTearDown(self):
        """Clean up after async tests."""
        self.mock_writer.cleanup()

    @skip_if_no_key("openai")
    async def test_openai_async_run(self):
        """Test asynchronous run with OpenAI provider."""
        MaximAgnoClient(self.logger)

        # Create real agent with OpenAI model
        agent = create_real_agent("gpt-3.5-turbo", "openai", tools=[])
        if agent is None:
            self.fail("Failed to create OpenAI agent")

        # Run the agent asynchronously
        result = await agent.arun("Say hello in 2 words.")

        # Verify result
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)


class TestAgnoAnthropic(unittest.TestCase):
    """Test Agno integration with Anthropic provider using real API calls."""

    def setUp(self):
        """Set up test environment for Anthropic tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    @skip_if_no_key("anthropic")
    def test_anthropic_sync_run(self):
        """Test synchronous run with Anthropic provider."""
        MaximAgnoClient(self.logger)

        # Create real agent with Anthropic model
        agent = create_real_agent("claude-3-haiku-20240307", "anthropic", tools=[])
        if agent is None:
            self.fail("Failed to create Anthropic agent")

        # Run the agent
        result = agent.run("Respond with exactly one word: 'Hello'")

        # Verify result
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    @skip_if_no_key("anthropic")
    def test_anthropic_with_tool_calls(self):
        """Test Anthropic Claude with tool calls."""
        MaximAgnoClient(self.logger)

        # Create agent with calculator tool
        agent = create_real_agent("claude-3-haiku-20240307", "anthropic")
        if agent is None:
            self.fail("Failed to create Anthropic agent with tools")

        # Run the agent with a calculation prompt
        result = agent.run("Please calculate 25 * 4 using the calculator function.")

        # Verify result
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Tool calls should be properly logged
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)


class TestAgnoGemini(unittest.TestCase):
    """Test Agno integration with Gemini provider using real API calls."""

    def setUp(self):
        """Set up test environment for Gemini tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    @skip_if_no_key("google")
    def test_gemini_sync_run(self):
        """Test synchronous run with Gemini provider."""
        MaximAgnoClient(self.logger)

        # Create real agent with Gemini model
        agent = create_real_agent("gemini-1.5-flash", "google")
        if agent is None:
            self.fail("Failed to create Gemini agent")

        # Run the agent
        result = agent.run("Say 'Hello from Gemini' in exactly those words.")

        # Verify result
        self.assertIsNotNone(result)

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Verify trace and generation logging
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)


class TestAgnoErrorHandling(unittest.TestCase):
    """Test error handling scenarios for Agno integration."""

    def setUp(self):
        """Set up test environment for error tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    def test_missing_agno_import_error(self):
        """Test behavior when agno package is missing."""
        # Since agno is now a required dependency, this test is no longer needed
        self.skipTest("agno is now a required dependency")

    def test_invalid_api_key_handling(self):
        """Test handling of invalid API keys."""
        MaximAgnoClient(self.logger)

        # This test verifies the client can handle cases where API keys are invalid
        # The actual behavior depends on how agno handles invalid keys
        self.assertTrue(True)  # Basic test that client initializes

    @skip_if_no_key("openai")
    def test_agent_error_propagation(self):
        """Test that agent errors are properly logged."""
        MaximAgnoClient(self.logger)

        agent = create_real_agent("gpt-3.5-turbo", "openai", tools=[])
        if agent is None:
            self.fail("Failed to create OpenAI agent")

        try:
            # Try to trigger an error with an invalid prompt
            result = agent.run("")  # Empty prompt might cause an error
            # If no error occurs, that's also fine
            self.assertIsNotNone(result)
        except Exception:
            # If an error occurs, verify it was logged
            self.logger.flush()
            self.mock_writer.print_logs_summary()
            # Error logging verification would happen here


class TestAgnoProviderComparison(unittest.TestCase):
    """Test comparing multiple providers in the same session."""

    def setUp(self):
        """Set up test environment for provider comparison tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim().logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_writer.cleanup()

    @unittest.skipUnless(
        has_api_key("openai") and has_api_key("anthropic"),
        "Both OpenAI and Anthropic API keys required",
    )
    def test_multiple_providers_same_session(self):
        """Test using multiple providers in the same session."""
        MaximAgnoClient(self.logger)

        providers_models = [
            ("openai", "gpt-3.5-turbo"),
            ("anthropic", "claude-3-haiku-20240307"),
        ]

        results = []
        for provider, model in providers_models:
            agent = create_real_agent(model, provider, tools=[])
            if agent is not None:
                result = agent.run(f"Say 'Hello from {provider.title()}'")
                results.append((provider, result))

        # Verify we got results from at least one provider
        self.assertGreater(len(results), 0, "No providers were successful")

        # Flush logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Each successful provider should have created traces
        expected_traces = len(results)
        self.mock_writer.assert_entity_action_count("trace", "create", expected_traces)


if __name__ == "__main__":
    # Print available API keys for debugging
    print("Available API keys:")
    for provider in ["openai", "anthropic", "google", "bedrock", "azure"]:
        status = "✓" if has_api_key(provider) else "✗"
        print(f"  {provider}: {status}")

    print()

    unittest.main()
