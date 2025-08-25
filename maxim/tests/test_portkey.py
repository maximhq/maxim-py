import json
import logging
import os
from typing import Any
import unittest
import uuid

import portkey_ai

from maxim import Maxim
from maxim.logger.portkey import MaximPortkeyClient, instrument_portkey
from maxim.tests.mock_writer import inject_mock_writer

# Create a mock logger for testing
portkey_api_key = os.getenv("PORTKEY_API_KEY")
portkey_virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
logger = Maxim({"base_url": baseUrl}).logger()


# Set up global logger state to debug for testing
logging.basicConfig(level=logging.DEBUG)


class TestPortkeyIntegration(unittest.TestCase):

    def setUp(self):
        # This is a hack to ensure that the Maxim instance is not cached
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        self.logger = Maxim({"base_url": baseUrl}).logger()
        self.mock_writer = inject_mock_writer(self.logger)

    def test_instrument_portkey_sync(self):
        client = instrument_portkey(
            portkey_ai.Portkey(
                api_key=portkey_api_key, virtual_key=portkey_virtual_key
            ),
            self.logger,
        )
        self.assertIsInstance(client, MaximPortkeyClient)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Tell me what is big bang theory in 100 words?",
                }
            ],
            model="gpt-4o",
            max_tokens=1000,
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

    def test_portkey_tool_calls_sync(self):
        """Test Portkey integration with tool calls (synchronous)."""
        # Create a Portkey client and instrument it
        portkey_client = portkey_ai.Portkey(
            api_key=portkey_api_key, virtual_key=portkey_virtual_key
        )
        instrumented_client = MaximPortkeyClient(portkey_client, self.logger)

        # Define tools for the model
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
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
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_tip",
                    "description": "Calculate tip amount for a bill",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bill_amount": {
                                "type": "number",
                                "description": "The total bill amount",
                            },
                            "tip_percentage": {
                                "type": "number",
                                "description": "The tip percentage (e.g., 15 for 15%)",
                            },
                        },
                        "required": ["bill_amount", "tip_percentage"],
                    },
                },
            },
        ]

        # Test 1: Simple tool call request
        print("=== Test 1: Weather Tool Call ===")
        try:
            response: Any = instrumented_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in San Francisco?",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=150,
            )
            print(f"Response: {response}")

            # Check if the model wants to call a tool
            if response.choices[0].message.tool_calls:
                print(f"Tool calls: {response.choices[0].message.tool_calls}")

                # Simulate tool execution and follow-up
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        # Parse arguments
                        args = json.loads(tool_call.function.arguments)
                        print(f"Weather requested for: {args.get('location')}")

                        # Simulate weather API response
                        weather_result = {
                            "location": args.get("location"),
                            "temperature": "72°F",
                            "condition": "Sunny",
                            "humidity": "65%",
                        }

                        # Continue the conversation with tool result
                        follow_up_response: Any = (
                            instrumented_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": "What's the weather like in San Francisco?",
                                    },
                                    response.choices[0].message.dict(),
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(weather_result),
                                    },
                                ],
                                tools=tools,
                                max_tokens=150,
                                extra_headers={
                                    "x-maxim-trace-id": "weather-test-trace",
                                    "x-maxim-generation-name": "weather-followup",
                                },
                            )
                        )
                        print(
                            f"Follow-up response: {follow_up_response.choices[0].message.content}"
                        )

        except Exception as e:
            print(f"Weather test error: {e}")

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # With tool calls, we expect multiple generations (initial + follow-up)
        # We'll verify that at least some logs were generated
        all_logs = self.mock_writer.get_all_logs()
        self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

    def test_portkey_multiple_tool_calls(self):
        portkey_client = portkey_ai.Portkey(
            api_key=portkey_api_key, virtual_key=portkey_virtual_key
        )
        instrumented_client = MaximPortkeyClient(portkey_client, self.logger)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
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
                                "description": "The temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_tip",
                    "description": "Calculate tip amount for a bill",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bill_amount": {
                                "type": "number",
                                "description": "The total bill amount",
                            },
                            "tip_percentage": {
                                "type": "number",
                                "description": "The tip percentage (e.g., 15 for 15%)",
                            },
                        },
                        "required": ["bill_amount", "tip_percentage"],
                    },
                },
            },
        ]
        trace_id = str(uuid.uuid4())
        print("\n=== Test 2: Multiple Tool Calls ===")
        try:
            response: Any = instrumented_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "I had dinner for $85.50. Calculate a 18% tip and also tell me the weather in New York.",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=200,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                    "x-maxim-generation-name": "multi-tool-generation",
                },
            )
            print(f"Response: {response}")

            if response.choices[0].message.tool_calls:
                print(
                    f"Number of tool calls: {len(response.choices[0].message.tool_calls)}"
                )
                for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                    print(f"Tool call {i+1}: {tool_call.function.name}")
                    print(f"Arguments: {tool_call.function.arguments}")

        except Exception as e:
            print(f"Multi-tool test error: {e}")

        # Test 3: Forced tool call
        print("\n=== Test 3: Forced Tool Call ===")
        try:
            response = instrumented_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "Calculate tip for a $50 bill with 20% tip",
                    }
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "calculate_tip"}},
                max_tokens=100,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                    "x-maxim-generation-name": "forced-tool-generation",
                },
            )
            print(f"Forced tool call response: {response}")

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                tip_amount = args.get("bill_amount", 0) * (
                    args.get("tip_percentage", 0) / 100
                )
                print(f"Calculated tip: ${tip_amount:.2f}")
        except Exception as e:
            print(f"Forced tool test error: {e}")

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # With multiple tool calls, we expect multiple generations
        # We'll verify that at least some logs were generated
        all_logs = self.mock_writer.get_all_logs()
        self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

    async def test_portkey_tool_calls_async(self):
        """Test Portkey integration with tool calls (asynchronous)."""
        # Create an async Portkey client and instrument it
        async_portkey_client = portkey_ai.AsyncPortkey(
            api_key=portkey_api_key, virtual_key=portkey_virtual_key
        )
        instrumented_client = MaximPortkeyClient(async_portkey_client, self.logger)

        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search for information in a database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "table": {
                                "type": "string",
                                "description": "The database table to search",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        trace_id = str(uuid.uuid4())
        print("=== Async Tool Call Test ===")
        try:
            response = await instrumented_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": "Search for user information for john.doe@example.com",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=100,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                    "x-maxim-generation-name": "async-search-generation",
                },
            )
            print(f"Async response: {response}")

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    print(f"Async tool call: {tool_call.function.name}")
                    print(f"Arguments: {tool_call.function.arguments}")

        except Exception as e:
            print(f"Async tool test error: {e}")

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Async tool calls should also generate logs
        all_logs = self.mock_writer.get_all_logs()
        self.assertGreater(len(all_logs), 0, "Expected at least one log to be captured")

    def test_tool_call_without_tools_parameter(self):
        """Test normal conversation without tools."""
        portkey_client = portkey_ai.Portkey(
            api_key=portkey_api_key, virtual_key=portkey_virtual_key
        )
        instrumented_client = MaximPortkeyClient(portkey_client, self.logger)
        trace_id = str(uuid.uuid4())
        print("=== Test Without Tools ===")
        try:
            response: Any = instrumented_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "Hello! How are you doing today?"}
                ],
                max_tokens=50,
                extra_headers={
                    "x-maxim-trace-id": trace_id,
                    "x-maxim-generation-name": "simple-chat",
                },
            )
            print(f"Simple chat response: {response.choices[0].message.content}")

            # Verify no tool calls
            self.assertIsNone(response.choices[0].message.tool_calls)

        except Exception as e:
            print(f"Simple chat error: {e}")

        # Flush the logger and verify logging
        self.logger.flush()
        self.mock_writer.print_logs_summary()

        # Simple chat should generate basic trace logs
        self.mock_writer.assert_entity_action_count("trace", "create", 1)
        self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
        self.mock_writer.assert_entity_action_count("generation", "result", 1)
        self.mock_writer.assert_entity_action_count("trace", "end", 1)

    def tearDown(self) -> None:
        # Print final summary for debugging
        self.mock_writer.print_logs_summary()

        # Cleanup the mock writer
        self.mock_writer.cleanup()


if __name__ == "__main__":
    # Run specific tests
    unittest.main()
