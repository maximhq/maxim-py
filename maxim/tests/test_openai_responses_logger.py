"""
Test case for logging OpenAI Responses API results with tool calls.

This test demonstrates:
1. Making an initial request with a tool definition
2. Capturing tool calls from the response
3. Submitting tool results back to the API
4. Logging the final response from the LLM considering the tool result

All dependencies and code stay within the test bounds.
"""

import json
import logging
import os
import unittest
from uuid import uuid4

import dotenv
from openai import OpenAI

from maxim import Maxim

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)

openai_api_key = os.getenv("OPENAI_API_KEY")
maxim_api_key = os.getenv("MAXIM_API_KEY")
maxim_base_url = os.getenv("MAXIM_BASE_URL")
maxim_repo_id = os.getenv("MAXIM_LOG_REPO_ID")


class TestOpenAIResponsesLogger(unittest.TestCase):
    """Test cases for OpenAI Responses API logging with tool calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset Maxim singleton
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            {
                "api_key": maxim_api_key,
                "base_url": maxim_base_url,
            }
        )
        self.logger = self.maxim.logger(
            {
                "id": maxim_repo_id,
            }
        )
        self.openai_client = OpenAI(api_key=openai_api_key)

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.flush()

    def simulate_weather_tool_result(self, location: str) -> str:
        """
        Simulate a weather tool result.

        Args:
            location: The location to get weather for.

        Returns:
            A simulated weather result as a JSON string.
        """
        # Simulated weather data
        weather_data = {
            "location": location,
            "temperature": 72,
            "humidity": 65,
            "conditions": "Partly cloudy",
            "wind_speed": 8,
        }
        return json.dumps(weather_data)

    def test_responses_api_with_tool_call_and_result(self):
        """
        Test logging OpenAI Responses API with tool calls and results.

        This test:
        1. Creates a session/trace/span in Maxim
        2. Makes an initial API call with a tool definition
        3. Extracts and logs the tool call on the span
        4. Simulates executing the tool
        5. Logs the tool result via the tool_call object
        6. Submits the tool result back to the API
        7. Logs the final response
        """
        # Create Maxim tracing hierarchy
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "OpenAI Responses API with Tools",
            }
        )

        trace = session.trace(
            {
                "id": str(uuid4()),
                "name": "Weather Tool Scenario",
            }
        )

        span = trace.span(
            {
                "id": str(uuid4()),
                "name": "User <> GPT-4o",
            }
        )

        # Create a generation to track messages and final result
        generation = span.generation(
            {
                "id": str(uuid4()),
                "name": "Weather Query with Tool",
                "messages": [],
                "model": "gpt-4o",
                "provider": "openai",
                "model_parameters": {},
            }
        )

        try:
            # Define the weather tool
            tools = [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ]

            # Initial prompt requesting the tool
            user_prompt = "What's the current weather in New York, NY?"

            # Add the user message to generation tracking
            generation.add_message({"role": "user", "content": user_prompt})
            trace.set_input(user_prompt)

            # Step 1: Make the initial API call with tool definition
            print("\n=== Step 1: Initial API call with tool definition ===")
            first_response = self.openai_client.responses.create(
                model="gpt-4o",
                input=user_prompt,
                tools=tools,
            )

            print(f"Response ID: {first_response.id}")
            print(f"Status: {first_response.status}")
            print(
                f"Output: {first_response.output_text if hasattr(first_response, 'output_text') else 'N/A'}"
            )

            # Log the first response to generation
            generation.add_message(
                {"role": "assistant", "content": first_response.output_text}
            )

            # Step 2: Extract tool calls from the response
            print("\n=== Step 2: Extract tool calls ===")
            tool_calls = []
            if hasattr(first_response, "output") and first_response.output:
                for item in first_response.output:
                    if hasattr(item, "type") and item.type == "function_call":
                        tool_calls.append(item)
                        print(f"Tool call found: {item.name} (ID: {item.call_id})")
                        print(f"Arguments: {item.arguments}")

            if not tool_calls:
                print("No tool calls found in response")
                generation.result(first_response)
                return

            # Step 3: Create tool call on span and simulate execution
            print("\n=== Step 3: Create tool call and simulate execution ===")
            tool_call = tool_calls[0]

            # Create tool call on the span
            span_tool_call = span.tool_call(
                {
                    "id": tool_call.call_id,
                    "name": tool_call.name,
                    "description": "built-in tool",
                    "args": tool_call.arguments,
                }
            )

            # Parse tool arguments
            try:
                args = json.loads(tool_call.arguments)
                location = args.get("location", "Unknown")
            except (json.JSONDecodeError, TypeError):
                location = "Unknown"

            # Simulate the tool
            tool_result = self.simulate_weather_tool_result(location)
            print(f"Tool result: {tool_result}")

            # Log the tool result via the tool_call object
            span_tool_call.result(tool_result)

            # Step 4: Add assistant message with tool call
            generation.add_message(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )

            # Step 5: Submit tool result and get final response
            print("\n=== Step 4: Submit tool result and get final response ===")
            final_response = self.openai_client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": tool_result,
                    }
                ],
                previous_response_id=first_response.id,
            )

            print(f"Final Response ID: {final_response.id}")
            print(f"Status: {final_response.status}")
            print(
                f"Output: {final_response.output_text if hasattr(final_response, 'output_text') else 'N/A'}"
            )

            # Log the final response to generation
            generation.result(final_response)

            # Verify we got meaningful responses
            self.assertIsNotNone(first_response.id)
            self.assertIsNotNone(final_response.id)
            self.assertEqual(first_response.status, "completed")
            self.assertEqual(final_response.status, "completed")

            print("\n=== Test completed successfully ===")

        except Exception as e:
            print(f"Error during test: {e}")
            raise
        finally:
            # Close tracing hierarchy
            span.end()
            trace.end()
            session.end()


if __name__ == "__main__":
    unittest.main()
