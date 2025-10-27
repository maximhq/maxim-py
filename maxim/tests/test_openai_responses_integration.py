import os
import json
import unittest
from uuid import uuid4

import dotenv
from openai import OpenAI
from maxim.logger.openai import MaximOpenAIClient
from openai.types.responses.function_tool_param import FunctionToolParam
# from openai.types.responses.tool_param import ToolParam

from maxim import Maxim


dotenv.load_dotenv()


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestOpenAIResponsesStreamingLogger(unittest.TestCase):
    def setUp(self):
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            {
                "api_key": os.getenv("MAXIM_API_KEY"),
                "base_url": os.getenv("MAXIM_BASE_URL"),
                "debug": True,
            }
        )
        self.logger = self.maxim.logger({"id": str(os.getenv("MAXIM_LOG_REPO_ID"))})
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def tearDown(self):
        if self.logger:
            self.logger.flush()

    def test_streaming_responses_logging(self):
        # Create local trace/generation via headers
        headers = {
            "x-maxim-generation-name": "Responses API Streaming Test",
        }

        wrapper = MaximOpenAIClient(self.client, logger=self.logger)
        with wrapper.responses.stream(
            model="gpt-4o",
            input="Write a 1-2 sentence haiku about autumn.",
            extra_headers=headers,
        ) as stream:
            # Iterate events and accumulate text to ensure stream behaves
            chunks = 0
            text_accum = []
            for event in stream:
                print(f"Event: {event}")
                chunks += 1
                t = getattr(event, "delta", None)
                if isinstance(t, str) and t:
                    text_accum.append(t)

            # Ensure we saw at least one event
            self.assertGreater(chunks, 0)
            # Combined text should be non-empty if deltas were present
            combined = "".join(text_accum)
            print(f"\n===\n{combined}\n===\n")
            self.assertIsInstance(combined, str)

    def test_non_streaming_responses_logging(self):
        # Create local trace/generation via headers
        headers = {
            "x-maxim-generation-name": "Responses API Non-Streaming Test",
        }

        wrapper = MaximOpenAIClient(self.client, logger=self.logger)
        response = wrapper.responses.create(
            model="gpt-4o",
            input="Explain JSON in one sentence.",
            extra_headers=headers,
        )

        # Basic assertions on Responses API object
        self.assertIsNotNone(getattr(response, "id", None))
        status = getattr(response, "status", None)
        self.assertIsInstance(status, str)
        # output_text should exist for basic text generations
        output_text = getattr(response, "output_text", None)
        self.assertIsInstance(output_text, str)

    def test_streaming_responses_multiturn_with_session(self):
        # Create a session and pass its ID via headers
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Responses API Multi-Turn Streaming Session",
            }
        )

        try:
            wrapper = MaximOpenAIClient(self.client, logger=self.logger)

            # Turn 1
            headers1 = {
                "x-maxim-generation-name": "Responses API Streaming MT - Turn 1",
                "x-maxim-session-id": session.id,
            }
            with wrapper.responses.stream(
                model="gpt-4o",
                input="Briefly describe the benefits of unit testing.",
                extra_headers=headers1,
            ) as stream1:
                chunks1 = 0
                for event in stream1:
                    chunks1 += 1
                first_final = stream1.get_final_response()

            self.assertGreater(chunks1, 0)
            self.assertIsNotNone(getattr(first_final, "id", None))

            # Turn 2 (multi-turn) using previous_response_id, same session
            headers2 = {
                "x-maxim-generation-name": "Responses API Streaming MT - Turn 2",
                "x-maxim-session-id": session.id,
            }
            with wrapper.responses.stream(
                model="gpt-4o",
                input="Can you provide one concrete example to illustrate it?",
                previous_response_id=first_final.id,
                extra_headers=headers2,
            ) as stream2:
                chunks2 = 0
                for event in stream2:
                    chunks2 += 1
                second_final = stream2.get_final_response()

            self.assertGreater(chunks2, 0)
            self.assertIsNotNone(getattr(second_final, "id", None))
        finally:
            session.end()

    def test_non_streaming_responses_multiturn_with_session(self):
        # Create a session and pass its ID via headers
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Responses API Multi-Turn Non-Streaming Session",
            }
        )

        try:
            wrapper = MaximOpenAIClient(self.client, logger=self.logger)

            headers = {
                "x-maxim-generation-name": "Responses API Non-Streaming MT",
                "x-maxim-session-id": session.id,
            }

            # Turn 1
            first = wrapper.responses.create(
                model="gpt-4o",
                input="What is JSON?",
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            # Turn 2
            second = wrapper.responses.create(
                model="gpt-4o",
                input="Give a tiny example JSON object.",
                previous_response_id=first.id,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(second, "id", None))
        finally:
            session.end()

    def test_responses_tool_call_flow_with_session(self):
        # Create a session and pass its ID via headers
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Responses API Tool Call Session",
            }
        )

        try:
            wrapper = MaximOpenAIClient(self.client, logger=self.logger)

            headers = {
                "x-maxim-generation-name": "Responses API Tool Call",
                "x-maxim-session-id": session.id,
            }

            # Define a function tool
            tools = [
                FunctionToolParam(
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City, State (e.g., San Francisco, CA)",
                                }
                            },
                            "additionalProperties": {
                                "type": "string",
                                "description": "Any other arbitrary key-value pairs as strings.",
                            },
                            "required": ["location"],
                        },
                        "strict": True,
                        "type": "function",
                    }
                )
            ]

            # Turn 1: ask a question that likely triggers a tool call
            first = wrapper.responses.create(
                model="gpt-4o",
                input="What's the weather in New York, NY right now?",
                tools=tools,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            # Extract tool calls, if any
            tool_calls = []
            output = getattr(first, "output", None)
            if output:
                for item in output:
                    if getattr(item, "type", None) == "function_call":
                        tool_calls.append(item)

            if tool_calls:
                tc = tool_calls[0]
                # Parse arguments
                try:
                    args = json.loads(getattr(tc, "arguments", "{}"))
                except Exception:
                    args = {}

                # Simulate a simple tool result
                location = args.get("location", "Unknown")
                tool_result = json.dumps(
                    {"location": location, "temperature": 70, "unit": "F"}
                )

                # Turn 2: submit tool result
                second = wrapper.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": tc.call_id,
                            "output": tool_result,
                        }
                    ],
                    previous_response_id=getattr(first, "id", None),
                    extra_headers=headers,
                )
                self.assertIsNotNone(getattr(second, "id", None))
        finally:
            session.end()

    def test_non_streaming_responses_multiturn_with_messages_array_session(self):
        # Create a session and pass its ID via headers
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Responses API Multi-Turn with Messages Array",
            }
        )

        try:
            wrapper = MaximOpenAIClient(self.client, logger=self.logger)

            headers = {
                "x-maxim-generation-name": "Responses API Non-Streaming MT (Messages Array)",
                "x-maxim-session-id": session.id,
            }

            # Turn 1
            first = wrapper.responses.create(
                model="gpt-4o",
                input="Tell me a fun fact about the moon.",
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            first_text = getattr(first, "output_text", "")

            # Turn 2: pass an array of messages instead of previous_response_id
            # Using EasyInputMessageParam shape (role + content string)
            conversation = [
                {"role": "user", "content": "Tell me a fun fact about the moon."},
                {"role": "assistant", "content": first_text},
                {"role": "user", "content": "Thanks. Can you format it as 2 bullet points?"},
            ]

            second = wrapper.responses.create(
                model="gpt-4o",
                input=conversation,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(second, "id", None))
        finally:
            session.end()


if __name__ == "__main__":
    unittest.main()
