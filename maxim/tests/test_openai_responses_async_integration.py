import os
import inspect
import json
import unittest
from uuid import uuid4

import dotenv
from openai import AsyncOpenAI
from openai.types.responses.function_tool_param import FunctionToolParam

from maxim import Maxim
from maxim.logger.openai import MaximOpenAIAsyncClient


dotenv.load_dotenv()


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestOpenAIResponsesAsyncIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.wrapper = MaximOpenAIAsyncClient(self.async_client, logger=self.logger)

    async def asyncTearDown(self):
        # Close AsyncOpenAI client to avoid leaking async HTTP connections
        try:
            client = getattr(self, "async_client", None)
            if client is not None:
                if hasattr(client, "aclose") and callable(getattr(client, "aclose")):
                    await client.aclose()
                elif hasattr(client, "close") and callable(getattr(client, "close")):
                    close_method = getattr(client, "close")
                    if inspect.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        # Fallback to sync close if provided
                        close_method()
        except Exception:
            # Best-effort; don't fail teardown for close errors
            pass

        if self.logger:
            self.logger.flush()

    async def test_async_streaming_responses_logging(self):
        headers = {
            "x-maxim-generation-name": "Async Responses API Streaming Test",
        }

        async with self.wrapper.responses.stream(
            model="gpt-4o",
            input="Write a 1-2 sentence haiku about autumn.",
            extra_headers=headers,
        ) as stream:
            chunks = 0
            text_accum = []
            async for event in stream:
                chunks += 1
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    text_accum.append(delta)

            final_response = await stream.get_final_response()
            print(
                f"\n\n======\nFINAL RESPONSE: {final_response.output_text}\n======\n\n"
            )
            self.assertGreater(chunks, 0)
            combined = "".join(text_accum)
            self.assertIsInstance(combined, str)

    async def test_async_non_streaming_responses_logging(self):
        headers = {
            "x-maxim-generation-name": "Async Responses API Non-Streaming Test",
        }

        response = await self.wrapper.responses.create(
            model="gpt-4o",
            input="Explain JSON in one sentence.",
            extra_headers=headers,
        )

        self.assertIsNotNone(getattr(response, "id", None))
        status = getattr(response, "status", None)
        self.assertIsInstance(status, str)
        output_text = getattr(response, "output_text", None)
        self.assertIsInstance(output_text, str)

    async def test_async_streaming_responses_multiturn_with_session(self):
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Async Responses API Multi-Turn Streaming Session",
            }
        )

        try:
            headers1 = {
                "x-maxim-generation-name": "Async Responses API Streaming MT - Turn 1",
                "x-maxim-session-id": session.id,
            }
            async with self.wrapper.responses.stream(
                model="gpt-4o",
                input="Briefly describe the benefits of unit testing.",
                extra_headers=headers1,
            ) as stream1:
                chunks1 = 0
                async for _ in stream1:
                    chunks1 += 1
                first_final = await stream1.get_final_response()

            self.assertGreater(chunks1, 0)
            self.assertIsNotNone(getattr(first_final, "id", None))

            headers2 = {
                "x-maxim-generation-name": "Async Responses API Streaming MT - Turn 2",
                "x-maxim-session-id": session.id,
            }
            async with self.wrapper.responses.stream(
                model="gpt-4o",
                input="Can you provide one concrete example to illustrate it?",
                previous_response_id=first_final.id,
                extra_headers=headers2,
            ) as stream2:
                chunks2 = 0
                async for _ in stream2:
                    chunks2 += 1
                second_final = await stream2.get_final_response()

            self.assertGreater(chunks2, 0)
            self.assertIsNotNone(getattr(second_final, "id", None))
        finally:
            session.end()

    async def test_async_non_streaming_responses_multiturn_with_session(self):
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Async Responses API Multi-Turn Non-Streaming Session",
            }
        )

        try:
            headers = {
                "x-maxim-generation-name": "Async Responses API Non-Streaming MT",
                "x-maxim-session-id": session.id,
            }

            first = await self.wrapper.responses.create(
                model="gpt-4o",
                input="What is JSON?",
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            second = await self.wrapper.responses.create(
                model="gpt-4o",
                input="Give a tiny example JSON object.",
                previous_response_id=first.id,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(second, "id", None))
        finally:
            session.end()

    async def test_async_responses_tool_call_flow_with_session(self):
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Async Responses API Tool Call Session",
            }
        )

        try:
            headers = {
                "x-maxim-generation-name": "Async Responses API Tool Call",
                "x-maxim-session-id": session.id,
            }

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

            first = await self.wrapper.responses.create(
                model="gpt-4o",
                input="What's the weather in New York, NY right now?",
                tools=tools,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            tool_calls = []
            output = getattr(first, "output", None)
            if output:
                for item in output:
                    if getattr(item, "type", None) == "function_call":
                        tool_calls.append(item)

            if tool_calls:
                tc = tool_calls[0]
                try:
                    args = json.loads(getattr(tc, "arguments", "{}"))
                except Exception:
                    args = {}

                location = args.get("location", "Unknown")
                tool_result = json.dumps(
                    {"location": location, "temperature": 70, "unit": "F"}
                )

                second = await self.wrapper.responses.create(
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

    async def test_async_non_streaming_responses_multiturn_with_messages_array_session(
        self,
    ):
        session = self.logger.session(
            {
                "id": str(uuid4()),
                "name": "Async Responses API Multi-Turn with Messages Array",
            }
        )

        try:
            headers = {
                "x-maxim-generation-name": "Async Responses API Non-Streaming MT (Messages Array)",
                "x-maxim-session-id": session.id,
            }

            first = await self.wrapper.responses.create(
                model="gpt-4o",
                input="Tell me a fun fact about the moon.",
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(first, "id", None))

            first_text = getattr(first, "output_text", "")

            conversation = [
                {"role": "user", "content": "Tell me a fun fact about the moon."},
                {"role": "assistant", "content": first_text},
                {
                    "role": "user",
                    "content": "Thanks. Can you format it as 2 bullet points?",
                },
            ]

            second = await self.wrapper.responses.create(
                model="gpt-4o",
                input=conversation,
                extra_headers=headers,
            )
            self.assertIsNotNone(getattr(second, "id", None))
        finally:
            session.end()


if __name__ == "__main__":
    unittest.main()
