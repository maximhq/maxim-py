"""
E2E tests: raw OpenAI ``responses.create`` with Responses-format ``input`` and
manual Maxim logging (``trace.generation`` + ``generation.result``).

Requires ``OPENAI_API_KEY``, ``MAXIM_API_KEY``, and ``MAXIM_LOG_REPO_ID``.
Optional: ``MAXIM_BASE_URL`` (same as ``test_openai_responses_integration.py``).
Reasoning-summary tests use ``REASONING_MODEL`` (see module constant).

These tests complement ``test_openai_responses_integration.py`` (wrapped client
+ header-driven traces) by exercising the same API with explicit trace/generation
lifecycle and Responses-shaped ``messages`` on the generation config (including
normalized multi-turn history on the logged generation even when the API call
uses ``previous_response_id``).
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Optional
from uuid import uuid4

import dotenv
from openai import APIStatusError, OpenAI
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.shared import Reasoning

from maxim import Maxim
from maxim.logger import Trace


dotenv.load_dotenv()


def _responses_output_text(response: object) -> Optional[str]:
    """Read ``output_text`` from an OpenAI Responses SDK object (no openai package __init__ import)."""
    try:
        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text
    except Exception:
        pass
    return None


def _weather_tool() -> list[FunctionToolParam]:
    return [
        FunctionToolParam(
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and region (e.g. San Francisco, CA)",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
            }
        )
    ]


def _weather_tools_plain_dicts() -> list[dict]:
    """JSON-serializable tool defs for ``model_parameters`` (same schema as API ``tools``)."""
    return [dict(t) for t in _weather_tool()]


def _responses_function_call_input_dict(fc: object) -> dict:
    """Map a Responses ``output`` function_call item to an input-shaped dict for logged ``messages``."""
    name = getattr(fc, "name", None) or ""
    arguments = getattr(fc, "arguments", None)
    if not isinstance(arguments, str):
        arguments = "{}" if arguments is None else json.dumps(arguments)
    call_id = getattr(fc, "call_id", None) or ""
    out: dict = {
        "type": "function_call",
        "name": name,
        "arguments": arguments,
        "call_id": call_id,
    }
    fc_id = getattr(fc, "id", None)
    if fc_id:
        out["id"] = fc_id
    return out


_E2E_REQUIRES = (
    os.getenv("OPENAI_API_KEY")
    and os.getenv("MAXIM_API_KEY")
    and os.getenv("MAXIM_LOG_REPO_ID")
)

REASONING_MODEL = "o3"


def _monty_answer_plausible(lowered: str) -> bool:
    return bool(
        "switch" in lowered
        or "switching" in lowered
        or "2/3" in lowered
        or "two-thirds" in lowered
        or "two thirds" in lowered
        or "probability" in lowered
        or "1/3" in lowered
        or "one-third" in lowered
        or "one third" in lowered
        or "66" in lowered
        or "67" in lowered
    )


def _skip_if_model_unavailable(exc: APIStatusError, model: str) -> None:
    body = f"{exc.message} {exc.body}"
    if exc.status_code in (400, 404) and (
        "model" in body.lower()
        or "does not exist" in body.lower()
        or "invalid_model" in body.lower()
    ):
        raise unittest.SkipTest(
            f"Skipping reasoning test: model {model!r} unavailable for this key ({body})"
        ) from exc


def _combined_visible_text(response: object) -> str:
    """``output_text`` plus reasoning summary blocks (lowercased, for loose assertions)."""
    parts: list[str] = []
    ot = _responses_output_text(response)
    if isinstance(ot, str) and ot.strip():
        parts.append(ot)
    out = getattr(response, "output", None) or []
    for item in out:
        if getattr(item, "type", None) == "reasoning":
            for block in getattr(item, "summary", None) or []:
                t = getattr(block, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t)
    return " ".join(parts).lower()


@unittest.skipUnless(_E2E_REQUIRES, "OPENAI_API_KEY, MAXIM_API_KEY, and MAXIM_LOG_REPO_ID required")
class TestOpenAIResponsesManualLoggingE2E(unittest.TestCase):
    """Raw OpenAI Responses API + manual ``trace.generation`` / ``generation.result``."""

    def setUp(self) -> None:
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
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def tearDown(self) -> None:
        if self.logger:
            self.logger.flush()

    def _reasoning_stream_attached_to_trace(
        self,
        trace: Trace,
        generation_name: str,
        user_prompt: str,
    ) -> object:
        """One generation on ``trace``: stream with reasoning, then ``generation.result(final)``.

        Thinking and assistant text are merged in the logged payload via
        ``parse_result`` / ``inject_thinking_into_openai_response_result`` on the
        same generation (no second trace).
        """
        responses_input: list[dict] = [
            {"type": "message", "role": "user", "content": user_prompt},
        ]
        generation = trace.generation(
            {
                "id": str(uuid4()),
                "model": REASONING_MODEL,
                "provider": "openai",
                "name": generation_name,
                "messages": responses_input,
                "model_parameters": {
                    "reasoning": {"effort": "medium", "summary": "detailed"},
                },
            }
        )
        try:
            with self.client.responses.stream(
                model=REASONING_MODEL,
                input=user_prompt,
                include=["reasoning.encrypted_content"],
                reasoning=Reasoning(effort="medium", summary="detailed"),
            ) as stream:
                for _ in stream:
                    pass
                response = stream.get_final_response()
        except APIStatusError as e:
            _skip_if_model_unavailable(e, REASONING_MODEL)
            raise

        generation.result(response)
        output_text = _responses_output_text(response) or ""
        if output_text:
            trace.set_output(output_text)
        return response

    def test_manual_single_turn_typed_message_input(self) -> None:
        """Responses ``input`` with ``type: message`` items; generation.messages normalize."""
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-typed-message",
            }
        )
        responses_input: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": "Reply with exactly the word: OK.",
            },
        ]
        generation = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-single-turn",
                "messages": responses_input,
            }
        )

        for msg in generation.messages:
            self.assertNotIn("type", msg)

        response = self.client.responses.create(model="gpt-4o-mini", input="Reply with exactly the word: OK.")
        self.assertEqual(getattr(response, "object", None), "response")
        self.assertIsNotNone(getattr(response, "id", None))
        generation.result(response)

        out = _responses_output_text(response)
        if isinstance(out, str) and out:
            trace.set_output(out)
        trace.end()

    def test_manual_multi_turn_responses_shaped_history(self) -> None:
        """Two turns: turn 1 typed ``message`` input; turn 2 continues via ``previous_response_id``.

        On Python 3.9, passing a typed ``message`` array to the OpenAI SDK can raise
        ``TypeError: unsupported operand type(s) for |: 'ModelMetaclass' and 'ModelMetaclass'``.
        So the OpenAI calls use plain string ``input`` while the manual Maxim generations
        still receive Responses-format ``messages`` arrays.
        """
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-multiturn",
            }
        )

        first_input: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": "Give one short fact about Jupiter (one sentence).",
            },
        ]
        gen1 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-mt-turn1",
                "messages": first_input,
            }
        )
        r1 = self.client.responses.create(
            model="gpt-4o-mini",
            input="Give one short fact about Jupiter (one sentence).",
        )
        self.assertEqual(getattr(r1, "object", None), "response")
        gen1.result(r1)

        first_text = getattr(r1, "output_text", None) or ""
        self.assertIsInstance(first_text, str)

        # Logged messages for gen2: user + assistant summary (what we would hand to Maxim).
        second_messages: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": "Give one short fact about Jupiter (one sentence).",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": first_text,
            },
            {
                "type": "message",
                "role": "user",
                "content": "Now answer with only: ACK.",
            },
        ]
        gen2 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-mt-turn2",
                "messages": second_messages,
            }
        )
        for msg in gen2.messages:
            self.assertNotIn("type", msg)

        # String input avoids Python 3.9 + OpenAI SDK ``TypeError`` on ``ModelMetaclass | ...``
        # when passing a list of typed message dicts to ``input``.
        r2 = self.client.responses.create(
            model="gpt-4o-mini",
            input="Now answer with only: ACK.",
            previous_response_id=getattr(r1, "id", None),
        )
        self.assertEqual(getattr(r2, "object", None), "response")
        gen2.result(r2)

        tail = _responses_output_text(r2)
        if isinstance(tail, str) and tail:
            trace.set_output(tail)
        trace.end()

    def test_manual_tool_follow_up_two_generations(self) -> None:
        """Tool call round: first generation logs call; second logs tool output follow-up."""
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-tool",
            }
        )

        first_input: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": "What is the weather in Boston, MA right now? Use the tool.",
            },
        ]
        gen1 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-tool-turn1",
                "messages": first_input,
                "model_parameters": {"tool_choice": "auto"},
            }
        )
        r1 = self.client.responses.create(
            model="gpt-4o-mini",
            input="What is the weather in Boston, MA right now? Use the tool.",
            tools=_weather_tool(),
        )
        self.assertEqual(getattr(r1, "object", None), "response")
        gen1.result(r1)

        tool_calls = []
        out = getattr(r1, "output", None) or []
        for item in out:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)

        if not tool_calls:
            trace.end()
            self.skipTest("Model did not emit a function_call; cannot exercise tool follow-up.")

        tc = tool_calls[0]
        try:
            args = json.loads(getattr(tc, "arguments", "{}") or "{}")
        except json.JSONDecodeError:
            args = {}
        location = args.get("location", "Boston, MA")
        tool_result = json.dumps(
            {"location": location, "temperature": 42, "unit": "F", "note": "stub"}
        )

        second_input: list[dict] = [
            {
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": tool_result,
            },
        ]
        gen2 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-tool-turn2",
                "messages": second_input,
            }
        )
        self.assertEqual(gen2.messages[0]["role"], "tool")

        r2 = self.client.responses.create(
            model="gpt-4o-mini",
            input=second_input,
            previous_response_id=getattr(r1, "id", None),
            tools=_weather_tool(),
        )
        self.assertEqual(getattr(r2, "object", None), "response")
        gen2.result(r2)

        final_text = _responses_output_text(r2)
        if isinstance(final_text, str) and final_text:
            trace.set_output(final_text)
        trace.end()

    def test_manual_messages_list_includes_function_call_and_tool_definitions(self) -> None:
        """Responses-format ``messages`` with user + function_call + function_call_output normalize correctly.

        Also checks ``model_parameters['tools']`` is preserved (stringified) so logged config
        still reflects tool schemas alongside the transcript.
        """
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-tool-full-messages",
            }
        )
        user_prompt = "What is the weather in Boston, MA right now? Use the tool."
        first_input: list[dict] = [
            {"type": "message", "role": "user", "content": user_prompt},
        ]
        gen1 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-tool-full-turn1",
                "messages": first_input,
                "model_parameters": {"tool_choice": "auto", "tools": _weather_tools_plain_dicts()},
            }
        )
        r1 = self.client.responses.create(
            model="gpt-4o-mini",
            input=user_prompt,
            tools=_weather_tool(),
        )
        self.assertEqual(getattr(r1, "object", None), "response")
        gen1.result(r1)

        tool_calls = []
        out = getattr(r1, "output", None) or []
        for item in out:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)
        if not tool_calls:
            trace.end()
            self.skipTest("Model did not emit a function_call; cannot exercise full tool messages.")

        tc = tool_calls[0]
        try:
            args = json.loads(getattr(tc, "arguments", "{}") or "{}")
        except json.JSONDecodeError:
            args = {}
        location = args.get("location", "Boston, MA")
        tool_result = json.dumps(
            {"location": location, "temperature": 42, "unit": "F", "note": "stub"}
        )

        full_messages: list[dict] = [
            {"type": "message", "role": "user", "content": user_prompt},
            _responses_function_call_input_dict(tc),
            {
                "type": "function_call_output",
                "call_id": getattr(tc, "call_id", None),
                "output": tool_result,
            },
        ]
        gen2 = trace.generation(
            {
                "id": str(uuid4()),
                "model": "gpt-4o-mini",
                "provider": "openai",
                "name": "manual-responses-tool-full-turn2",
                "messages": full_messages,
                "model_parameters": {
                    "tool_choice": "auto",
                    "tools": _weather_tools_plain_dicts(),
                },
            }
        )

        self.assertEqual(len(gen2.messages), 3)
        for msg in gen2.messages:
            self.assertNotIn("type", msg)
        self.assertEqual(gen2.messages[0]["role"], "user")
        self.assertEqual(gen2.messages[0]["content"], user_prompt)
        self.assertEqual(gen2.messages[1]["role"], "assistant")
        self.assertIn("get_weather", gen2.messages[1]["content"])
        call_id = getattr(tc, "call_id", None)
        if isinstance(call_id, str) and call_id:
            self.assertIn(call_id, gen2.messages[1]["content"])
        self.assertEqual(gen2.messages[2]["role"], "tool")
        self.assertEqual(gen2.messages[2]["content"], tool_result)

        tools_logged = gen2.model_parameters.get("tools", "")
        self.assertIsInstance(tools_logged, str)
        self.assertIn("get_weather", tools_logged)

        r2 = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": getattr(tc, "call_id", None),
                    "output": tool_result,
                },
            ],
            previous_response_id=getattr(r1, "id", None),
            tools=_weather_tool(),
        )
        self.assertEqual(getattr(r2, "object", None), "response")
        gen2.result(r2)

        final_text = _responses_output_text(r2)
        if isinstance(final_text, str) and final_text:
            trace.set_output(final_text)
        trace.end()

    def test_manual_reasoning_water_jug_puzzle(self) -> None:
        """One trace, one generation: streamed reasoning + answer; thinking merged in logged result."""
        user_prompt = (
            "You have an empty 3-gallon jug and an empty 5-gallon jug. "
            "Unlimited water. Explain step by step how to end up with exactly 4 gallons "
            "in one container, then state which container holds the 4 gallons. Give me the reasoning summary."
        )
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-reasoning-water-jug",
            }
        )
        response = self._reasoning_stream_attached_to_trace(
            trace,
            generation_name="manual-responses-reasoning-water-jug",
            user_prompt=user_prompt,
        )
        self.assertEqual(getattr(response, "object", None), "response")
        self.assertTrue((_responses_output_text(response) or "").strip())
        trace.end()

    def test_manual_reasoning_monty_hall_style(self) -> None:
        """One trace, one generation: Monty Hall with reasoning; answer plausibility from combined output."""
        user_prompt = (
            "Three doors: behind one is a car, behind the other two are goats. "
            "You pick door 1. The host, who knows what's behind each door, opens door 3 "
            "and shows a goat. You may switch to door 2 or stay on door 1. "
            "Should you switch or stay to maximize the chance of winning the car? "
            "Reason through the probabilities briefly, then give a clear recommendation. Give me the reasoning summary."
        )
        trace = self.logger.trace(
            {
                "id": str(uuid4()),
                "name": "e2e-manual-responses-reasoning-monty",
            }
        )
        response = self._reasoning_stream_attached_to_trace(
            trace,
            generation_name="manual-responses-reasoning-monty",
            user_prompt=user_prompt,
        )
        self.assertEqual(getattr(response, "object", None), "response")
        self.assertTrue((_responses_output_text(response) or "").strip())
        self.assertTrue(
            _monty_answer_plausible(_combined_visible_text(response)),
            "Expected the model answer to discuss switching or probabilities",
        )
        trace.end()


if __name__ == "__main__":
    unittest.main()
