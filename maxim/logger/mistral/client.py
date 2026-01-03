from typing import Any, List, Optional, Tuple
from uuid import uuid4
import json

from mistralai.chat import Chat
from mistralai.models import CompletionEvent
from mistralai.sdk import Mistral

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import MistralUtils

_MISTRAL_INFLIGHT_TOOL_CALLS: dict[str, Any] = {}


class MaximMistralChat:
    def __init__(self, chat: Chat, logger: Logger):
        self._chat = chat
        self._logger = logger

    def _log_tool_calls(self, trace: Optional[Trace], response: Any) -> None:
        """Create Maxim ToolCall spans on the trace for any Mistral tool calls.

        This mirrors the behavior we have for Groq/LiteLLM: we do *not* change
        the generation result format, we only add separate ToolCall entities on
        the trace when tools are used.
        """
        if trace is None:
            return

        try:
            tool_calls: List[Any] = []

            # Primary path: Mistral ChatCompletionResponse object
            choices = getattr(response, "choices", None)
            if choices:
                first_choice = choices[0]
                message = getattr(first_choice, "message", None)
                if message is not None:
                    tc_attr = getattr(message, "tool_calls", None)
                    if isinstance(tc_attr, list):
                        tool_calls = tc_attr

            # Fallback path: dict-like response structure
            if not tool_calls and isinstance(response, dict):
                choices_dict = response.get("choices") or []
                if choices_dict:
                    first_choice_dict = choices_dict[0] or {}
                    message_dict = first_choice_dict.get("message") or {}
                    tc_dict = message_dict.get("tool_calls")
                    if isinstance(tc_dict, list):
                        tool_calls = tc_dict

            if not tool_calls:
                return

            for tc in tool_calls:
                # tc may be an SDK object or a plain dict
                if hasattr(tc, "function"):
                    fn = getattr(tc, "function", None)
                    tool_call_id = getattr(tc, "id", None)
                elif isinstance(tc, dict):
                    fn = tc.get("function")
                    tool_call_id = tc.get("id")
                else:
                    continue

                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = str(uuid4())

                if fn is not None and hasattr(fn, "name"):
                    tool_name = getattr(fn, "name", "unknown")
                elif isinstance(fn, dict):
                    tool_name = fn.get("name", "unknown")
                else:
                    tool_name = "unknown"

                if fn is not None and hasattr(fn, "arguments"):
                    tool_args = getattr(fn, "arguments", "")
                elif isinstance(fn, dict):
                    tool_args = fn.get("arguments", "")
                else:
                    tool_args = ""

                tool_call_entity = trace.tool_call(
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "description": "Mistral tool call",
                        "args": tool_args,
                    }
                )
                _MISTRAL_INFLIGHT_TOOL_CALLS[tool_call_id] = tool_call_entity
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error creating tool_call entities: {e}"
            )

    def _complete_tool_results_from_messages(self, messages: Any) -> None:
        """
        Complete Maxim ToolCall entities using OpenAI-style tool messages.

        We look for messages of the form:
        {"role": "tool", "tool_call_id": "...", "content": ...}
        and call tool_call.result(...) with the concrete tool output.
        """
        if messages is None:
            return
        if not isinstance(messages, list):
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            raw_content = msg.get("content")
            if raw_content is None:
                result_str = ""
            elif isinstance(raw_content, str):
                result_str = raw_content
            else:
                try:
                    result_str = json.dumps(raw_content)
                except Exception:
                    result_str = str(raw_content)

            tool_call = _MISTRAL_INFLIGHT_TOOL_CALLS.get(tool_call_id)
            if tool_call is None:
                continue
            try:
                tool_call.result(result_str)
                _MISTRAL_INFLIGHT_TOOL_CALLS.pop(tool_call_id, None)
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][MaximMistralChat] Error setting tool_call result for id={tool_call_id}: {e}"
                )

    def _setup_logging(
        self,
        model: Optional[str],
        messages: Any,
        trace_id: Optional[str],
        generation_name: Optional[str],
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[Trace], Optional[Generation]]:
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        trace_config: dict[str, Any] = {"id": final_trace_id}
        if session_id is not None:
            trace_config["session_id"] = session_id
        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace(trace_config)
            generation = trace.generation(
                {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "mistral",
                    "name": generation_name,
                    "model_parameters": MistralUtils.get_model_params(**kwargs),
                    "messages": MistralUtils.parse_message_param(messages),
                }
            )
            input_message = None
            if messages:
                for message in messages:
                    content = message.get("content", None)
                    if content is None:
                        continue
                    if isinstance(content, str):
                        input_message = content
                        break
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                input_message = item.get("text", "")
                                break
            if input_message is not None:
                trace.set_input(input_message)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in generating content: {e}"
            )

        return is_local_trace, trace, generation

    def _finalize_logging(
        self,
        response: Any,
        is_local_trace: bool,
        trace: Optional[Trace],
        generation: Optional[Generation],
    ) -> None:
        try:
            if generation is not None:
                generation.result(MistralUtils.parse_completion(response))
            # Create ToolCall entities on the trace when tools are used.
            self._log_tool_calls(trace, response)
            if is_local_trace and trace is not None:
                if getattr(response, "choices", None):
                    text = MistralUtils._message_content(response.choices[0].message)
                    trace.set_output(text)
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in logging generation: {e}"
            )

    def complete(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        session_id = kwargs.pop("session_id", None)
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        self._complete_tool_results_from_messages(messages)

        # Create a copy of kwargs without model and messages to avoid conflicts
        logging_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }

        is_local_trace, trace, generation = self._setup_logging(
            model,
            messages,
            trace_id,
            generation_name,
            session_id=session_id,
            **logging_kwargs,
        )

        response = self._chat.complete(*args, **kwargs)

        self._finalize_logging(response, is_local_trace, trace, generation)

        return response

    def stream(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        self._complete_tool_results_from_messages(messages)

        # Create a copy of kwargs without model and messages to avoid conflicts
        logging_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }

        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace({"id": final_trace_id})
            generation = trace.generation(
                {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "mistral",
                    "name": generation_name,
                    "model_parameters": MistralUtils.get_model_params(**logging_kwargs),
                    "messages": MistralUtils.parse_message_param(messages),
                }
            )
            input_message = None
            if messages:
                for message in messages:
                    content = message.get("content", None)
                    if content is None:
                        continue
                    if isinstance(content, str):
                        input_message = content
                        break
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                input_message = item.get("text", "")
                                break
            if input_message is not None:
                trace.set_input(input_message)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in generating content: {e}"
            )

        stream = self._chat.stream(*args, **kwargs)
        chunks: List[dict] = []
        for event in stream:
            if isinstance(event, CompletionEvent):
                chunks.append(MistralUtils.parse_stream_response(event))
            yield event

        try:
            if generation is not None:
                generation.result(MistralUtils.combine_chunks(chunks))
            if is_local_trace and trace is not None:
                text = "".join(
                    chunk.get("delta", {}).get("content", "")
                    for c in chunks
                    for chunk in c.get("choices", [])
                )
                trace.set_output(text)
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in logging generation: {e}"
            )

    async def complete_async(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        self._complete_tool_results_from_messages(messages)

        # Create a copy of kwargs without model and messages to avoid conflicts
        logging_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }

        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace({"id": final_trace_id})
            generation = trace.generation(
                {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "mistral",
                    "name": generation_name,
                    "model_parameters": MistralUtils.get_model_params(**logging_kwargs),
                    "messages": MistralUtils.parse_message_param(messages),
                }
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in generating content: {e}"
            )

        response = await self._chat.complete_async(*args, **kwargs)

        try:
            if generation is not None:
                generation.result(MistralUtils.parse_completion(response))
            # Create ToolCall entities on the trace when tools are used.
            self._log_tool_calls(trace, response)
            if is_local_trace and trace is not None:
                if response.choices:
                    text = MistralUtils._message_content(response.choices[0].message)
                    trace.set_output(text)
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in logging generation: {e}"
            )

        return response

    async def stream_async(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        self._complete_tool_results_from_messages(messages)

        # Create a copy of kwargs without model and messages to avoid conflicts
        logging_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }

        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace({"id": final_trace_id})
            generation = trace.generation(
                {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "mistral",
                    "name": generation_name,
                    "model_parameters": MistralUtils.get_model_params(**logging_kwargs),
                    "messages": MistralUtils.parse_message_param(messages),
                }
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in generating content: {e}"
            )

        stream = await self._chat.stream_async(*args, **kwargs)
        chunks: List[dict] = []
        async for event in stream:
            if isinstance(event, CompletionEvent):
                chunks.append(MistralUtils.parse_stream_response(event))
            yield event

        try:
            if generation is not None:
                generation.result(MistralUtils.combine_chunks(chunks))
            if is_local_trace and trace is not None:
                text = "".join(
                    chunk.get("delta", {}).get("content", "")
                    for c in chunks
                    for chunk in c.get("choices", [])
                )
                trace.set_output(text)
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in logging generation: {e}"
            )


class MaximMistralClient:
    def __init__(self, client: Mistral, logger: Logger):
        self._client = client
        self._logger = logger

    @property
    def chat(self) -> MaximMistralChat:
        return MaximMistralChat(self._client.chat, self._logger)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_client", "_logger"}:
            super().__setattr__(name, value)
        else:
            setattr(self._client, name, value)

    def __enter__(self) -> "MaximMistralClient":
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> "MaximMistralClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
