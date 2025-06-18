from typing import Any, List, Optional, Tuple
from uuid import uuid4

from mistralai.chat import Chat
from mistralai.models import CompletionEvent
from mistralai.sdk import Mistral

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import MistralUtils


class MaximMistralChat:
    def __init__(self, chat: Chat, logger: Logger):
        self._chat = chat
        self._logger = logger

    def _setup_logging(
        self,
        model: Optional[str],
        messages: Any,
        trace_id: Optional[str],
        generation_name: Optional[str],
        **kwargs: Any,
    ) -> Tuple[bool, Optional[Trace], Optional[Generation]]:
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
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
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        # Create a copy of kwargs without model and messages to avoid conflicts
        logging_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }

        is_local_trace, trace, generation = self._setup_logging(
            model, messages, trace_id, generation_name, **logging_kwargs
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
