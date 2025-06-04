from typing import Any, List, Optional
from uuid import uuid4

from mistralai.sdk import Mistral
from mistralai.chat import Chat
from mistralai.models import CompletionEvent

from ...scribe import scribe
from ..logger import Generation, GenerationConfig, Logger, Trace, TraceConfig
from .utils import MistralUtils


class MaximMistralChat:
    def __init__(self, chat: Chat, logger: Logger):
        self._chat = chat
        self._logger = logger

    def complete(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="mistral",
                name=generation_name,
                model_parameters=MistralUtils.get_model_params(**kwargs),
                messages=MistralUtils.parse_message_param(messages),
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximMistralChat] Error in generating content: {e}"
            )

        response = self._chat.complete(*args, **kwargs)

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

    def stream(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        generation_name = kwargs.pop("generation_name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="mistral",
                name=generation_name,
                model_parameters=MistralUtils.get_model_params(**kwargs),
                messages=MistralUtils.parse_message_param(messages),
            )
            generation = trace.generation(gen_config)
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
        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="mistral",
                name=generation_name,
                model_parameters=MistralUtils.get_model_params(**kwargs),
                messages=MistralUtils.parse_message_param(messages),
            )
            generation = trace.generation(gen_config)
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
        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="mistral",
                name=generation_name,
                model_parameters=MistralUtils.get_model_params(**kwargs),
                messages=MistralUtils.parse_message_param(messages),
            )
            generation = trace.generation(gen_config)
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
