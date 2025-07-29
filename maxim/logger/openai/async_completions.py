from typing import AsyncIterator, List, Optional
from uuid import uuid4

from openai import AsyncOpenAI
from openai.resources.chat import AsyncCompletions
from openai.types.chat import ChatCompletionChunk
from typing_extensions import override

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import OpenAIUtils


class AsyncStreamWrapper:
    """
    Async wrapper for OpenAI streaming chat completions that handles Maxim logging.

    This class wraps an async OpenAI stream to automatically log generation results
    and trace information when the stream is fully consumed. It accumulates chunks
    as they are yielded and processes the complete response for logging when the
    stream ends.
    """

    def __init__(
        self,
        stream: AsyncIterator[ChatCompletionChunk],
        generation: Optional[Generation],
        trace: Optional[Trace],
        is_local_trace: bool,
    ) -> None:
        self._stream = stream
        self._generation = generation
        self._trace = trace
        self._is_local_trace = is_local_trace
        self._chunks: List[ChatCompletionChunk] = []
        self._consumed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._chunks.append(chunk)
            return chunk
        except StopAsyncIteration:
            if not self._consumed:
                self._consumed = True
                try:
                    if self._generation is not None and self._chunks:
                        # Create a combined response from all chunks
                        combined_response = OpenAIUtils.parse_completion_from_chunks(
                            self._chunks
                        )
                        self._generation.result(combined_response)
                    if self._is_local_trace and self._trace is not None:
                        # Extract combined text from chunks
                        combined_text = "".join(
                            choice.delta.content or ""
                            for chunk in self._chunks
                            for choice in chunk.choices
                            if hasattr(choice.delta, "content")
                        )
                        self._trace.set_output(combined_text)
                        self._trace.end()
                except Exception as e:
                    if self._generation is not None:
                        self._generation.error({ "message": str(e) })
                    scribe().warning(
                        f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in logging stream completion: {str(e)}"
                    )
            raise


class MaximAsyncOpenAIChatCompletions(AsyncCompletions):
    def __init__(self, client: AsyncOpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    @override
    async def create(self, *args, **kwargs):
        metadata = kwargs.get("metadata", None)
        trace_id = None
        generation_name = None
        maxim_metadata = None
        if metadata is not None:
            maxim_metadata = metadata.get("maxim", None)
            if maxim_metadata is not None:
                trace_id = maxim_metadata.get("trace_id", None)
                generation_name = maxim_metadata.get("generation_name", None)
        is_local_trace = trace_id is None
        model = kwargs.get("model", None)
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        messages = kwargs.get("messages", None)
        is_streaming = kwargs.get("stream", False)

        # Add stream_options with include_usage if not present
        if is_streaming and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
        elif is_streaming and "include_usage" not in kwargs.get("stream_options", {}):
            kwargs["stream_options"]["include_usage"] = True

        try:
            trace = self._logger.trace({"id": final_trace_id})
            gen_config = {
                "id": str(uuid4()),
                "model": model,
                "provider": "openai",
                "name": generation_name,
                "model_parameters": OpenAIUtils.get_model_params(**kwargs),
                "messages": OpenAIUtils.parse_message_param(messages or []),
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in generating content: {str(e)}"
            )

        try:
            response = await super().create(*args, **kwargs)
        except Exception as e:
            if generation is not None:
                generation.error({ "message": str(e) })
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in generating content: {str(e)}"
            )
            # Will raise the error to the caller to handle
            raise

        if is_streaming:
            # For streaming responses, return a wrapped stream that handles logging
            return AsyncStreamWrapper(response, generation, trace, is_local_trace)

            # For non-streaming responses, log immediately
        try:
            if generation is not None:
                generation.result(OpenAIUtils.parse_completion(response))
            if is_local_trace and trace is not None:
                trace.set_output(response.choices[0].message.content or "")
                trace.end()
        except Exception as e:
            if generation is not None:
                generation.error({ "message": str(e) })
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in logging generation: {str(e)}"
            )

        return response
