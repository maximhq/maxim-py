from typing import Optional
from uuid import uuid4

from openai import OpenAI
from openai.resources.chat import Completions
from typing_extensions import override

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import OpenAIUtils


class StreamWrapper:
    def __init__(self, stream, generation, trace, is_local_trace):
        self._stream = stream
        self._generation = generation
        self._trace = trace
        self._is_local_trace = is_local_trace
        self._chunks = []
        self._consumed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._chunks.append(chunk)
            return chunk
        except StopIteration:
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
                        f"[MaximSDK][MaximOpenAIChatCompletions] Error in logging stream completion: {str(e)}"
                    )
            raise


class MaximOpenAIChatCompletions(Completions):
    def __init__(self, client: OpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    @override
    def parse(self, *args, **kwargs):
        extra_headers = kwargs.get("extra_headers", None)
        trace_id = None
        generation_name = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
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
                "messages": OpenAIUtils.parse_message_param(messages),
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIChatCompletions] Error in generating content: {str(e)}"
            )
        try:
            response = super().parse(*args, **kwargs)
        except Exception as e:
            if generation is not None:
                generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][MaximOpenAIChatCompletions] Error in parsing content: {str(e)}"
            )
            raise
        if is_streaming:
            # For streaming responses, return a wrapped stream that handles logging
            return StreamWrapper(response, generation, trace, is_local_trace)
        else:
            # For non-streaming responses, log immediately
            try:
                if generation is not None:
                    result = OpenAIUtils.parse_completion(response)
                    generation.result(result)
                if is_local_trace and trace is not None:
                    trace.set_output(response.choices[0].message.content or "")
                    trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][MaximOpenAIChatCompletions] Error in logging generation: {str(e)}"
                )
        return response

    @override
    def create(self, *args, **kwargs):
        extra_headers = kwargs.get("extra_headers", None)
        trace_id = None
        generation_name = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
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
                "messages": OpenAIUtils.parse_message_param(messages),
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIChatCompletions] Error in generating content: {str(e)}"
            )

        try:
            response = super().create(*args, **kwargs)
        except Exception as e:
            if generation is not None:
                generation.error({ "message": str(e) })
            scribe().warning(
                f"[MaximSDK][MaximOpenAIChatCompletions] Error in generating content: {str(e)}"
            )
            # Will raise the error to the caller to handle
            raise

        if is_streaming:
            # For streaming responses, return a wrapped stream that handles logging
            return StreamWrapper(response, generation, trace, is_local_trace)
        else:
            # For non-streaming responses, log immediately
            try:
                if generation is not None:
                    result = OpenAIUtils.parse_completion(response)
                    generation.result(result)
                if is_local_trace and trace is not None:
                    trace.set_output(response.choices[0].message.content or "")
                    trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({ "message": str(e) })
                scribe().warning(
                    f"[MaximSDK][MaximOpenAIChatCompletions] Error in logging generation: {str(e)}"
                )

        return response
