from typing import Any, Dict, Optional
from uuid import uuid4

from openai import OpenAI
from openai.resources.responses import Responses
from openai.lib.streaming.responses._responses import (
    ResponseStreamManager,
    ResponseStream,
)

from maxim.logger.components.generation import GenerationConfigDict

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import OpenAIUtils


class ResponsesIteratorWrapper(ResponseStreamManager):
    def __init__(
        self,
        manager: ResponseStreamManager,
        generation: Optional[Generation],
        trace: Optional[Trace],
        is_local_trace: bool,
    ):
        self._manager = manager
        self._generation = generation
        self._trace = trace
        self._is_local_trace = is_local_trace
        self._final_response = None
        self._stream: Optional[ResponseStream] = None

    def __enter__(self):
        # Enter the underlying manager to obtain the actual ResponseStream
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        try:
            get_final = getattr(self._stream, "get_final_response", None)
            if callable(get_final):
                self._final_response = get_final()

            if self._final_response is not None:
                try:
                    if self._generation is not None:
                        if exc_type is None:
                            self._generation.result(self._final_response)
                        else:
                            error_message = str(exc) if exc is not None else str(exc_type)
                            error_type = getattr(exc_type, "__name__", None) if exc_type is not None else None
                            self._generation.error({"message": error_message, "type": error_type})
                    if self._is_local_trace and self._trace is not None:
                        try:
                            output_text = OpenAIUtils.extract_responses_output_text(
                                self._final_response
                            )
                            if isinstance(output_text, str):
                                self._trace.set_output(output_text)
                        except Exception:
                            pass
                        self._trace.end()
                except Exception as e:
                    if self._generation is not None:
                        self._generation.error({"message": str(e)})
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIResponses] Error in logging streamed generation: {str(e)}"
                    )
        finally:
            # Defer to the underlying manager for cleanup
            return self._manager.__exit__(exc_type, exc, tb)

class MaximOpenAIResponses(Responses):
    def __init__(self, client: OpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    def _start_trace_and_generation(
        self,
        *,
        extra_headers: Optional[Dict[str, str]],
        model: Optional[str],
        messages,
        model_parameters: Dict[str, Any],
    ):
        trace_id = None
        generation_name = None
        session_id = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
            session_id = extra_headers.get("x-maxim-session-id", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())

        trace: Optional[Trace] = None
        generation: Optional[Generation] = None
        try:
            trace = self._logger.trace({"id": final_trace_id, "session_id": session_id})
            gen_config: GenerationConfigDict = {
                "id": str(uuid4()),
                "model": str(model),
                "provider": "openai",
                "name": generation_name,
                "model_parameters": model_parameters,
                "messages": messages,
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIResponses] Error starting trace/generation: {str(e)}"
            )

        return is_local_trace, trace, generation

    def _messages_from_input(self, input_value: Any):
        # Map Responses API input to a single user message when possible
        if input_value is None:
            return []
        try:
            if isinstance(input_value, str):
                return OpenAIUtils.parse_message_param(
                    [{"role": "user", "content": input_value}]
                )
            # For complex inputs (list/dict), coerce to string for a rough summary
            return OpenAIUtils.parse_message_param(
                [{"role": "user", "content": str(input_value)}]
            )
        except Exception:
            return []

    def create(self, *args, **kwargs):  # type: ignore[override]
        extra_headers = kwargs.get("extra_headers", None)
        model = kwargs.get("model", None)
        input_value = kwargs.get("input", None)

        messages = OpenAIUtils.parse_responses_input_to_messages(input_value)
        model_parameters = OpenAIUtils.get_responses_model_params(**kwargs)

        is_local_trace, trace, generation = self._start_trace_and_generation(
            extra_headers=extra_headers,
            model=model,
            messages=messages,
            model_parameters=model_parameters,
        )

        try:
            response = self._client.responses.create(*args, **kwargs)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIResponses] Error generating response: {str(e)}"
            )
            raise e

        try:
            if generation is not None:
                generation.result(response)
            if is_local_trace and trace is not None:
                try:
                    output_text = OpenAIUtils.extract_responses_output_text(response)
                    if isinstance(output_text, str):
                        trace.set_output(output_text)
                except Exception:
                    pass
                trace.end()
        except Exception as e:
            if generation is not None:
                generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][MaximOpenAIResponses] Error in logging generation: {str(e)}"
            )

        return response

    def stream(self, *args, **kwargs):
        extra_headers = kwargs.get("extra_headers", None)
        model = kwargs.get("model", None)
        input_value = kwargs.get("input", None)

        messages = OpenAIUtils.parse_responses_input_to_messages(input_value)
        model_parameters = OpenAIUtils.get_responses_model_params(**kwargs)

        is_local_trace, trace, generation = self._start_trace_and_generation(
            extra_headers=extra_headers,
            model=model,
            messages=messages,
            model_parameters=model_parameters,
        )

        try:
            manager: ResponseStreamManager = self._client.responses.stream(
                *args, **kwargs
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIResponses] Error starting streaming response: {str(e)}"
            )
            raise e

        # Return a wrapper that delegates iteration to the underlying ResponseStream
        # but logs the final response automatically on context exit.
        return ResponsesIteratorWrapper(manager, generation, trace, is_local_trace)
