from typing import Any, Dict, Optional
import inspect
from uuid import uuid4

from openai import AsyncOpenAI
from openai.resources.responses import AsyncResponses
from openai.lib.streaming.responses._responses import (
    AsyncResponseStreamManager,
    AsyncResponseStream,
)

from maxim.logger.components.generation import GenerationConfigDict

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from .utils import OpenAIUtils


class AsyncResponsesIteratorWrapper(AsyncResponseStreamManager):
    def __init__(
        self,
        manager: AsyncResponseStreamManager,
        generation: Optional[Generation],
        trace: Optional[Trace],
        is_local_trace: bool,
    ) -> None:
        self._manager = manager
        self._generation = generation
        self._trace = trace
        self._is_local_trace = is_local_trace
        self._final_response = None
        self._stream: Optional[AsyncResponseStream] = None

    async def __aenter__(self):
        # Enter the underlying async manager to obtain the actual ResponseStream
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        # Try to capture the final response if available
        try:
            get_final = getattr(self._stream, "get_final_response", None)
            if get_final is not None:
                try:
                    result = get_final()
                    if inspect.isawaitable(result):
                        result = await result
                    self._final_response = result
                except Exception as e:
                    # Record failure to retrieve final response on generation
                    if self._generation is not None:
                        self._generation.error(
                            {
                                "message": str(e),
                                "type": getattr(type(e), "__name__", None),
                            }
                        )
                    scribe().warning(
                        f"[MaximSDK][MaximAsyncOpenAIResponses] Error getting final streamed response: {str(e)}"
                    )
        except Exception as e:
            # Extremely defensive: do not let unexpected errors here swallow original exceptions
            if self._generation is not None:
                self._generation.error(
                    {
                        "message": str(e),
                        "type": getattr(type(e), "__name__", None),
                    }
                )
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIResponses] Unexpected error during final response capture: {str(e)}"
            )

        # Log generation result or error
        try:
            if self._generation is not None:
                if exc_type is not None:
                    error_message = str(exc) if exc is not None else str(exc_type)
                    error_type = (
                        getattr(exc_type, "__name__", None)
                        if exc_type is not None
                        else None
                    )
                    self._generation.error(
                        {"message": error_message, "type": error_type}
                    )
                elif self._final_response is not None:
                    self._generation.result(self._final_response)

            # Update trace output when we have a final response
            if (
                self._final_response is not None
                and self._is_local_trace
                and self._trace is not None
            ):
                try:
                    output_text = OpenAIUtils.extract_responses_output_text(
                        self._final_response
                    )
                    if isinstance(output_text, str):
                        self._trace.set_output(output_text)
                except Exception:
                    # Best-effort only
                    pass
        except Exception as e:
            if self._generation is not None:
                self._generation.error(
                    {
                        "message": str(e),
                        "type": getattr(type(e), "__name__", None),
                    }
                )
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIResponses] Error in logging streamed generation: {str(e)}"
            )
        finally:
            # Always end local trace
            if self._is_local_trace and self._trace is not None:
                try:
                    self._trace.end()
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximAsyncOpenAIResponses] Error ending trace: {str(e)}"
                    )

        # Delegate to underlying manager and return its result
        return await self._manager.__aexit__(exc_type, exc, tb)


class MaximAsyncOpenAIResponses(AsyncResponses):
    def __init__(self, client: AsyncOpenAI, logger: Logger):
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
                f"[MaximSDK][MaximAsyncOpenAIResponses] Error starting trace/generation: {str(e)}"
            )

        return is_local_trace, trace, generation

    async def create(self, *args, **kwargs):  # type: ignore[override]
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
            response = await self._client.responses.create(*args, **kwargs)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIResponses] Error generating response: {str(e)}"
            )
            # Mark generation as errored if available
            try:
                if generation is not None:
                    generation.error(
                        {
                            "message": str(e),
                            "type": getattr(type(e), "__name__", None),
                        }
                    )
            except Exception:
                # Best-effort; do not mask original exception
                pass

            # Ensure local trace is closed on error
            if is_local_trace and trace is not None:
                try:
                    try:
                        trace.add_error(
                            {
                                "message": str(e),
                                "type": getattr(type(e), "__name__", None),
                            }
                        )
                    except Exception:
                        # Ignore add_error failure; still try to end the trace
                        pass
                    trace.end()
                except Exception:
                    # Best-effort; do not mask original exception
                    pass

            # Re-raise original exception
            raise

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
                f"[MaximSDK][MaximAsyncOpenAIResponses] Error in logging generation: {str(e)}"
            )

        return response

    def stream(self, *args, **kwargs):  # Returns async context manager
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
            manager = self._client.responses.stream(*args, **kwargs)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIResponses] Error starting streaming response: {str(e)}"
            )
            # Mark generation as errored if available
            try:
                if generation is not None:
                    generation.error(
                        {
                            "message": str(e),
                            "type": getattr(type(e), "__name__", None),
                        }
                    )
            except Exception:
                # Best-effort; do not mask original exception
                pass

            # Ensure local trace is closed on error
            if is_local_trace and trace is not None:
                try:
                    try:
                        trace.add_error(
                            {
                                "message": str(e),
                                "type": getattr(type(e), "__name__", None),
                            }
                        )
                    except Exception:
                        # Ignore add_error failure; still try to end the trace
                        pass
                    trace.end()
                except Exception:
                    # Best-effort; do not mask original exception
                    pass

            raise

        return AsyncResponsesIteratorWrapper(manager, generation, trace, is_local_trace)
