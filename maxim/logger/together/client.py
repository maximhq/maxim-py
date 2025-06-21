"""Together SDK instrumentation helpers."""

import functools
from typing import Any
from uuid import uuid4

try:
    from together.resources.chat import AsyncChatCompletions, ChatCompletions
except ImportError as e:
    raise ImportError(
        (
            "The 'together' package is required for Together integration. "
            "Install it with `pip install together` or `uv add together`."
        )
    ) from e

from ..logger import Generation, Logger, Trace
from ..openai.utils import OpenAIUtils
from ..scribe import scribe


_INSTRUMENTED = False


def instrument_together(logger: Logger) -> None:
    """Patch Together's chat completion methods for logging."""

    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().debug("[MaximSDK] Together already instrumented")
        return

    def _wrap_sync_create(create_func):
        @functools.wraps(create_func)
        def wrapper(self: ChatCompletions, *args: Any, **kwargs: Any):
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get(
                    "x-maxim-generation-name", None
                )

            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Generation | None = None
            trace: Trace | None = None
            messages = kwargs.get("messages", None)

            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "together",
                    "name": generation_name,
                    "model_parameters": OpenAIUtils.get_model_params(**kwargs),
                    "messages": OpenAIUtils.parse_message_param(messages),
                }
                generation = trace.generation(gen_config)
            except Exception as e:  # pragma: no cover - best effort
                scribe().warning(
                    f"[MaximSDK][TogetherInstrumentation] Error in generating content: {e}"
                )

            response = create_func(self, *args, **kwargs)

            try:
                if generation is not None:
                    generation.result(OpenAIUtils.parse_completion(response))
                if is_local_trace and trace is not None:
                    trace.set_output(response.choices[0].message.content or "")
                    trace.end()
            except Exception as e:  # pragma: no cover - best effort
                scribe().warning(
                    f"[MaximSDK][TogetherInstrumentation] Error in logging generation: {e}"
                )

            return response

        return wrapper

    def _wrap_async_create(create_func):
        @functools.wraps(create_func)
        async def wrapper(self: AsyncChatCompletions, *args: Any, **kwargs: Any):
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get(
                    "x-maxim-generation-name", None
                )

            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Generation | None = None
            trace: Trace | None = None
            messages = kwargs.get("messages", None)

            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "together",
                    "name": generation_name,
                    "model_parameters": OpenAIUtils.get_model_params(**kwargs),
                    "messages": OpenAIUtils.parse_message_param(messages),
                }
                generation = trace.generation(gen_config)
            except Exception as e:  # pragma: no cover - best effort
                scribe().warning(
                    f"[MaximSDK][TogetherInstrumentation] Error in generating content: {e}"
                )

            response = await create_func(self, *args, **kwargs)

            try:
                if generation is not None:
                    generation.result(OpenAIUtils.parse_completion(response))
                if is_local_trace and trace is not None:
                    trace.set_output(response.choices[0].message.content or "")
                    trace.end()
            except Exception as e:  # pragma: no cover - best effort
                scribe().warning(
                    f"[MaximSDK][TogetherInstrumentation] Error in logging generation: {e}"
                )

            return response

        return wrapper

    ChatCompletions.create = _wrap_sync_create(ChatCompletions.create)
    AsyncChatCompletions.create = _wrap_async_create(AsyncChatCompletions.create)
    _INSTRUMENTED = True
