"""
This module is a wrapper around the Portkey client that allows for easy integration with Maxim.

It instruments the Portkey client to log to Maxim.

It also provides a wrapper around the Portkey chat-completions client that allows for easy integration with Maxim.
"""
from typing import Optional, Union
from uuid import uuid4

from portkey_ai import ChatCompletion

from ...scribe import scribe
from ..logger import Generation, Logger, Trace
from ..openai.utils import OpenAIUtils

try:
    from portkey_ai import AsyncPortkey, Portkey  # type: ignore
except ImportError as e:
    raise ImportError(
        (
            "The 'portkey-ai' package is required for Portkey integration. "
            "Install it with `pip install portkey-ai` or `uv add portkey-ai`."
        )
    ) from e


class MaximPortkeyClient:
    """Maxim instrumenter for Portkey client that directly handles chat.completion method."""

    def __init__(self, client: Union[Portkey, AsyncPortkey], logger: Logger):
        self._client = client
        self._logger = logger

        # Handle both sync and async clients
        if isinstance(client, AsyncPortkey):
            self._chat = MaximAsyncPortkeyChat(client, logger)
        else:
            self._chat = MaximPortkeyChat(client, logger)

    @property
    def chat(self) -> Union["MaximPortkeyChat", "MaximAsyncPortkeyChat"]:
        return self._chat

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying Portkey client."""
        return getattr(self._client, name)


class MaximPortkeyChat:
    """Maxim instrumenter for Portkey chat functionality."""

    def __init__(self, client: Portkey, logger: Logger):
        self._client = client
        self._logger = logger
        self._completions = MaximPortkeyChatCompletions(client, logger)

    @property
    def completions(self) -> "MaximPortkeyChatCompletions":
        return self._completions

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying chat client."""
        return getattr(self._client.chat, name)


class MaximAsyncPortkeyChat:
    """Maxim instrumenter for async Portkey chat functionality."""

    def __init__(self, client: AsyncPortkey, logger: Logger):
        self._client = client
        self._logger = logger
        self._completions = MaximAsyncPortkeyChatCompletions(client, logger)

    @property
    def completions(self) -> "MaximAsyncPortkeyChatCompletions":
        return self._completions

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying chat client."""
        return getattr(self._client.chat, name)


class MaximPortkeyChatCompletions:
    """Maxim instrumenter for Portkey chat completions."""

    def __init__(self, client: Portkey, logger: Logger):
        self._client = client
        self._logger = logger

    def create(self, *args, **kwargs):
        """Instrumented create method that logs to Maxim."""
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

        try:
            trace = self._logger.trace({"id": final_trace_id})
            gen_config = {
                "id": str(uuid4()),
                "model": model,
                "provider": "portkey",
                "name": generation_name,
                "model_parameters": OpenAIUtils.get_model_params(**kwargs),
                "messages": OpenAIUtils.parse_message_param(messages),
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximPortkeyChatCompletions] Error in generating content: {str(e)}"
            )

        # Call the actual Portkey completion
        response = self._client.chat.completions.create(*args, **kwargs)

        try:
            parsed_response = None
            if generation is not None:
                parsed_response = OpenAIUtils.parse_completion(response)
                generation.result(parsed_response)
            if is_local_trace and trace is not None and parsed_response is not None:
                trace.set_output(
                    parsed_response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximPortkeyChatCompletions] Error in logging generation: {str(e)}"
            )

        return response

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying completions client."""
        return getattr(self._client.chat.completions, name)


class MaximAsyncPortkeyChatCompletions:
    """Maxim instrumenter for async Portkey chat completions."""

    def __init__(self, client: AsyncPortkey, logger: Logger):
        self._client = client
        self._logger = logger

    async def create(self, *args, **kwargs):
        """Instrumented async create method that logs to Maxim."""
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

        try:
            trace = self._logger.trace({"id": final_trace_id})
            gen_config = {
                "id": str(uuid4()),
                "model": model,
                "provider": "portkey",
                "name": generation_name,
                "model_parameters": OpenAIUtils.get_model_params(**kwargs),
                "messages": OpenAIUtils.parse_message_param(messages),
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncPortkeyChatCompletions] Error in generating content: {str(e)}"
            )

        # Call the actual async Portkey completion
        response = await self._client.chat.completions.create(*args, **kwargs)

        try:
            parsed_response = None
            if generation is not None:
                parsed_response = OpenAIUtils.parse_completion(response)
                generation.result(parsed_response)
            if is_local_trace and trace is not None and parsed_response is not None:
                trace.set_output(
                    parsed_response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncPortkeyChatCompletions] Error in logging generation: {str(e)}"
            )

        return response

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying completions client."""
        return getattr(self._client.chat.completions, name)
