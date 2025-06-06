from typing import Any

try:
    from portkey_ai.api_resources.client import Portkey, AsyncPortkey
except ImportError as e:
    raise ImportError(
        "The portkey-ai package is required. Please install it using pip: "
        "`pip install portkey-ai` or `uv add portkey-ai`",
    ) from e

from ..openai import MaximOpenAIAsyncClient, MaximOpenAIClient
from ..logger import Logger


def instrument_portkey(client: Any, logger: Logger) -> Any:
    """Attach Maxim OpenAI wrappers to a Portkey client.

    This helper patches the ``openai_client`` attribute of a ``Portkey`` or
    ``AsyncPortkey`` instance so that all OpenAI-compatible calls are logged
    via Maxim.

    Args:
        client: Instance of ``portkey_ai.api_resources.client.Portkey`` or
            ``AsyncPortkey``.
        logger: Maxim ``Logger`` instance.

    Returns:
        The same client instance with its ``openai_client`` patched.
    """

    if isinstance(client, Portkey):
        client.openai_client = MaximOpenAIClient(client.openai_client, logger)
    elif isinstance(client, AsyncPortkey):
        client.openai_client = MaximOpenAIAsyncClient(client.openai_client, logger)
    else:
        raise TypeError("client must be an instance of Portkey or AsyncPortkey")

    return client
