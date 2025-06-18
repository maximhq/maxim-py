"""
This module is a wrapper around the Portkey client that allows for easy integration with Maxim.

It instruments the Portkey client to log to Maxim.

It also provides a wrapper around the Portkey chat-completions client that allows for easy integration with Maxim.
"""

from typing import Union

try:
    from portkey_ai.api_resources.client import AsyncPortkey, Portkey
except ImportError as e:
    raise ImportError(
        (
            "The 'portkey-ai' package is required for Portkey integration. "
            "Install it with `pip install portkey-ai` or `uv add portkey-ai`.",
        )
    ) from e


from ..logger import Logger
from .portkey import MaximPortkeyClient

"""Portkey AI client instrumentation for Maxim logging."""


# Type alias for better clarity
PortkeyClient = Union[Portkey, AsyncPortkey]


def instrument_portkey(client: PortkeyClient, logger: Logger) -> MaximPortkeyClient:
    """Attach Maxim OpenAI wrappers to a Portkey client.

    This helper patches the ``openai_client`` attribute of a ``Portkey`` or
    ``AsyncPortkey`` instance so that all OpenAI-compatible calls are logged
    via Maxim.

    Args:
        client: Instance of Portkey or AsyncPortkey client.
        logger: Maxim ``Logger`` instance.

    Returns:
        The same client instance with its ``openai_client`` patched.
    """
    return MaximPortkeyClient(client, logger)
