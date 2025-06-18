"""Portkey AI integration for Maxim logging.

This module provides one-line integration with Portkey AI clients,
enabling automatic logging of OpenAI-compatible calls via Maxim.
"""

from .client import instrument_portkey
from .portkey import (
    MaximPortkeyClient,
)

__all__ = [
    "MaximPortkeyClient",
    "instrument_portkey",
]
