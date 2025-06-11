"""Together AI integration for Maxim logging.

This module exposes helpers to instrument the Together SDK so that
OpenAI-compatible calls are automatically logged via Maxim.
"""

from .client import instrument_together

__all__ = ["instrument_together"]
