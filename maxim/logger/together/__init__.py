"""Together AI integration for Maxim logging.

This module provides one-line integration with Together AI clients,
enabling automatic logging of Together AI calls via Maxim.
"""
from .client import instrument_together

__all__ = ["instrument_together"]