"""Portkey AI integration for Maxim logging.

This module provides one-line integration with Portkey AI clients,
enabling automatic logging of OpenAI-compatible calls via Maxim.
"""
from .client import instrument_portkey

__all__ = ["instrument_portkey"]
