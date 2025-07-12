"""Groq integration for Maxim logging.

This module provides one-line integration with Groq clients,
enabling automatic logging of Groq calls via Maxim.
"""

from .client import instrument_groq

__all__ = ["instrument_groq"]
