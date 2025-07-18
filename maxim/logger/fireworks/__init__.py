"""Fireworks AI integration for Maxim logging.

This module provides one-line integration with Fireworks AI clients,
enabling automatic logging of Fireworks AI calls via Maxim.
"""
from .client import instrument_fireworks

__all__ = ["instrument_fireworks"]