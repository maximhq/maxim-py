"""Agno integration module for Maxim SDK.

This module provides instrumentation for the Agno AI agent framework.
"""

import importlib.util

from .client import instrument_agno

if importlib.util.find_spec("agno") is None:
    raise ImportError(
        "The agno package is required. Please install it using pip: `pip install agno`",
    )

__all__ = ["instrument_agno"]
