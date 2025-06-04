"""Mistral integration for the Maxim logger.

This module provides logging and tracing capabilities for Mistral SDK operations,
wrapping the Mistral client and chat interfaces with automatic instrumentation.
"""
import importlib.util

from .client import MaximMistralClient
from .utils import MistralUtils

_MISTRALAI_IMPORT_ERROR = (
    "The mistralai package is required. Please install it using pip: `pip install mistralai`"
)

if importlib.util.find_spec("mistralai") is None:
    raise ImportError(_MISTRALAI_IMPORT_ERROR)

__all__ = ["MaximMistralClient", "MistralUtils"]
