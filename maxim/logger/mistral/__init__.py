import importlib.util

from .client import MaximMistralClient
from .utils import MistralUtils

if importlib.util.find_spec("mistralai") is None:
    raise ImportError(
        "The mistralai package is required. Please install it using pip: `pip install mistralai`"
    )

__all__ = ["MaximMistralClient", "MistralUtils"]
