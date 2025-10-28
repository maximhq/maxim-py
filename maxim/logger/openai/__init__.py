import importlib.util

from .async_client import MaximOpenAIAsyncClient
from .client import MaximOpenAIClient
from .responses import MaximOpenAIResponses
from .async_responses import MaximAsyncOpenAIResponses
from .utils import OpenAIUtils

if importlib.util.find_spec("openai") is None:
    raise ImportError(
        "The openai package is required. Please install it using pip: `pip install openai` or `uv add openai`"
    )

__all__ = [
    "OpenAIUtils",
    "MaximOpenAIAsyncClient",
    "MaximOpenAIClient",
    "MaximOpenAIResponses",
    "MaximAsyncOpenAIResponses",
]
