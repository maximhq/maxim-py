import importlib.util

from .client import MaximAgnoClient, instrument_agno

if importlib.util.find_spec("agno") is None:
    raise ImportError(
        "The agno package is required. Please install it using pip: `pip install agno`"
    )

__all__ = ["MaximAgnoClient", "instrument_agno"]
