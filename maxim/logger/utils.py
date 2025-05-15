import collections.abc
import inspect
import re
import types
from datetime import datetime
from decimal import Decimal
from typing import Any


def make_object_serializable(obj: Any) -> Any:
    """
    Convert any Python object into a JSON-serializable format while preserving
    as much information as possible about the original object.

    Args:
        obj: Any Python object

    Returns:
        A JSON-serializable representation of the object
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic types that are already serializable
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle Decimal
    if isinstance(obj, Decimal):
        return str(obj)

    # Handle complex numbers
    if isinstance(obj, complex):
        return {"type": "complex", "real": obj.real, "imag": obj.imag}

    # Handle bytes and bytearray
    if isinstance(obj, (bytes, bytearray)):
        return {"type": "bytes", "data": obj.hex(), "encoding": "hex"}

    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Handle regular expressions
    if isinstance(obj, re.Pattern):
        return {"type": "regex", "pattern": obj.pattern, "flags": obj.flags}

    # Handle functions
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        return {
            "type": "function",
            "name": obj.__name__,
            "module": obj.__module__,
            "source": inspect.getsource(obj) if inspect.isroutine(obj) else None,
            "signature": str(inspect.signature(obj))
            if inspect.isroutine(obj)
            else None,
        }

    # Handle exceptions
    if isinstance(obj, Exception):
        return {
            "type": "error",
            "error_type": obj.__class__.__name__,
            "message": str(obj),
            "args": make_object_serializable(obj.args),
            "traceback": str(obj.__traceback__) if obj.__traceback__ else None,
        }

    # Handle sets
    if isinstance(obj, (set, frozenset)):
        return {
            "type": "set",
            "is_frozen": isinstance(obj, frozenset),
            "values": [make_object_serializable(item) for item in obj],
        }

    # Handle dictionaries and mapping types
    if isinstance(obj, collections.abc.Mapping):
        return {str(key): make_object_serializable(value) for key, value in obj.items()}

    # Handle lists, tuples, and other iterables
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, collections.abc.Iterable)
        and not isinstance(obj, (str, bytes, bytearray))
    ):
        return [make_object_serializable(item) for item in obj]

    # Handle custom objects
    try:
        # Try to convert object's dict representation
        obj_dict = obj.__dict__
        return {
            "type": "custom_object",
            "class": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "attributes": make_object_serializable(obj_dict),
        }
    except AttributeError:
        # If object doesn't have __dict__, try to get string representation
        return {
            "type": "unknown",
            "class": obj.__class__.__name__,
            "string_repr": str(obj),
        }
