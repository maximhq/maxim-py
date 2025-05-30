import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from ...scribe import scribe


class Entity(Enum):
    SESSION = "session"
    TRACE = "trace"
    SPAN = "span"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    GENERATION = "generation"
    FEEDBACK = "feedback"
    RETRIEVAL = "retrieval"


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class CommitLog:

    def __init__(
        self,
        entity: Entity,
        entity_id: str,
        action: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.entity = entity
        self.entity_id = entity_id
        self.action = action
        self.data = data

    def serialize(self, custom_data: Optional[Dict[str, Any]] = None) -> str:
        if custom_data is not None:
            if self.data is None:
                self.data = {}
            self.data.update(custom_data)
        # if action is add-attachment, we need to strip data field in data
        # as this data is already uploaded via separate worker
        # we just need to send the metadata of it
        if (
            self.action == "add-attachment"
            and self.data is not None
            and "data" in self.data
        ):
            del self.data["data"]
        # Here we will check if data is json parsable
        # If not we wont error out but just log it
        if self.data is not None:
            try:
                json.dumps(self.data, cls=DateTimeEncoder)
            except Exception:
                # Find the problematic key by trying to serialize each key-value pair
                problematic_keys = []
                for key, value in self.data.items():
                    try:
                        json.dumps({key: value}, cls=DateTimeEncoder)
                    except Exception:
                        problematic_keys.append(key)

                # Remove only the problematic keys
                for key in problematic_keys:
                    del self.data[key]
                    scribe().error(
                        f"[MaximSDK] Key '{key}' is not serializable and will be removed from data"
                    )

        return f"{self.entity.value}{{id={self.entity_id},action={self.action},data={json.dumps(self.data,cls=DateTimeEncoder)}}}"


def object_to_dict(obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
    """
    Convert a complex object structure to a dictionary, handling nested custom objects.

    Args:
        obj: Any Python object to convert

    Returns:
        A dictionary representation of the object
    """
    if obj is None:
        return None

    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle datetime objects
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [object_to_dict(item) for item in obj]

    # Handle SimpleNamespace
    if isinstance(obj, SimpleNamespace):
        return object_to_dict(vars(obj))

    # Handle custom objects with __dict__
    if hasattr(obj, "__dict__"):
        # Get all attributes, including properties
        attrs = {}
        for key in dir(obj):
            # Skip private attributes and methods
            if not key.startswith("_"):
                value = getattr(obj, key)
                # Skip callable attributes (methods)
                if not callable(value):
                    attrs[key] = object_to_dict(value)
        return attrs

    # Handle other types by converting to string
    return str(obj)


class GenerationErrorTypedDict(TypedDict, total=False):
    message: str
    code: Optional[str]
    type: Optional[str]


@deprecated(
    "This class is deprecated and will be removed in a future version. Use GenerationErrorTypedDict instead."
)
@dataclass
class GenerationError:
    """
    @deprecated: This class is deprecated and will be removed in a future version. Use GenerationErrorTypedDict instead.
    """
    message: str
    code: Optional[str] = None
    type: Optional[str] = None
