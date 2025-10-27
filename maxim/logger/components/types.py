import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from ...scribe import scribe


class Entity(Enum):
    """Entity.

    This class represents an entity.
    """

    SESSION = "session"
    TRACE = "trace"
    SPAN = "span"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    GENERATION = "generation"
    FEEDBACK = "feedback"
    RETRIEVAL = "retrieval"


class CustomEncoder(json.JSONEncoder):
    """Custom encoder.

    This class represents a custom encoder that can handle any object type.
    """

    def default(self, o):
        # Handle datetime objects
        if isinstance(o, datetime):
            return o.isoformat()

        # Handle any object with model_dump (newer Pydantic)
        if hasattr(o, "model_dump") and callable(o.model_dump):
            return o.model_dump()

        # Handle any object with dict (older Pydantic)
        if hasattr(o, "dict") and callable(o.dict):
            return o.dict()

        # Handle any object with to_dict
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()

        # Handle any object with __dict__
        if hasattr(o, "__dict__"):
            return vars(o)

        # Handle any object with _asdict (namedtuples)
        if hasattr(o, "_asdict") and callable(o._asdict):
            return o._asdict()

        return super().default(o)


class CommitLog:
    """Commit log.

    This class represents a commit log.
    """

    def __init__(
        self,
        entity: Entity,
        entity_id: str,
        action: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a commit log.

        Args:
            entity: The entity of the commit log.
            entity_id: The id of the entity.
            action: The action of the commit log.
            data: The data of the commit log.
        """
        self.entity = entity
        self.entity_id = entity_id
        self.action = action
        self.data = data

    def serialize(self, custom_data: Optional[Dict[str, Any]] = None) -> str:
        """Serialize the commit log.

        Args:
            custom_data: The custom data to serialize.

        Returns:
            str: The serialized commit log.
        """
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
                json.dumps(self.data, cls=CustomEncoder)
            except Exception:
                # Find the problematic key by trying to serialize each key-value pair
                problematic_keys = []
                for key, value in self.data.items():
                    try:
                        json.dumps({key: value}, cls=CustomEncoder)
                    except Exception:
                        problematic_keys.append(key)

                # Remove only the problematic keys
                for key in problematic_keys:
                    del self.data[key]
                    scribe().error(
                        f"[MaximSDK] Key '{key}' is not serializable and will be removed from data"
                    )

        return f"{self.entity.value}{{id={self.entity_id},action={self.action},data={json.dumps(self.data,cls=CustomEncoder)}}}"


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
    """Generation error typed dict.

    This class represents a generation error typed dict.
    """

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
