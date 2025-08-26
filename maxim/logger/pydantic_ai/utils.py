"""Utility functions for Pydantic AI integration."""

import ast
import re
import warnings
from typing import Any, Sequence, Union

EXCLUDE_AGENT_ATTRS = {
    "model": True,
    "name": True,
    "deps_type": True,
}

EXCLUDE_TOOL_ATTRS = {
    "name": True,
    "description": True,
}


def is_primitive(obj: Any) -> bool:
    """Check if an object is a known primitive type."""
    return isinstance(obj, (int, float, str, bool, type(None)))


def dictify(
    obj: Any, maxdepth: int = 0, depth: int = 1, seen: Union[set[int], None] = None
) -> Any:
    """Recursively compute a dictionary representation of an object."""
    if seen is None:
        seen = set()

    if not is_primitive(obj):
        obj_id = id(obj)
        if obj_id in seen:
            # Avoid infinite recursion with circular references
            return stringify(obj)
        else:
            seen.add(obj_id)

    if maxdepth > 0 and depth > maxdepth:
        return stringify(obj)

    if is_primitive(obj):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [dictify(v, maxdepth, depth + 1, seen) for v in obj]
    elif isinstance(obj, dict):
        dict_result = {}
        for k, v in obj.items():
            if isinstance(k, str) and should_redact(k):
                dict_result[k] = REDACTED_VALUE
            else:
                dict_result[k] = dictify(v, maxdepth, depth + 1, seen)
        return dict_result

    if hasattr(obj, "to_dict"):
        try:
            as_dict = obj.to_dict()
            if isinstance(as_dict, dict):
                to_dict_result = {}
                for k, v in as_dict.items():
                    if isinstance(k, str) and should_redact(k):
                        to_dict_result[k] = REDACTED_VALUE
                    elif maxdepth == 0 or depth < maxdepth:
                        to_dict_result[k] = dictify(v, maxdepth, depth + 1, seen)
                    else:
                        to_dict_result[k] = stringify(v)
                return to_dict_result
        except Exception:
            pass

    result: dict[Any, Any] = {}
    result["__class__"] = {
        "module": obj.__class__.__module__,
        "qualname": obj.__class__.__qualname__,
        "name": obj.__class__.__name__,
    }
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        try:
            for i, item in enumerate(obj):
                result[i] = dictify(item, maxdepth, depth + 1, seen)
        except Exception:
            return stringify(obj)
    else:
        pydantic_internal_attrs = {
            "model_fields",
            "model_computed_fields",
            "model_config",
            "__slots__",
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
            "metadata",
            "_sa_instance_state",
            "registry",
            "full_returning",
        }

        # Limit depth to avoid infinite recursion
        if depth > 3:
            return stringify(obj)
            
        for attr_name in dir(obj):
            if (
                attr_name.startswith("_")
                and not attr_name.startswith("__")
                or attr_name in pydantic_internal_attrs
            ):
                continue

            try:
                attr_value = getattr(obj, attr_name)
                if callable(attr_value):
                    continue

                if isinstance(attr_name, str) and should_redact(attr_name):
                    result[attr_name] = REDACTED_VALUE
                else:
                    # Only go deeper for simple types to avoid recursion
                    if is_primitive(attr_value) or depth < 2:
                        result[attr_name] = dictify(attr_value, maxdepth, depth + 1, seen)
                    else:
                        result[attr_name] = stringify(attr_value)
            except Exception:
                continue

    return result


def stringify(obj: Any) -> str:
    """Convert an object to a string representation."""
    try:
        return str(obj)
    except Exception:
        return f"<{type(obj).__name__} object>"


def pydantic_ai_postprocess_inputs(inputs: dict) -> dict:
    """Post-process inputs for Pydantic AI operations."""
    results = {}
    for k, v in inputs.items():
        if k == "self":
            # Extract relevant agent information
            if hasattr(v, "name"):
                results["agent_name"] = v.name
            if hasattr(v, "model"):
                results["model"] = str(v.model)
            if hasattr(v, "deps_type"):
                results["deps_type"] = str(v.deps_type)
        elif k == "user_prompt":
            results["user_prompt"] = v
        elif k == "deps":
            results["deps"] = dictify(v)
        elif k == "model_settings":
            results["model_settings"] = dictify(v)
        elif k == "message_history":
            results["message_history"] = dictify(v)
        else:
            results[k] = dictify(v)

    return results


def get_agent_display_name(inputs: dict) -> str:
    """Get display name for agent operations."""
    agent_representation = inputs.get("self")
    name = ""

    if isinstance(agent_representation, dict):
        name = agent_representation.get("agent_name", "")
    elif agent_representation is not None:
        name = getattr(agent_representation, "name", "")

    if not name:
        return "pydantic_ai.Agent"

    return str(name).replace("\n", "").strip()


def get_tool_display_name(inputs: dict) -> str:
    """Get display name for tool operations."""
    tool_representation = inputs.get("self")
    name = ""

    if isinstance(tool_representation, dict):
        name = tool_representation.get("name", "")
    elif tool_representation is not None:
        name = getattr(tool_representation, "name", "")

    if not name:
        return "pydantic_ai.Tool"

    return str(name).replace("\n", "").strip()


def extract_tool_details(tool_def: Any) -> dict[str, Any]:
    """Extract tool details from tool definition."""
    try:
        name = getattr(tool_def, "name", None)
        description = getattr(tool_def, "description", None)
        
        # Try to extract parameters from schema
        parameters = {}
        if hasattr(tool_def, "parameters"):
            parameters = dictify(tool_def.parameters)
        
        return {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    except Exception:
        return {
            "name": None,
            "description": None,
            "parameters": {},
        }


REDACT_KEYS = (
    "api_key",
    "auth_headers",
    "authorization",
    "token",
    "secret",
)
REDACTED_VALUE = "REDACTED"


def should_redact(key: str) -> bool:
    """Check if a key should be redacted."""
    return key.lower() in REDACT_KEYS 