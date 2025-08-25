import ast
import re
import warnings
from typing import Any, Sequence, Union

EXCLUDE_TASK_ATTRS = {"agent": True}

EXCLUDE_AGENT_ATTRS = {
    "crew": True,
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
        # TODO: If obj at this point is a simple type,
        #       maybe we should just return it rather than stringify
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
            # If to_dict fails, we can fall through to the generic object serialization
            # Or, if preferred, raise a more specific error or log it.
            # For now, let's make it fall through by not re-raising or returning here.
            pass  # Fall through to generic handling

    result: dict[Any, Any] = {}
    result["__class__"] = {
        "module": obj.__class__.__module__,
        "qualname": obj.__class__.__qualname__,
        "name": obj.__class__.__name__,
    }
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        # Custom list-like object
        try:
            for i, item in enumerate(obj):
                result[i] = dictify(item, maxdepth, depth + 1, seen)
        except Exception:
            return stringify(obj)
    else:
        # Define Pydantic internal attributes to skip during generic iteration
        # as these are handled by model_dump (via to_dict) or are metadata.
        pydantic_internal_attrs = {
            "model_fields",
            "model_computed_fields",
            "model_config",
            "__slots__",
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
            # Adding common SQLAlchemy internal/deprecated attrs that might appear via dir()
            # though this doesn't fix the SADeprecationWarning origin, it avoids getattr on them here.
            "metadata",
            "_sa_instance_state",
            "registry",
            "full_returning",
        }
        for attr in dir(obj):
            if attr.startswith("_") or attr in pydantic_internal_attrs:
                continue
            if should_redact(attr):
                result[attr] = REDACTED_VALUE
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    val = getattr(obj, attr)
                if callable(val):
                    continue
                if maxdepth == 0 or depth < maxdepth:
                    result[attr] = dictify(val, maxdepth, depth + 1, seen)
                else:
                    result[attr] = stringify(val)
            except Exception as e_attr:
                # Handle error for specific attribute instead of aborting the whole object
                result[attr] = stringify(
                    f"<Error serializing attribute {attr}: {e_attr}>"
                )
    return result


MAX_STR_LEN = 1000


def stringify(obj: Any, limit: int = MAX_STR_LEN) -> str:
    """This is a fallback for objects that we don't have a better way to serialize."""
    rep = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            rep = repr(obj)
    except RecursionError:
        rep = f"<{type(obj).__name__}: RecursionError>"
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                rep = str(obj)
        except Exception:
            rep = f"<{type(obj).__name__}: {id(obj)}>"
    if isinstance(rep, str):
        if len(rep) > limit:
            rep = rep[: limit - 3] + "..."
    return rep


def serialize_crewai_objects(obj: Any) -> Any:
    """Safely serialize CrewAI objects to prevent recursion."""
    # Return primitive types directly
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Everything else is serialized as a dict
    if hasattr(obj, "__class__"):
        if obj.__class__.__name__ == "Agent":
            return serialize_crewai_agent(obj)
        elif obj.__class__.__name__ == "Task":
            return serialize_crewai_task(obj)
        else:
            return dictify(obj)


def serialize_crewai_agent(obj: Any) -> dict[str, Any]:
    # Ensure obj is a pydantic BaseModel
    if not hasattr(obj, "model_dump"):
        return {"type": "Agent", "error": "Not a valid Pydantic model"}

    result = {
        "type": "Agent",
    }

    # Core identity attributes. We want to surface these first.
    core_attributes = ["role", "goal", "backstory", "id"]
    for attr in core_attributes:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if attr == "id" and value is not None:
                result[attr] = stringify(value)
            elif value is not None:
                result[attr] = value

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        attr_dict = obj.model_dump(
            exclude=EXCLUDE_AGENT_ATTRS,
            exclude_none=True,
        )

    for attr, value in attr_dict.items():
        result[attr] = stringify(value)

    return result


def serialize_crewai_task(obj: Any) -> dict[str, Any]:
    if not hasattr(obj, "model_dump"):
        return {"type": "Task", "error": "Not a valid Pydantic model"}

    result = {
        "type": "Task",
    }

    # Core identity attributes. We want to surface these first.
    core_attributes = ["name", "description", "expected_output", "id"]
    for attr in core_attributes:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if attr == "id" and value is not None:
                result[attr] = stringify(value)
            elif value is not None:
                result[attr] = value

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        attr_dict = obj.model_dump(
            exclude=EXCLUDE_TASK_ATTRS,
            exclude_none=True,
        )

    for attr, value in attr_dict.items():
        if attr.startswith("_") or value == "":
            continue
        else:
            result[attr] = stringify(value)

    return result


def crewai_postprocess_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Process CrewAI inputs to prevent recursion."""
    return {k: serialize_crewai_objects(v) for k, v in inputs.items()}


def crew_kickoff_postprocess_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Postprocess the inputs to the Crew.kickoff method.

    The method has a self which should be an instance of `Crew` which is a pydantic model.
    The method also has an inputs which is a dict or list[dict] of arguments to pass to the `kickoff` method.
    """
    results = {}
    for k, v in inputs.items():
        if k == "self":
            if hasattr(v, "model_dump"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    crew_dict = v.model_dump()
                    if isinstance(crew_dict, dict):
                        results["self"] = {
                            k2: v2
                            for k2, v2 in crew_dict.items()
                            if v2 is not None
                            and not (isinstance(v2, list) and len(v2) == 0)
                        }
                    else:
                        results["self"] = crew_dict
        if k == "inputs":
            results["inputs"] = dictify(v)

    return results


def get_task_display_name(inputs: dict) -> str:
    task_representation = inputs.get("self")
    name_value = None  # Variable to hold the raw value of name

    if isinstance(task_representation, dict):
        # If 'self' is a dictionary, get the 'name' value
        name_value = task_representation.get("name")
    elif task_representation is not None:
        # If 'self' is an object, get the 'name' attribute, defaulting to None if not found or if attribute itself is None
        name_value = getattr(task_representation, "name", None)

    # Process the name_value: if it's a string, clean it; otherwise, use an empty string
    if isinstance(name_value, str):
        name = name_value.replace("\\n", "").strip()
    else:
        name = ""  # Default to empty string if name_value is None or not a string

    if not name:  # Check the cleaned and stripped name
        return "crewai.Task"
    return name


def get_agent_display_name(inputs: dict) -> str:
    agent_representation = inputs.get("self")
    title = ""

    if isinstance(agent_representation, dict):
        # If 'self' is a dictionary (likely already serialized)
        title = agent_representation.get("role", "")
        if not title:
            title = agent_representation.get("name", "")
    elif agent_representation is not None:
        # If 'self' is not a dictionary but is not None,
        # assume it's an object (e.g., Agent instance) and try to access attributes.
        # Use getattr for safe access with a default empty string.
        title = getattr(agent_representation, "role", "")
        if not title:  # If role is not found or is empty, try name
            title = getattr(agent_representation, "name", "")

    if not title:  # If title is still empty after trying role and name
        return "crewai.Agent"

    # Ensure title is a string before calling string methods, then clean and strip
    return str(title).replace("\n", "").strip()


def extract_tool_name(text: str) -> str:
    match = re.search(r"\.([\w]+)\._run$", text)
    return match.group(1) if match else ""


def extract_tool_details(description: str) -> dict[str, Any]:
    try:
        name_match = re.search(r"Tool Name:\s*(.+)", description)
        tool_name = name_match.group(1).strip() if name_match else None
    except Exception:
        tool_name = None

    # Extract Tool Arguments as Python dict
    tool_args = {}
    try:
        args_match = re.search(
            r"Tool Arguments:\s*(\{.*?\})\s*Tool Description:", description, re.DOTALL
        )
        if args_match:
            tool_args = ast.literal_eval(args_match.group(1))
    except Exception:
        tool_args = None

    # Extract Tool Description
    try:
        desc_match = re.search(r"Tool Description:\s*(.+)", description)
        tool_description = desc_match.group(1).strip() if desc_match else None
    except Exception:
        tool_description = None

    return {
        "name": tool_name,
        "args": tool_args,
        "description": tool_description,
    }


REDACT_KEYS = (
    "api_key",
    "auth_headers",
    "authorization",
)
REDACTED_VALUE = "REDACTED"


def should_redact(key: str) -> bool:
    return key.lower() in REDACT_KEYS
