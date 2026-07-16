import datetime
import decimal
import enum
import json
import pathlib
import types
import uuid
from typing import Any, Dict, List, Optional

# A bit of workaround to make sure there are no breakages when openai is not installed
try:
    from openai.types.chat import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function
except Exception:  # pragma: no cover
    ChatCompletionMessageToolCall = None  # type: ignore[assignment]
    Function = None  # type: ignore[assignment]

from ...scribe import scribe
from .core import (
    validate_content,
    validate_optional_type,
    validate_type,
    validate_type_to_be_one_of,
)


def parse_function_call(function_call_data):
    """
    Parse function call from a dictionary.

    Args:
        function_call_data: The dictionary to parse.

    Returns:
        The parsed function call.
    """
    if Function is not None and isinstance(function_call_data, Function):
        validate_type(function_call_data.name, str, "name")
        validate_type(function_call_data.arguments, str, "arguments")
    else:
        if function_call_data.get("name") is not None:
            validate_type(function_call_data.get("name"), str, "name")
        else:
            validate_type(function_call_data.name, str, "name")

        if function_call_data.get("arguments") is not None:
            validate_type(function_call_data.get("arguments"), str, "arguments")
        else:
            validate_type(function_call_data.arguments, str, "arguments")

    return function_call_data


def parse_tool_calls(tool_calls_data):
    """
    Parse tool calls from a dictionary.

    Args:
        tool_calls_data: The dictionary to parse.

    Returns:
        The parsed tool calls.
    """
    if ChatCompletionMessageToolCall is not None and isinstance(
        tool_calls_data, ChatCompletionMessageToolCall
    ):
        validate_type(tool_calls_data.id, str, "id")
        validate_type(tool_calls_data.type, str, "type")
        parse_function_call(tool_calls_data.function)
    else:
        if tool_calls_data.get("id") is not None:
            validate_type(tool_calls_data.get("id"), str, "id")
        else:
            validate_type(tool_calls_data.id, str, "id")

        if tool_calls_data.get("type") is not None:
            validate_type(tool_calls_data.get("type"), str, "type")
        else:
            validate_type(tool_calls_data.type, str, "type")

        if tool_calls_data.get("function") is not None:
            parse_function_call(tool_calls_data.get("function"))
        else:
            parse_function_call(tool_calls_data.get("function"))

    return tool_calls_data


def parse_content_list(content_list_data):
    """
    Parse content list from a dictionary.

    Args:
        content_list_data: The dictionary to parse.

    Returns:
        The parsed content list.
    """
    for content in content_list_data:
        if content is None:
            continue
        if "type" in content and content["type"] == "audio":
            validate_type(content.get("transcript"), str, "transcript")
        elif "type" in content and content["type"] == "text":
            validate_type(content.get("text"), str, "text")
        elif "type" in content and content["type"] == "image_url":
            validate_type(content.get("image_url"), str, "image_url")
        else:
            raise ValueError(
                f"Invalid content type. We expect 'text', 'image' or 'audio' type. Got: {content.get('type')}"
            )
    return content_list_data


def parse_chat_completion_choice(messages_data):
    """
    Parse chat completion choice from a dictionary.

    Args:
        messages_data: The dictionary to parse.

    Returns:
        The parsed chat completion choice.
    """
    validate_type(messages_data.get("role"), str, "role")
    # Here it can be either string or list
    if isinstance(messages_data.get("content"), list):
        parse_content_list(messages_data.get("content"))
    else:
        validate_optional_type(messages_data.get("content"), str, "content")
    if messages_data.get("function_call") is not None:
        parse_function_call(messages_data.get("function_call"))
    elif messages_data.get("tool_calls") is not None:
        # Check if its a list of tool calls
        if isinstance(messages_data.get("tool_calls"), list):
            for tool_call in messages_data.get("tool_calls"):
                parse_tool_calls(tool_call)
        else:
            parse_tool_calls(messages_data.get("tool_calls"))
    return messages_data


def parse_choice(choice_data):
    """
    Parse choice from a dictionary.

    Args:
        choice_data: The dictionary to parse.

    Returns:
        The parsed choice.
    """
    validate_type(choice_data.get("index"), int, "index")
    validate_optional_type(choice_data.get("finish_reason"), str, "finish_reason")

    # Checking if text completion or chat completion
    if choice_data.get("text") is not None:
        validate_type(choice_data.get("text"), str, "text")
    elif choice_data.get("message") is not None:
        parse_chat_completion_choice(choice_data.get("message"))
    # TODO remove this as this is deprecated and wrong
    elif choice_data.get("messages") is not None:
        parse_chat_completion_choice(choice_data.get("messages"))

    return choice_data


def parse_usage(usage_data):
    """
    Parse usage from a dictionary.

    Args:
        usage_data: The dictionary to parse.

    Returns:
        The parsed usage.
    """
    if usage_data is None:
        return None
    if (
        usage_data.get("input_audio_duration") is not None
        or usage_data.get("output_audio_duration") is not None
    ):
        input_audio_duration = usage_data.get("input_audio_duration")
        output_audio_duration = usage_data.get("output_audio_duration")
        if input_audio_duration is not None:
            validate_optional_type(input_audio_duration, float, "input_audio_duration")
        if output_audio_duration is not None:
            validate_optional_type(output_audio_duration, float, "output_audio_duration")
    else:
        validate_type(usage_data.get("prompt_tokens"), int, "prompt_tokens")
        validate_type(usage_data.get("completion_tokens"), int, "completion_tokens")
        validate_type(usage_data.get("total_tokens"), int, "total_tokens")
    return usage_data


def parse_generation_error(error_data):
    """
    Parse generation error from a dictionary.

    Args:
        error_data: The dictionary to parse.

    Returns:
        The parsed generation error.
    """
    if error_data is None:
        return None
    validate_type(error_data.get("message"), str, "message")
    validate_optional_type(error_data.get("code"), str, "code")
    validate_optional_type(error_data.get("type"), str, "type")
    return error_data


def default_json_serializer(o: Any) -> Any:
    """
    Default JSON serializer for objects.

    Args:
        o: The object to serialize.

    Returns:
        The serialized object.
    """
    if isinstance(o, enum.Enum):
        return o.value
    # Read-only dict wrappers (e.g. a class' __dict__) are not JSON serializable
    # by default.
    if isinstance(o, types.MappingProxyType):
        return dict(o)
    # Classes must be handled before the instance-method branches below, since
    # those methods exist on the class as unbound functions and would raise if
    # called without an instance. A model *class* is how structured-output
    # formats are usually declared (e.g. response_format=MyModel), and its
    # schema is the meaningful representation.
    if isinstance(o, type):
        # pydantic v2 model class
        if callable(getattr(o, "model_json_schema", None)):
            return o.model_json_schema()
        # pydantic v1 model class
        if callable(getattr(o, "schema", None)):
            return o.schema()
        return {"type": o.__name__}
    if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
        return o.isoformat()
    if isinstance(o, datetime.timedelta):
        return o.total_seconds()
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, uuid.UUID):
        return str(o)
    if isinstance(o, pathlib.PurePath):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    if callable(getattr(o, "to_dict", None)):
        return o.to_dict()
    # Pydantic v2 model instances expose model_dump().
    if callable(getattr(o, "model_dump", None)):
        return o.model_dump()
    # numpy scalars and arrays (duck-typed, so numpy stays an optional import).
    # tolist() covers both and returns native python types.
    if callable(getattr(o, "tolist", None)):
        return o.tolist()
    # Functions and exceptions would otherwise fall through to vars() and
    # serialize as a misleading empty dict, so describe them instead.
    if isinstance(
        o, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)
    ):
        return {"type": getattr(o, "__name__", "function")}
    if isinstance(o, BaseException):
        return {"type": type(o).__name__, "message": str(o)}

    try:
        return vars(o)
    except TypeError:
        pass

    # Objects using __slots__ have no __dict__, so vars() above fails on them.
    slots: List[str] = []
    for klass in type(o).__mro__:
        klass_slots = getattr(klass, "__slots__", ())
        if isinstance(klass_slots, str):
            klass_slots = (klass_slots,)
        slots.extend(klass_slots)
    if slots:
        return {s: getattr(o, s) for s in slots if hasattr(o, s)}

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def is_openai_response_structure(data: Any) -> bool:
    """
    Check if data matches the general top-level shape of an OpenAI Responses API result.

    The OpenAI Responses API structure includes:
    - id: string identifier
    - object: string (must be "response")
    - created_at: integer timestamp
    - status: string (e.g., "completed", "in_progress")
    - output: list of output items
    - usage: dict with token usage

    Args:
        data: The dictionary to check.

    Returns:
        True if the data matches the OpenAI Responses API structure, False otherwise.
    """
    if not isinstance(data, dict):
        return False

    # Check for required OpenAI Responses API fields
    required_fields = {
        "id": str,
        "object": str,
        "created_at": (int, float),
        "status": str,
        "output": list,
        "usage": dict,
    }

    for field, expected_type in required_fields.items():
        if field not in data:
            return False

        value = data[field]
        if not isinstance(value, expected_type):
            return False

    # Verify that object is specifically "response"
    if data.get("object") != "response":
        return False

    return True


def _responses_output_item_type(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("type") if isinstance(item.get("type"), str) else None
    t = getattr(item, "type", None)
    return t if isinstance(t, str) else None


def _responses_reasoning_summary_text(item: Any) -> str:
    summary = item.get("summary") if isinstance(item, dict) else getattr(item, "summary", None)
    if summary is None:
        return ""
    if isinstance(summary, str):
        return summary
    if not isinstance(summary, (list, tuple)):
        return str(summary)
    parts: List[str] = []
    for block in summary:
        if isinstance(block, dict):
            if block.get("type") == "summary_text":
                tx = block.get("text")
                if isinstance(tx, str):
                    parts.append(tx)
        else:
            if getattr(block, "type", None) == "summary_text":
                tx = getattr(block, "text", None)
                if isinstance(tx, str):
                    parts.append(tx)
    return "".join(parts)


def _responses_assistant_output_text(item: Any) -> str:
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content) if content is not None else ""
    parts: List[str] = []
    for c in content:
        if isinstance(c, dict):
            ct = c.get("type")
            if ct == "output_text":
                tx = c.get("text")
                if isinstance(tx, str):
                    parts.append(tx)
            elif ct == "refusal":
                ref = c.get("refusal")
                if isinstance(ref, str):
                    parts.append(f"[refusal] {ref}")
        else:
            ct = getattr(c, "type", None)
            if ct == "output_text":
                tx = getattr(c, "text", None)
                if isinstance(tx, str):
                    parts.append(tx)
            elif ct == "refusal":
                ref = getattr(c, "refusal", None)
                if isinstance(ref, str):
                    parts.append(f"[refusal] {ref}")
    return "".join(parts)


def _responses_is_assistant_message_with_output_text(item: Any) -> bool:
    if _responses_output_item_type(item) != "message":
        return False
    role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
    if role != "assistant":
        return False
    return bool(_responses_assistant_output_text(item).strip()) or (
        isinstance(item, dict)
        and isinstance(item.get("content"), list)
        or (not isinstance(item, dict) and isinstance(getattr(item, "content", None), list))
    )


def _responses_prepend_thinking_to_message_dict(msg: dict, prefix: str) -> None:
    content = msg.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "output_text":
            t = block.get("text")
            if isinstance(t, str):
                if t.startswith(prefix):
                    return
                block["text"] = prefix + t
            else:
                block["text"] = prefix
            return


def _responses_dict_has_output_text_block(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "output_text" for b in content)


def _responses_item_has_output_text_block(item: Any) -> bool:
    if isinstance(item, dict):
        return _responses_dict_has_output_text_block(item)
    content = getattr(item, "content", None)
    if not isinstance(content, (list, tuple)):
        return False
    for b in content:
        if getattr(b, "type", None) == "output_text":
            return True
    return False


def inject_thinking_into_openai_response_result(result: Dict[str, Any]) -> None:
    """Prepend ``<think>...</think>`` to assistant output_text when immediately preceded by reasoning.

    Mutates ``result["output"]`` in place. Used when logging OpenAI Responses API results.
    """
    output = result.get("output")
    if not isinstance(output, list):
        return
    pending: List[str] = []
    for item in output:
        itype = _responses_output_item_type(item)
        if itype == "reasoning":
            chunk = _responses_reasoning_summary_text(item)
            if chunk:
                pending.append(chunk)
            continue
        if itype == "message" and isinstance(item, dict):
            role = item.get("role")
            if role == "assistant":
                if pending and _responses_dict_has_output_text_block(item):
                    combined = "".join(pending)
                    prefix = f"<think>\n\n{combined}\n\n</think>\n\n"
                    _responses_prepend_thinking_to_message_dict(item, prefix)
                pending = []
            else:
                pending = []
            continue
        pending = []


def compose_openai_responses_output_text_with_thinking(output: Any) -> str:
    """Build visible text from Responses ``output`` items, including ``<think>`` prefixes (read-only)."""
    if not isinstance(output, list):
        return ""
    pending: List[str] = []
    chunks: List[str] = []
    for item in output:
        itype = _responses_output_item_type(item)
        if itype == "reasoning":
            chunk = _responses_reasoning_summary_text(item)
            if chunk:
                pending.append(chunk)
            continue
        if itype == "message":
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            if role == "assistant":
                body = _responses_assistant_output_text(item)
                has_ot = _responses_item_has_output_text_block(item)
                if pending:
                    if has_ot:
                        chunks.append(f"<think>\n\n{''.join(pending)}\n\n</think>\n\n")
                    pending = []
                if body:
                    chunks.append(body)
            else:
                pending = []
            continue
        pending = []
    return "".join(chunks)


def parse_result(data: Any) -> Dict[str, Any]:
    """
    Parse result from a dictionary.

    Supports both OpenAI Chat Completion API and OpenAI Responses API result structures.

    Args:
        data: The dictionary to parse.

    Returns:
        The parsed result.
    """
    if not isinstance(data, dict):
        raise ValueError("Text completion is not supported.")

    # Check if this is an OpenAI Responses API result structure
    if is_openai_response_structure(data):
        # For Responses API results, return as-is without deep validation
        # Only the general top-level shape is validated by is_openai_response_structure
        inject_thinking_into_openai_response_result(data)
        return data

    # Otherwise, process as Chat Completion API result (existing behavior)
    validate_type(data.get("id"), str, "id")
    validate_optional_type(data.get("object"), str, "object")
    validate_type(data.get("created"), int, "created")
    validate_optional_type(data.get("model"), str, "model")

    choices_data = data.get("choices")
    validate_type_to_be_one_of(choices_data, [list, List], "choices")
    if choices_data is None:
        choices_data = []
    choices = [parse_choice(choice) for choice in choices_data]
    usage = parse_usage(data.get("usage", None))
    error = parse_generation_error(data.get("error", None))
    result = {
        "id": data["id"],
        "object": data["object"] if "object" in data else None,
        "created": data["created"],
        "choices": choices,
        "usage": usage,
        "error": error if error else None,
    }
    # removing all None from result
    result = {k: v for k, v in result.items() if v is not None}
    return result


def parse_message(message: Any) -> Any:
    """
    Parse message from a dictionary.

    Args:
        message: The dictionary to parse.

    Returns:
        The parsed message.
    """
    validate_type(message.get("role"), str, "role")
    validate_content(
        message.get("role"), ["user", "assistant", "system", "bot", "chatbot", "model"]
    )
    validate_type_to_be_one_of(message.get("content"), [str, object], "type")
    if isinstance(message.get("content"), object):
        # Making sure if content has type and corresponding data
        content = message.get("content")
        validate_type(content.get("type"), str, "type")
        validate_content(content.get("type"), ["image_url", "text"])
        # Making sure type is image or text
        type = content.get("type")
        if type == "image_url":
            validate_type(content.get("image_url"), str, "image_url")
        elif type == "text":
            validate_type(content.get("text"), str, "text")
        else:
            raise ValueError(
                f"Invalid content type. We expect 'text' or 'image' type. Got: {type}"
            )
    return message


def parse_messages(messages: List[Any]) -> List[Any]:
    """
    Parse messages from a list.

    Args:
        messages: The list to parse.

    Returns:
        The parsed messages.
    """
    if len(messages) == 0:
        return []
    return [parse_message(message) for message in messages]


def parse_model_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse model parameters from a dictionary.

    Args:
        parameters: The dictionary to parse.

    Returns:
        The parsed model parameters.
    """
    # convert parameters dict into JSON string
    if parameters is None:
        return {}
    new_parameters = {}
    # we will go through each key and make sure it is a string
    # if not we will do json.dumps on it
    for key, value in parameters.items():
        if value is None:
            continue
        if not isinstance(value, str):
            try:
                new_parameters[key] = json.dumps(value, default=default_json_serializer)
            except Exception as e:
                scribe().warning(
                    f'[MaximSDK] Failed to stringify model_parameters key - "{key}": {e}. Skipping it'
                )
        else:
            new_parameters[key] = value
    return new_parameters
