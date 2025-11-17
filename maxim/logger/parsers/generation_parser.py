import enum
import json
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
    if hasattr(o, "to_dict"):
        return o.to_dict()

    try:
        return vars(o)
    except TypeError:
        pass

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
