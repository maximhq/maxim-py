import json
from typing import Any, Dict, List, Optional

from ...scribe import scribe
from .core import (
    validate_content,
    validate_optional_type,
    validate_type,
    validate_type_to_be_one_of,
)


def parse_function_call(function_call_data):
    validate_type(function_call_data.get("name"), str, "name")
    validate_type(function_call_data.get("arguments"), str, "arguments")
    return function_call_data


def parse_tool_calls(tool_calls_data):
    validate_type(tool_calls_data.get("id"), str, "id")
    validate_type(tool_calls_data.get("type"), str, "type")
    parse_function_call(tool_calls_data.get("function"))
    return tool_calls_data


def parse_content_list(content_list_data):
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
    validate_type(choice_data.get("index"), int, "index")
    validate_type(choice_data.get("finish_reason"), str, "finish_reason")

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
    if usage_data is None:
        return None
    validate_type(usage_data.get("prompt_tokens"), int, "prompt_tokens")
    validate_type(usage_data.get("completion_tokens"), int, "completion_tokens")
    validate_type(usage_data.get("total_tokens"), int, "total_tokens")
    return usage_data


def parse_generation_error(error_data):
    if error_data is None:
        return None
    validate_type(error_data.get("message"), str, "message")
    validate_optional_type(error_data.get("code"), str, "code")
    validate_optional_type(error_data.get("type"), str, "type")
    return error_data


def parse_result(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Text completion is not supported.")
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
    if len(messages) == 0:
        return []
    return [parse_message(message) for message in messages]


def parse_model_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
                new_parameters[key] = json.dumps(value)
            except Exception as e:
                scribe().error(
                    f'[MaximSDK] Failed to stringify model_parameters key - "{key}": {e}. Skipping it'
                )
        else:
            new_parameters[key] = value
    return new_parameters
