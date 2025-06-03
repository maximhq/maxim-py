import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.generation import Generation

from ...scribe import scribe
from ..components.types import GenerationError

logger = logging.getLogger("MaximSDK")


def parse_langchain_model_and_provider(model: str, provider: str) -> Tuple[str, str]:
    return model, provider


def parse_langchain_provider(serialized: Dict[str, Any]):
    """Parses langchain provider from serialized data
    Args:
        serialized: Dict[str, Any]: Serialized data to parse provider from
    Returns:
        str: Parsed provider
    """
    provider = serialized.get("name", "").lower()
    if provider.startswith("chat"):
        return provider.replace("chat", "")
    elif "azure" in provider:
        return "azure"
    elif "anthropic" in provider:
        return "anthropic"
    elif "huggingface" in provider:
        return "huggingface"
    elif "bedrock" in provider:
        return "bedrock"
    elif "aws" in provider:
        return "bedrock"
    return provider


def parse_langchain_llm_error(
    error: Union[Exception, BaseException, KeyboardInterrupt],
) -> GenerationError:
    """Parses langchain LLM error into a format that is accepted by Maxim logger
    Args:
        error: Union[Exception, KeyboardInterrupt]: Error to be parsed
    Returns:
        GenerationError: Parsed LLM error
    """
    if isinstance(error, KeyboardInterrupt):
        return GenerationError(message="Generation was interrupted by the user")
    if isinstance(error, Exception):
        return GenerationError(message=str(error))
    else:
        message = error.__dict__.get("message", "")
        type = error.__dict__.get("type", None)
        code = error.__dict__.get("code", None)
        return GenerationError(message=message, type=type, code=code)


def parse_langchain_model_parameters(**kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    """Parses langchain kwargs into model and model parameters. You can use this function with any langchain _start callback function
    Args:
        kwargs: Dict[str, Any]: Kwargs to be parsed
    Returns:
        Tuple[str, Dict[str, Any]]: Model and model parameters
    Raises:
        Exception: If model_name is not found in kwargs
    """
    model_parameters = kwargs.get("invocation_params", {})
    # Checking if model_name present
    model = "unknown"
    if "model_name" in model_parameters:
        model = model_parameters["model_name"]
        del model_parameters["model_name"]
    # If not then checking if invocation_params exist
    elif "model" in model_parameters:
        model = model_parameters["model"]
        del model_parameters["model"]
    elif "model_id" in model_parameters:
        model = model_parameters["model_id"]
        del model_parameters["model_id"]
    return model, model_parameters


def parse_base_message_to_maxim_generation(message: BaseMessage):
    """Parses langchain BaseMessage into a format that is accepted by Maxim logger
    Args:
        message: BaseMessage
    Returns:
        Dict[str, Any]: Parsed message
    """
    choice = parse_langchain_message(message)
    usage = (
        message.__dict__["usage_metadata"] if message.__dict__["usage_metadata"] else {}
    )
    return {
        "id": str(uuid4()),
        "created": int(time.time()),
        "choices": [choice],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def parse_langchain_message(message: BaseMessage):
    """Parses langchain BaseMessage into a choice of openai message
    Args:
        message: BaseMessage
    Returns:
        Dict[str, Any]: Parsed message
    """
    message_type = message.__dict__["type"]
    if message_type == "ai":
        ai_message = AIMessage(
            content=message.content, additional_kwargs=message.additional_kwargs
        )
        tool_calls = (
            ai_message.tool_calls or message.lc_attributes.get("tool_calls") or []
        )
        return {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    message.content
                    if isinstance(message.content, str)
                    else json.dumps(message.content)
                ),
                "tool_calls": parse_langchain_tool_call(tool_calls),
            },
            "finish_reason": (
                message.response_metadata.get("finish_reason")
                or message.response_metadata.get("stop_reason")
                or "stop"
            ),
            "logprobs": (message.response_metadata.get("logprobs") or None),
        }
    else:
        return {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message.content,
            },
            "finish_reason": message.response_metadata["finish_reason"]
            if message.response_metadata["finish_reason"]
            else None,
            "logprobs": message.response_metadata["logprobs"]
            if message.response_metadata["logprobs"]
            else None,
        }


def parse_langchain_tool_call(tool_calls: List[ToolCall]):
    final_tool_calls = []
    for tool_call in tool_calls:
        try:
            final_tool_calls.append(
                {
                    "type": "function",
                    "id": tool_call.get("id", str(uuid4())),
                    "function": {
                        "name": tool_call.get("name", "unknown"),
                        "arguments": json.dumps(tool_call.get("args", {})),
                    },
                }
            )
        except AttributeError as e:
            scribe().debug(f"Error parsing tool call: {str(e)}")
            # trying the dict way
            dict_tool_call = tool_call.__dict__
            final_tool_calls.append(
                {
                    "type": "function",
                    "id": dict_tool_call.get("id", str(uuid4())),
                    "function": {
                        "name": dict_tool_call.get("name", "unknown"),
                        "arguments": json.dumps(dict_tool_call.get("args", {})),
                    },
                }
            )
        except Exception as e:
            scribe().error(f"Error parsing tool call: {str(e)}")
    return final_tool_calls


def parse_langchain_chat_generation_chunk(generation: ChatGeneration):
    choices = []
    content = generation.text
    finish_reason = (
        generation.message.response_metadata.get("stop_reason")
        if generation.message.response_metadata
        else "stop"
    )
    if finish_reason is None:
        finish_reason = (
            generation.generation_info.get("finish_reason")
            if generation.generation_info
            else "stop"
        )
    choices.append(
        {
            "index": 0,
            "message": {"role": "assistant", "content": content, "tool_calls": []},
            "finish_reason": finish_reason,
            "logprobs": generation.generation_info.get("logprobs")
            if generation.generation_info
            else None,
        }
    )
    return choices


def parse_langchain_chat_generation(generation: ChatGeneration):
    choices = []
    message = generation.message
    if message.type == "ai":
        ai_message = AIMessage(
            content=message.content, additional_kwargs=message.additional_kwargs
        )
        finish_reason = (
            generation.message.response_metadata.get("stop_reason")
            if generation.message.response_metadata
            else "stop"
        )
        if finish_reason is None:
            finish_reason = (
                generation.generation_info.get("finish_reason")
                if generation.generation_info
                else "stop"
            )
        tool_calls = (
            ai_message.tool_calls or message.lc_attributes.get("tool_calls") or []
        )
        content = ""
        if isinstance(ai_message.content, str):
            content = ai_message.content
        elif isinstance(ai_message.content, list):
            for item in ai_message.content:
                if isinstance(item, str):
                    content += item
                elif isinstance(item, dict):
                    if "type" in item and item["type"] == "text":
                        content += item["text"]
                    elif "type" in item and item["type"] == "image_url":
                        content += item["image_url"]
                    else:
                        content += json.dumps(item)
        else:
            content = json.dumps(ai_message.content)
        choices.append(
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": parse_langchain_tool_call(tool_calls),
                },
                "finish_reason": finish_reason,
                "logprobs": generation.generation_info.get("logprobs")
                if generation.generation_info
                else None,
            }
        )
    return choices


def parse_langchain_text_generation(generation: Generation):
    choices = []
    messages = parse_langchain_messages([generation.text], "system")
    if len(messages) > 0:
        for i, message in enumerate(messages):
            choices.append(
                {
                    "index": i,
                    "text": message["content"],
                    "logprobs": generation.generation_info.get("logprobs")
                    if generation.generation_info
                    else None,
                    "finish_reason": generation.generation_info.get("finish_reason")
                    if generation.generation_info
                    else None,
                }
            )
    return choices


def parse_langchain_generation(generation: Generation):
    """Parses langchain generation into a format that is accepted by Maxim logger
    Args:
        generation: Generation: Generation to be parsed
    Returns:
        Dict[str, Any]: Parsed generation
    """
    # Sequence of checks matter here as ChatGenerationChunk is a subclass of ChatGeneration
    if isinstance(generation, ChatGenerationChunk):
        scribe().debug("[MaximSDK][Langchain] Parsing ChatGenerationChunk")
        return parse_langchain_chat_generation_chunk(generation)
    elif isinstance(generation, ChatGeneration):
        scribe().debug("[MaximSDK][Langchain] Parsing ChatGeneration")
        return parse_langchain_chat_generation(generation)
    elif isinstance(generation, Generation):
        scribe().debug("[MaximSDK][Langchain] Parsing Generation")
        return parse_langchain_text_generation(generation)


def parse_token_usage_for_result(result: LLMResult):
    """
    Parses token usage for a given LLM result
    """
    usage = result.llm_output.get("token_usage") if result.llm_output else None
    if usage is not None:
        return usage
    llm_usage = result.llm_output.get("usage") if result.llm_output else None
    if llm_usage:
        if llm_usage.get("input_tokens") is not None:
            usage = {
                "prompt_tokens": llm_usage.get("input_tokens", 0),
                "completion_tokens": llm_usage.get("output_tokens", 0),
                "total_tokens": llm_usage.get("input_tokens", 0)
                + llm_usage.get("output_tokens", 0),
            }
        elif llm_usage.get("prompt_tokens") is not None:
            usage = {
                "prompt_tokens": llm_usage.get("prompt_tokens", 0),
                "completion_tokens": llm_usage.get("completion_tokens", 0),
                "total_tokens": llm_usage.get("prompt_tokens", 0)
                + llm_usage.get("completion_tokens", 0),
            }
    if usage is not None:
        return usage
    # Here we might have to go down to each generation and sum up all usages
    prompt_tokens = 0
    output_tokens = 0
    total_tokens = 0
    generations = result.generations
    if generations is not None:
        for _, generation in enumerate(generations):
            for _, gen in enumerate(generation):
                usage_data = gen.__dict__.get("message").__dict__.get("usage_metadata")
                if usage_data is not None:
                    if usage_data.get("input_tokens") is not None:
                        prompt_tokens += usage_data.get("input_tokens", 0)
                        output_tokens += usage_data.get("output_tokens", 0)
                        total_tokens += usage_data.get(
                            "input_tokens", 0
                        ) + usage_data.get("output_tokens", 0)
                        continue
                    elif usage_data.get("prompt_tokens") is not None:
                        prompt_tokens += usage_data.get("prompt_tokens", 0)
                        output_tokens += usage_data.get("completion_tokens", 0)
                        total_tokens += usage_data.get(
                            "prompt_tokens", 0
                        ) + usage_data.get("completion_tokens", 0)
                        continue
                resp_metadata = gen.__dict__.get("message").__dict__.get(
                    "response_metadata"
                )
                if resp_metadata is not None:
                    usage_data = resp_metadata.get("usage") or None
                    if usage_data is not None:
                        if usage_data.get("input_tokens") is not None:
                            prompt_tokens += usage_data.get("input_tokens", 0)
                            output_tokens += usage_data.get("output_tokens", 0)
                            total_tokens += usage_data.get(
                                "input_tokens", 0
                            ) + usage_data.get("output_tokens", 0)
                            continue
                        elif usage_data.get("prompt_tokens") is not None:
                            prompt_tokens += usage_data.get("prompt_tokens", 0)
                            output_tokens += usage_data.get("completion_tokens", 0)
                            total_tokens += usage_data.get(
                                "prompt_tokens", 0
                            ) + usage_data.get("completion_tokens", 0)
                            continue
                    # The last case we check for is bedrock
                    usage_data = (
                        resp_metadata.get("amazon-bedrock-invocationMetrics")
                        if resp_metadata
                        else None
                    )
                    if usage_data is not None:
                        prompt_tokens += usage_data.get("inputTokenCount", 0)
                        output_tokens += usage_data.get("outputTokenCount", 0)
                        total_tokens += usage_data.get(
                            "inputTokenCount", 0
                        ) + usage_data.get("outputTokenCount", 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

def parse_langchain_chat_result(result: ChatResult) -> Dict[str, Any]:
    """Parses langchain Chat result into a format that is accepted by Maxim logger
    Args:
        result: ChatResult: Chat result to be parsed
    Returns:
        Dict[str, Any]: Parsed Chat result
    Raises:
        Exception: If error parsing Chat result
    """
    try:
        generations = result.generations
        choices = []
        if generations is not None:
            for _, generation in enumerate(generations):
                for _, gen in enumerate(generation):
                    parsed_generations = parse_langchain_generation(gen)
                    if isinstance(parsed_generations, list):
                        choices.extend(parsed_generations)
                    else:
                        choices.append(parsed_generations)
        usage = parse_token_usage_for_result(result)
        # Adding index to each choice
        for i, choice in enumerate(choices):
            choices[i] = {**choice, "index": i}
        return {
            "id": str(uuid4()),
            "created": int(time.time()),
            "choices": choices,
            "usage": usage,
        }
    except Exception as e:
        logger.error(f"Error parsing LLM result: {str(e)}")
        raise Exception(f"Error parsing LLM result: {str(e)}") from e


def parse_langchain_llm_result(result: LLMResult) -> Dict[str, Any]:
    """Parses langchain LLM result into a format that is accepted by Maxim logger
    Args:
        result: LLMResult: LLM result to be parsed
    Returns:
        Dict[str, Any]: Parsed LLM result
    Raises:
        Exception: If error parsing LLM result
    """
    try:
        generations = result.generations
        choices = []
        if generations is not None:
            for _, generation in enumerate(generations):
                for _, gen in enumerate(generation):
                    parsed_generations = parse_langchain_generation(gen)
                    if isinstance(parsed_generations, list):
                        choices.extend(parsed_generations)
                    else:
                        choices.append(parsed_generations)
        usage = parse_token_usage_for_result(result)
        # Adding index to each choice
        for i, choice in enumerate(choices):
            choices[i] = {**choice, "index": i}
        return {
            "id": str(uuid4()),
            "created": int(time.time()),
            "choices": choices,
            "usage": usage,
        }
    except Exception as e:
        logger.error(f"Error parsing LLM result: {str(e)}")
        raise Exception(f"Error parsing LLM result: {str(e)}")


def parse_langchain_messages(
    input: Union[List[str], List[List[Any]]], default_role="user"
):
    """Parses langchain messages into messages that are accepted by Maxim logger
    Args:
        input: List[str] or List[List[Any]]: List of messages to be parsed
        default_role: str: Default role to assign to messages without a role
    Returns:
        List[Dict[str, str]]: List of messages with role and content
    Raises:
        Exception: If input is not List[str] or List[List[Any]]
        Exception: If message type is not str or list
        Exception: If message type is not recognized
    """
    try:
        delimiter_to_role = {
            "System": "system",
            "Human": "user",
            "User": "user",
            "Assistant": "assistant",
            "Model": "model",
            "Tool": "tool",
        }
        messages = []
        # checking if input is List[str] or List[List]
        if isinstance(input[0], list):
            for message_list in input or []:
                for message in message_list:
                    if isinstance(message, str):
                        continue
                    message_type = type(message).__name__
                    if message_type.endswith("SystemMessage"):
                        messages.append(
                            {"role": "system", "content": message.content or ""}
                        )
                    elif message_type.endswith("HumanMessage"):
                        messages.append(
                            {"role": "user", "content": message.content or ""}
                        )
                    elif message_type.endswith("AIMessage"):
                        messages.append(
                            {"role": "assistant", "content": message.content or ""}
                        )
                    elif message_type.endswith("ToolMessage"):
                        messages.append(
                            {"role": "tool", "content": message.content or ""}
                        )
                    else:
                        logger.error(f"Invalid message type: {type(message)}")
                        raise Exception(f"Invalid message type: {type(message)}")
        else:
            for message in input or []:
                if not isinstance(message, str):
                    logger.error(f"Invalid message type: {type(message)}")
                    raise Exception(f"Invalid message type: {type(message)}")
                # get type of message
                # Define the delimiter pattern
                pattern = r"(System:|Human:|User:|Assistant:|Model:|Tool:)"
                # Split the text using the pattern
                splits = re.split(pattern, message)
                # Remove any leading/trailing whitespace and empty strings
                splits = [s.strip() for s in splits if s.strip()]
                # Pair up the delimiters with their content
                for i in range(0, len(splits), 2):
                    if i + 1 < len(splits):
                        # Remove ":" from delimiter and trim both delimiter and content
                        delimiter = splits[i].rstrip(":").strip()
                        content = splits[i + 1].strip()
                        messages.append(
                            {
                                "role": delimiter_to_role.get(delimiter, "user"),
                                "content": content,
                            }
                        )
                    else:
                        if splits[i].find(":") == -1:
                            messages.append(
                                {
                                    "role": delimiter_to_role.get(default_role, "user"),
                                    "content": splits[i],
                                }
                            )
                        else:
                            # Handle case where there's a delimiter without content
                            delimiter = splits[i].rstrip(":").strip()
                            messages.append(
                                {
                                    "role": delimiter_to_role.get(delimiter, "user"),
                                    "content": "",
                                }
                            )
        return messages
    except Exception as e:
        logger.error(f"Error parsing messages: {e}")
        raise Exception(f"Error parsing messages: {e}")
