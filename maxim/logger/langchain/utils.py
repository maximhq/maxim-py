import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.generation import Generation, GenerationChunk

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
    mapping = {
        "azure": "azure",
        "anthropic": "anthropic",
        "huggingface": "huggingface",
        "bedrock": "bedrock",
        "aws": "bedrock",
        "openai": "openai",
        "groq": "groq",
        "ollama": "ollama",
        "gemini": "google",
        "vertexai": "google",
        "deepseek": "deepseek",
        "qwen": "qwen",
    }
    for key, target in mapping.items():
        if key in provider:
            return target
    return "unknown"


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
    print(f"MESSAGE TYPE: {message_type}")
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

def check_for_computer_action(additional_kwargs):
    try:
        tool_outputs = additional_kwargs["tool_outputs"][0]
        action = tool_outputs["action"]
        return True
    except:
        return False


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
        elif check_for_computer_action(ai_message.additional_kwargs):
            content = json.dumps(ai_message.additional_kwargs["tool_outputs"])
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


def parse_langchain_generation_chunk(generation: GenerationChunk):
    return [
        {
            "index": 0,
            "text": generation.text,
            "logprobs": generation.generation_info.get("logprobs")
            if generation.generation_info
            else None,
            "finish_reason": generation.generation_info.get("finish_reason")
            if generation.generation_info
            else "stop",
        }
    ]


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
    elif isinstance(generation, GenerationChunk):
        scribe().debug("[MaximSDK][Langchain] Parsing GenerationChunk")
        return parse_langchain_generation_chunk(generation)
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
            if generation is None:
                continue
            for _, gen in enumerate(generation):
                if gen is None or isinstance(gen, str):
                    continue
                usage_data = None
                if "message" in gen.__dict__:
                    message_obj = gen.__dict__.get("message")
                    if message_obj and hasattr(message_obj, "__dict__"):
                        usage_data = message_obj.__dict__.get("usage_metadata")
                elif (
                    "generation_info" in gen.__dict__
                    and gen.__dict__.get("generation_info", None) is not None
                ):
                    usage_data = gen.__dict__.get("generation_info", {}).get(
                        "usage_metadata"
                    )
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
                    elif usage_data.get("prompt_token_count") is not None:
                        prompt_tokens += usage_data.get("prompt_token_count", 0)
                        output_tokens += usage_data.get("candidates_token_count", 0)
                        total_tokens += usage_data.get(
                            "prompt_token_count", 0
                        ) + usage_data.get("completion_token_count", 0)
                        continue
                message_obj = gen.__dict__.get("message")
                resp_metadata = None
                if message_obj and hasattr(message_obj, "__dict__"):
                    resp_metadata = message_obj.__dict__.get("response_metadata")
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
                        messages.append({"role": default_role, "content": message})
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

# def parse_computer_use_outputs(messages:list):
#     for msg in messages:
#         if isinstance(msg, str):
#             pass
#         if isinstance(msg, dict):
#             if "route" in msg:
#                 #TODO: handle route condition
#                 pass
#             elif "instance_id" in msg:
#                 #TODO: handle vm logging condition
#                 pass
#             elif "messages" in msg:
#                 message = msg["messages"]
#                 if isinstance(message, dict):
#                     #TODO: screenshot (tool as defined by CUA agent, NOT a tool call)
#                     pass
#                 elif isinstance(message, AIMessage):
#                     additional_kwargs = AIMessage.additional_kwargs
#                     tool_outputs = additional_kwargs["tool_outputs"][0]
#                     id = tool_outputs["id"]
#                     action = tool_outputs["action"]
#                     call_id = tool_outputs["call_id"]

#                     #TODO: computer action
#                     pass


#     pass




#     case 1: str -> "title"
#     case 2: dict: routing -> key:route
#                 : create_vm_instance -> key: instance_id
#                 : screenshots ->  key: "messages", val: {'role':'tool', 'content'...}. 
#                 : action : AIMessage NO CONTENT but action in additional_kwargs: types = [screenshot, click, type, keypress]
#                 : return text ->  AIMessage with content



"""
{'messages': [SystemMessage(content="You're an advanced AI computer use assistant. The browser you are using is already initialized, and visiting google.com.", additional_kwargs={}, response_metadata={}, id='517ca8d9-bbc2-441c-8649-8006da5af591'), HumanMessage(content="find today's top 1 song on billboard's website and return for the artist's instagram handle", additional_kwargs={}, response_metadata={}, id='8418a4ea-0a60-4b3b-ac5a-6e2bc4c0ac2e'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662b6703a48198898862cb49b8a0ad05713ae5da8f4aa9', 'action': {'type': 'screenshot'}, 'call_id': 'call_F4mVRJtsFyoNFAVbbNCw7mBd', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b669bbc8198be8e794003a5bd5c05713ae5da8f4aa9', 'created_at': 1751526246.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--89c73650-d364-440f-91d4-b1ff545c4399-0', usage_metadata={'input_tokens': 535, 'output_tokens': 7, 'total_tokens': 542, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='9388d765-78e4-45e4-8eac-aae612ca259b', tool_call_id='call_F4mVRJtsFyoNFAVbbNCw7mBd'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662b6f7a408198b26473992f9e2b9805713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662b71496481989963d8a74236558705713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 415, 'y': 59}, 'call_id': 'call_2YA3Azh4Hu7dRfOrrndkdFp8', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b6e0f648198899a7dc89098e8d005713ae5da8f4aa9', 'created_at': 1751526254.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--f694aa27-0f95-40c1-9700-db6ccfc74f09-0', usage_metadata={'input_tokens': 2311, 'output_tokens': 68, 'total_tokens': 2379, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='9f418410-75f2-444a-a557-aa9207430a7d', tool_call_id='call_2YA3Azh4Hu7dRfOrrndkdFp8'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662b775a6c8198afa775fa7b38433405713ae5da8f4aa9', 'action': {'text': 'Billboard', 'type': 'type'}, 'call_id': 'call_tDTPe0lf3DQXzmM9U87i0avC', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b747e7c8198964c56cad1b91d8705713ae5da8f4aa9', 'created_at': 1751526260.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--1729363b-3a53-431e-b996-2a99bf405e7a-0', usage_metadata={'input_tokens': 1792, 'output_tokens': 16, 'total_tokens': 1808, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='9cdaf9b2-6fd2-47f3-a879-7f9311440eb1', tool_call_id='call_tDTPe0lf3DQXzmM9U87i0avC'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662b7caef08198bf43b02caf25bf4605713ae5da8f4aa9', 'action': {'keys': ['ENTER'], 'type': 'keypress'}, 'call_id': 'call_QgoystGlMJfxMvvS92GabDXy', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b7a9330819891173284618e338305713ae5da8f4aa9', 'created_at': 1751526266.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--8f01ee12-3a2a-4fd2-b37b-eec039fbe5c3-0', usage_metadata={'input_tokens': 2398, 'output_tokens': 17, 'total_tokens': 2415, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='c2eb5632-9685-4c7a-8eda-7d0a58084b73', tool_call_id='call_QgoystGlMJfxMvvS92GabDXy'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662b8334008198962150124eda75a605713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662b85fc9c819899a4972e9b29330505713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 260, 'y': 397}, 'call_id': 'call_4C8J73EFhhWYh5AI2cPOZ6h3', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b80109c819888b0785f0c9255ff05713ae5da8f4aa9', 'created_at': 1751526272.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--a1bf70d1-e36e-4b9e-9abd-5d7cbd813d36-0', usage_metadata={'input_tokens': 6055, 'output_tokens': 64, 'total_tokens': 6119, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='db252ed4-cb7a-433d-9aaf-892b192dff0c', tool_call_id='call_4C8J73EFhhWYh5AI2cPOZ6h3'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662b8d738c8198b8d2175ce21a0fb105713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662b903aa08198a112821390af33ae05713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 232, 'y': 227}, 'call_id': 'call_H1fPl6YSqD6SrgMSbg2GSSKL', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b8a9d948198b4f1c0b273acb1f005713ae5da8f4aa9', 'created_at': 1751526282.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--2d9d4fd4-8efd-4fd8-8d6e-2e5edc206e3f-0', usage_metadata={'input_tokens': 7379, 'output_tokens': 76, 'total_tokens': 7455, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='7573d764-4c43-4bdf-9c19-64c22865bae2', tool_call_id='call_H1fPl6YSqD6SrgMSbg2GSSKL'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662b99aff48198ba5bd7aba361ec1105713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662b9d7c108198be405b4327231da305713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 148, 'y': 18}, 'call_id': 'call_AFapoP0O92hXJP4ep8HepbuT', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662b95e7e0819880bb1cdde9e9db2b05713ae5da8f4aa9', 'created_at': 1751526294.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--1c62244e-87bd-4d49-a3da-7161f58c050e-0', usage_metadata={'input_tokens': 8712, 'output_tokens': 73, 'total_tokens': 8785, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='50059004-4a01-43ff-b7d1-ce9ca41a70f9', tool_call_id='call_AFapoP0O92hXJP4ep8HepbuT'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662ba5c30881988a1c86b5b852886d05713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662ba9109c81989f90eb1b5676bb8605713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 232, 'y': 231}, 'call_id': 'call_oq13aQvTGWwVw9nidvT2z6f9', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662ba236b88198a5d0f1b981058cab05713ae5da8f4aa9', 'created_at': 1751526306.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--c93cd670-ec5b-4526-baa7-8ef6e3f59d3f-0', usage_metadata={'input_tokens': 10024, 'output_tokens': 55, 'total_tokens': 10079, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='3d8d4c6d-cc8f-46fc-94b7-4d937fe89868', tool_call_id='call_oq13aQvTGWwVw9nidvT2z6f9'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662bb1db7c81989dbbe77bf429fa8305713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662bb606e88198b656b3e4bd66de2e05713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 851, 'y': 101}, 'call_id': 'call_UKJY8zOfQdGv401FWzjzh1to', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662bade7048198b0d5beb7f1baa4d105713ae5da8f4aa9', 'created_at': 1751526318.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--e3718395-48e6-427c-a12c-82b2e4555f6a-0', usage_metadata={'input_tokens': 10181, 'output_tokens': 74, 'total_tokens': 10255, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='44f85a60-15bd-4e26-988a-513a50670958', tool_call_id='call_UKJY8zOfQdGv401FWzjzh1to'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662bbee7fc81988741c84201bf370005713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662bc301d48198970b7e05320f574a05713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 729, 'y': 19}, 'call_id': 'call_PNtneCfaw47dATU5fn4K32SH', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662bbb0dc48198b4690ad5dab91e4605713ae5da8f4aa9', 'created_at': 1751526331.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--e625a257-87ad-4cc7-8003-97c961e0c608-0', usage_metadata={'input_tokens': 10336, 'output_tokens': 53, 'total_tokens': 10389, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='db4ae0ab-a8cb-4456-98ec-b685208c959b', tool_call_id='call_PNtneCfaw47dATU5fn4K32SH'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662bcc9ee88198bc05325694740b6d05713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662bd0b23c8198b4cd2b167660e45605713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 391, 'y': 22}, 'call_id': 'call_EivZQTL2bC2QkUUYp1EsovJW', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662bc7fbb081989841b7d2dd4bf55b05713ae5da8f4aa9', 'created_at': 1751526344.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--215f413b-0291-490b-8c25-cba670e597a6-0', usage_metadata={'input_tokens': 10480, 'output_tokens': 63, 'total_tokens': 10543, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='cc9fe9db-c406-49b1-8cdc-b57a3c45e8e1', tool_call_id='call_EivZQTL2bC2QkUUYp1EsovJW'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662bdb2dc88198850fc84e1836473d05713ae5da8f4aa9', 'action': {'type': 'wait'}, 'call_id': 'call_l5w3sJ3YTxrsqOCf9FdMYre9', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662bd6eb8c81989b5fed2f2fedb76205713ae5da8f4aa9', 'created_at': 1751526359.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--2f1bccec-4d86-4d51-b0c2-10b15a00ae9e-0', usage_metadata={'input_tokens': 5296, 'output_tokens': 11, 'total_tokens': 5307, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='87e57054-f40b-4b34-bff9-8abafde9bd73', tool_call_id='call_l5w3sJ3YTxrsqOCf9FdMYre9'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662be5f7388198870c8ef61b6addab05713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662bea2ef481988fb135b3d698e5f605713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 257, 'y': 19}, 'call_id': 'call_Q7ojOK0Sw4LoDU08ev1hLtr2', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662be176f88198944e4d468bb1c0b505713ae5da8f4aa9', 'created_at': 1751526369.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--c26f2bec-827f-4829-a9c4-f68b28b8e49a-0', usage_metadata={'input_tokens': 10664, 'output_tokens': 47, 'total_tokens': 10711, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='01eb5193-d453-4e21-8d8c-53b74cc8ca50', tool_call_id='call_Q7ojOK0Sw4LoDU08ev1hLtr2'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662bf50ce48198b7d54c615d3bada805713ae5da8f4aa9', 'action': {'type': 'wait'}, 'call_id': 'call_UtFmLpYHXb0ckr7393y9X8wp', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662beff34c8198826ac27131d3bf1805713ae5da8f4aa9', 'created_at': 1751526384.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--88ca83b7-88e5-4de5-a49c-b74f75b81b45-0', usage_metadata={'input_tokens': 5380, 'output_tokens': 11, 'total_tokens': 5391, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='438d7c1d-1c54-45e2-b932-ac85a3b21486', tool_call_id='call_UtFmLpYHXb0ckr7393y9X8wp'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662bfff0ec8198a9fb22989c96af8105713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662c0430a48198a51e8a22fe219e2605713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 382, 'y': 12}, 'call_id': 'call_hy7VkWuqaX1bu5PLOzel6HM2', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662bfb94288198b27bfc8b4a2b942405713ae5da8f4aa9', 'created_at': 1751526395.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--e66507d8-6189-4261-842e-424fe88ab56d-0', usage_metadata={'input_tokens': 10838, 'output_tokens': 53, 'total_tokens': 10891, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='4d78f261-91bd-44c5-8b41-1359b1e9f3a0', tool_call_id='call_hy7VkWuqaX1bu5PLOzel6HM2'), AIMessage(content=[], additional_kwargs={'tool_outputs': [{'id': 'cu_68662c0e2fdc819897c0723c44f7975305713ae5da8f4aa9', 'action': {'type': 'wait'}, 'call_id': 'call_YoAVJ6v8bHPmgPu6fiU9Gc9T', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662c096bb88198a3ac53d2c736234f05713ae5da8f4aa9', 'created_at': 1751526409.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--9c304689-28b1-4546-a5d9-675930e1b526-0', usage_metadata={'input_tokens': 5470, 'output_tokens': 11, 'total_tokens': 5481, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='a0e1cbba-c45d-4dee-adea-95c47cf749e4', tool_call_id='call_YoAVJ6v8bHPmgPu6fiU9Gc9T'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662c188e048198bfd33cc37c025ed905713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662c1d3c1c81988652ba45a6979fff05713ae5da8f4aa9', 'action': {'button': 'right', 'type': 'click', 'x': 371, 'y': 24}, 'call_id': 'call_CKJOx1B9d3bSt8ECUyYc8SoB', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662c143f3c8198b574914c7e7e942905713ae5da8f4aa9', 'created_at': 1751526420.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--6c82d935-cf5d-4442-9e3f-a71e03e395dc-0', usage_metadata={'input_tokens': 11040, 'output_tokens': 75, 'total_tokens': 11115, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='383ed4fa-81f6-4103-884b-ff5b136ea8bf', tool_call_id='call_CKJOx1B9d3bSt8ECUyYc8SoB'), AIMessage(content=[], additional_kwargs={'reasoning': {'id': 'rs_68662c26bd6081988654134a9979542e05713ae5da8f4aa9', 'summary': [], 'type': 'reasoning'}, 'tool_outputs': [{'id': 'cu_68662c2b27248198af8842dc2b6c88ae05713ae5da8f4aa9', 'action': {'button': 'left', 'type': 'click', 'x': 255, 'y': 18}, 'call_id': 'call_Rk6Vi7PfyVVkKG1mPGHn21nD', 'pending_safety_checks': [], 'status': 'completed', 'type': 'computer_call'}]}, response_metadata={'id': 'resp_68662c22036c8198a66aead57b675b7b05713ae5da8f4aa9', 'created_at': 1751526434.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='run--a3b38c5e-53a8-464a-8f80-25001e77e95f-0', usage_metadata={'input_tokens': 11235, 'output_tokens': 92, 'total_tokens': 11327, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}), ToolMessage(content=[{'type': 'input_image', 'image_url': 'IMAGE DATA'}], id='545db022-64a3-458b-a925-fbb83ebbe54a', tool_call_id='call_Rk6Vi7PfyVVkKG1mPGHn21nD'), AIMessage(content=[{'type': 'text', 'text': 'An irrelevant tab opened. What would you like to do?', 'annotations': []}], additional_kwargs={}, response_metadata={'id': 'resp_68662c3090588198a3bbf7431d746a4705713ae5da8f4aa9', 'created_at': 1751526448.0, 'metadata': {}, 'model': 'computer-use-preview-2025-03-11', 'object': 'response', 'service_tier': 'default', 'status': 'completed', 'model_name': 'computer-use-preview-2025-03-11'}, id='msg_68662c34e33c8198a926f0da4166a2af05713ae5da8f4aa9', usage_metadata={'input_tokens': 5688, 'output_tokens': 14, 'total_tokens': 5702, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})], 'instance_id': 's-57dc1a9a', 'stream_url': 'https://api.proxy.scrapybara.com/v1/instance/s-57dc1a9a/stream', 'authenticated_id': None}
"""