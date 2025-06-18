import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from uuid import uuid4

from google.genai.types import (
    Content,
    ContentDict,
    ContentListUnion,
    ContentListUnionDict,
    ContentUnion,
    File,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    Part,
    PartUnionDict,
    PIL_Image,
)

from ..logger import GenerationRequestMessage


def get_type_from_name(name: str) -> str:
    if name == "int":
        return "number"
    elif name == "str":
        return "string"
    elif name == "float":
        return "number"
    elif name == "bool":
        return "boolean"
    return "string"


class GeminiUtils:
    @staticmethod
    def parse_chat_message(
        role: str,
        message: Union[list[PartUnionDict], PartUnionDict],
    ) -> List[GenerationRequestMessage]:
        request_messages: List[GenerationRequestMessage] = []
        if isinstance(message, list):
            for m in message:
                if m is None:
                    continue
                if isinstance(m, dict):
                    pass
                else:
                    request_messages.append(
                        {
                            "role": role,
                            "content": GeminiUtils.parse_part_text(m) or "",
                        }
                    )
        elif isinstance(message, str):
            request_messages.append(
                {
                    "role": role,
                    "content": message,
                }
            )
        return request_messages

    @staticmethod
    def get_model_params(
        config: Optional[GenerateContentConfigOrDict] = None,
    ) -> Dict[str, Any]:
        model_params = {}
        if config is None:
            return model_params
        tools: Optional[Union[Any, List[Any]]] = None
        # We parse tool calls separately
        if isinstance(config, dict):
            for key, value in config.items():
                if key == "system_instruction" or key == "history":
                    continue
                elif key == "tools":
                    tools = value
                    continue
                model_params[key] = value
        elif isinstance(config, GenerateContentConfig):
            for key, value in config.dict():
                if key == "system_instruction" or key == "history":
                    continue
                elif key == "tools":
                    tools = value
                    continue
                model_params[key] = value

        if tools is not None:
            # parsing tools
            try:
                final_tools = []
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, Callable):
                            name = tool.__name__
                            description = tool.__doc__
                            args = []
                            for k, v in tool.__annotations__.items():
                                if k == "return":
                                    continue
                                args.append(
                                    {"name": k, "type": get_type_from_name(v.__name__)}
                                )
                            final_tools.append(
                                {
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "description": description,
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                arg["name"]: {"type": arg["type"]}
                                                for arg in args
                                            },
                                        },
                                    },
                                }
                            )
                    model_params["tools"] = final_tools
            except Exception as e:
                logging.error(f"[MaximSDK] Error parsing tools: {e}")

        return model_params

    @staticmethod
    def parse_messages(
        contents: Union[ContentListUnion, ContentListUnionDict],
    ) -> List[GenerationRequestMessage]:
        messages: List[GenerationRequestMessage] = []
        if isinstance(contents, list):
            for content in contents:
                messages.append(GeminiUtils.parse_content(content))
        else:
            messages.append(GeminiUtils.parse_content(contents))
        return messages

    @staticmethod
    def parse_part_text(part: Any) -> Optional[str]:
        if isinstance(part, str):
            return part
        elif isinstance(part, File):
            return part.download_uri
        elif isinstance(part, Part):
            return part.text
        elif isinstance(part, PIL_Image):
            return "<IMAGE IS NOT SUPPORTED>"
        return None

    @staticmethod
    def parse_content(
        content: Optional[Union[ContentUnion, ContentDict]],
        override_role: Optional[str] = None,
    ) -> GenerationRequestMessage:
        message: GenerationRequestMessage = {
            "role": override_role or "user",
            "content": "",
        }
        if isinstance(content, dict):
            if (
                role := content.get("role", None)
            ) is not None and override_role is None:
                message["role"] = role
            parts = content.get("parts", None)
            if parts is not None:
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("text", None)
                    else:
                        text = getattr(part, "text", None)
                    if text is not None and isinstance(message["content"], str):
                        message["content"] += text
        elif isinstance(content, Content):
            if content.role is not None and override_role is None:
                if content.role == "model":
                    message["role"] = "system"
                else:
                    message["role"] = content.role
            if content.parts is not None:
                for part in content.parts:
                    if (
                        part is not None
                        and part.text is not None
                        and isinstance(message["content"], str)
                    ):
                        message["content"] += part.text
        elif isinstance(content, List):
            for part in content:
                text = GeminiUtils.parse_part_text(part)
                if text is not None and isinstance(message["content"], str):
                    message["content"] += text
        else:
            text = GeminiUtils.parse_part_text(content)
            if text is not None and isinstance(message["content"], str):
                message["content"] += text

        return message

    @staticmethod
    def parse_gemini_generation_content_iterator(
        itr: Iterator[GenerateContentResponse],
    ) -> Dict[str, Any]:
        # Checking if it is correct or not
        if itr is None:
            raise ValueError("No response chunks")
        response: Dict[str, Any] = {}
        responses = []
        for chunk in itr:
            if not isinstance(chunk, GenerateContentResponse):
                raise ValueError("Invalid response chunk")
            responses.append(GeminiUtils.parse_gemini_generation_content(chunk))
        # merging these responses into one
        generation_result: Dict[str, Any] = {
            "id": str(uuid4()),
            "created": (
                responses[0].get("created", int(time.time()))
                if responses[0]
                else int(time.time())
            ),
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        text = ""
        for response in responses:
            text += response["choices"][0]["message"]["content"]
            generation_result["usage"]["prompt_tokens"] += response["usage"][
                "prompt_tokens"
            ]
            generation_result["usage"]["completion_tokens"] += response["usage"][
                "completion_tokens"
            ]
            generation_result["usage"]["total_tokens"] += response["usage"][
                "total_tokens"
            ]
        # Create a top-level choice and then keep individual choice
        generation_result["choices"].append(
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "logprobs": None,
                "finish_reason": responses[-1]["choices"][0]["finish_reason"],
            }
        )
        return generation_result

    @staticmethod
    def parse_gemini_generation_content(
        response: GenerateContentResponse,
    ) -> Dict[str, Any]:
        """
        Parse the generation response from Google's Generative AI API into a standardized format.

        Args:
            response (GenerateContentResponse): The raw response from the model

        Returns:
            Dict[str, Any]: A dictionary containing the parsed response with standardized fields:
                - usage: Token counts for prompt and completion
                - id: Generated UUID for the response
                - created: Unix timestamp of when response was parsed
        """
        generation_result: Dict[str, Any] = {}
        choices: List[Dict[str, Any]] = []
        tool_calls = []
        if (
            response.automatic_function_calling_history is not None
            and len(response.automatic_function_calling_history) > 1
        ):
            # we find the history of last model message
            model_tool_call: Optional[Content] = (
                response.automatic_function_calling_history[-2]
            )
            if model_tool_call is not None and model_tool_call.parts is not None:
                part = model_tool_call.parts[0]
                if part.function_call is not None:
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": part.function_call.id or str(uuid4()),
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args),
                            },
                        }
                    )
        if response.candidates is not None:
            for candidate in response.candidates:
                content = ""
                if (
                    candidate.content is not None
                    and candidate.content.parts is not None
                ):
                    for part in candidate.content.parts:
                        if part.text is not None:
                            content += part.text
                choices.append(
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        },
                        "logprobs": (
                            candidate.logprobs_result.chosen_candidates
                            if candidate.logprobs_result is not None
                            else None
                        ),
                        "finish_reason": (
                            candidate.finish_reason.value
                            if candidate.finish_reason is not None
                            else ""
                        ),
                    }
                )
        prompt_tokens: int = 0
        completion_tokens: int = 0
        if response.usage_metadata is not None:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
        generation_result["choices"] = choices
        generation_result["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        generation_result["id"] = str(uuid4())
        generation_result["created"] = int(time.time())
        return generation_result
