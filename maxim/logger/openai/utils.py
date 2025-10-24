import time
from typing import Any, Dict, Iterable, List, Optional, Union

from openai._streaming import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import ResponseInputParam

from ..logger import GenerationRequestMessage


class OpenAIUtils:
    @staticmethod
    def parse_message_param(
        messages: Iterable[Dict[str, Any]],
        override_role: Optional[str] = None,
    ) -> List[GenerationRequestMessage]:
        parsed_messages: List[GenerationRequestMessage] = []

        for msg in messages:
            role = override_role or msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle content blocks for multimodal
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=text_content)
                )
            else:
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=str(content))
                )

        return parsed_messages

    @staticmethod
    def get_model_params(
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_params = {}
        skip_keys = ["messages", "tools"]
        max_tokens = kwargs.get("max_tokens", None)
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens

        param_keys = [
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "tools",
            "tool_choice",
        ]
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None and key not in skip_keys:
                model_params[key] = kwargs[key]

        for key, value in kwargs.items():
            if key not in param_keys and key not in skip_keys and value is not None:
                model_params[key] = value

        return model_params

    @staticmethod
    def get_responses_model_params(**kwargs: Any) -> Dict[str, Any]:
        model_params: Dict[str, Any] = {}
        skip_keys = [
            "input",
            "extra_headers",
            "model"
        ]

        for key, value in kwargs.items():
            if key not in skip_keys and value is not None:
                model_params[key] = value

        return model_params

    @staticmethod
    def parse_responses_input_to_messages(
        input_value: Union[Union[str, ResponseInputParam], None],
    ) -> List[GenerationRequestMessage]:
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [GenerationRequestMessage(role="user", content=input_value)]

        def _content_from_input_message_list(items: Any) -> Any:
            # items is ResponseInputMessageContentListParam (list of text/image/file)
            content_list: List[Dict[str, Any]] = []
            if not isinstance(items, list):
                return str(items)
            for item in items:
                if not isinstance(item, dict):
                    # fall back to string
                    content_list.append({"type": "text", "text": str(item)})
                    continue
                t = item.get("type")
                if t == "input_text" and "text" in item:
                    content_list.append({"type": "text", "text": str(item.get("text", ""))})
                elif t == "input_image":
                    # Map to our image content. Prefer image_url, otherwise file_id.
                    image_url = item.get("image_url")
                    file_id = item.get("file_id")
                    url_val = image_url or (f"file:{file_id}" if file_id else None)
                    if url_val:
                        content_list.append({"type": "image", "image_url": url_val})
                    else:
                        content_list.append({"type": "text", "text": "[image]"})
                elif t == "input_file":
                    # Represent non-image file input succinctly
                    name = item.get("filename") or item.get("file_url") or item.get("file_id")
                    content_list.append({"type": "text", "text": f"[file:{name}]"})
                else:
                    # Unknown structured item
                    content_list.append({"type": "text", "text": str(item)})
            return content_list if content_list else ""

        def _assistant_text_from_output_message(content: Any) -> str:
            # content is iterable of ResponseOutputTextParam | ResponseOutputRefusalParam
            if not isinstance(content, (list, tuple)):
                return str(content)
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t == "output_text":
                        txt = c.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                    elif t == "refusal":
                        ref = c.get("refusal")
                        if isinstance(ref, str):
                            parts.append(f"[refusal] {ref}")
                    else:
                        # unknown subtype
                        parts.append(str(c))
                else:
                    parts.append(str(c))
            return "".join(parts)

        def _summarize(obj: Dict[str, Any]) -> str:
            # Generic safe summary fallback
            try:
                typ = obj.get("type")
                if typ:
                    return f"[{typ}] " + ", ".join(
                        f"{k}={v}" for k, v in obj.items() if k != "type"
                    )
            except Exception:
                pass
            return str(obj)

        messages: List[GenerationRequestMessage] = []

        # input_value is expected to be a list (ResponseInputParam)
        items: Any = input_value
        if not isinstance(items, list):
            return [GenerationRequestMessage(role="user", content=str(items))]

        for item in items:
            # Each item can be one of ResponseInputItemParam union members
            if isinstance(item, str):
                messages.append(GenerationRequestMessage(role="user", content=item))
                continue

            if not isinstance(item, dict):
                messages.append(GenerationRequestMessage(role="user", content=str(item)))
                continue

            item_type = item.get("type")

            # EasyInputMessageParam or Message (both have role and content)
            if ("role" in item) and ("content" in item) and (
                item_type is None or item_type == "message"
            ):
                role_val = item.get("role") or "user"
                if not isinstance(role_val, str) or role_val.strip() == "":
                    role_val = "user"
                # Map developer -> system for our internal roles
                role_map = {"developer": "system"}
                final_role = role_map.get(role_val, role_val)
                content_val = item.get("content")
                if isinstance(content_val, str):
                    messages.append(GenerationRequestMessage(role=final_role, content=content_val))
                else:
                    messages.append(
                        GenerationRequestMessage(
                            role=final_role, content=_content_from_input_message_list(content_val)
                        )
                    )
                continue

            # ResponseOutputMessageParam (assistant)
            if item_type == "message" and item.get("role") == "assistant" and "content" in item:
                assistant_text = _assistant_text_from_output_message(item.get("content"))
                messages.append(GenerationRequestMessage(role="assistant", content=assistant_text))
                continue

            # Function/tool CALL intents (assistant role)
            if item_type in (
                "function_call",
                "file_search_call",
                "computer_call",
                "code_interpreter_call",
                "web_search_call",
                "local_shell_call",
                "image_generation_call",
            ):
                # Create a concise assistant message describing the call
                name = item.get("name") or item_type
                args = item.get("arguments") or item.get("queries") or item.get("action")
                call_id = item.get("call_id") or item.get("id")
                summary = f"{name} call"
                if call_id:
                    summary += f" id={call_id}"
                if args is not None:
                    summary += f" args={args}"
                messages.append(GenerationRequestMessage(role="assistant", content=summary))
                continue

            # Tool OUTPUTS (tool role)
            if item_type in ("function_call_output", "local_shell_call_output", "computer_call_output"):
                if item_type == "computer_call_output":
                    output = item.get("output", {})
                    if isinstance(output, dict) and output.get("type") == "computer_screenshot":
                        image_url = output.get("image_url")
                        if isinstance(image_url, str) and image_url:
                            messages.append(
                                GenerationRequestMessage(
                                    role="tool",
                                    content=[{"type": "image", "image_url": image_url}],
                                )
                            )
                            continue
                # Default: pass raw output as string
                out_val = item.get("output")
                messages.append(
                    GenerationRequestMessage(role="tool", content=str(out_val) if out_val is not None else "")
                )
                continue

            # MCP items
            if item_type in ("mcp_list_tools", "mcp_approval_request", "mcp_approval_response", "mcp_call"):
                # Heuristic: responses with output -> tool; otherwise assistant (call intent)
                if item_type == "mcp_call" and item.get("output") is not None:
                    messages.append(GenerationRequestMessage(role="tool", content=str(item.get("output"))))
                elif item_type == "mcp_approval_response":
                    messages.append(GenerationRequestMessage(role="tool", content=_summarize(item)))
                else:
                    messages.append(GenerationRequestMessage(role="assistant", content=_summarize(item)))
                continue

            # Reasoning item -> assistant
            if item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, (list, tuple)):
                    txt = "".join(s.get("text", "") for s in summary if isinstance(s, dict))
                else:
                    txt = str(summary)
                messages.append(GenerationRequestMessage(role="assistant", content=txt))
                continue

            # Item reference -> assistant note
            if item_type == "item_reference" or ("id" in item and item.get("type") is None and len(item) == 1):
                messages.append(GenerationRequestMessage(role="assistant", content=f"[item_reference] id={item.get('id')}"))
                continue

            # Unknown dict item -> user as final fallback
            messages.append(GenerationRequestMessage(role="user", content=str(item)))

        return messages

    @staticmethod
    def extract_responses_output_text(response: Any) -> Optional[str]:
        try:
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str):
                return output_text
        except Exception:
            pass
        # Fallback for dict-like structure
        try:
            if isinstance(response, dict):
                output = response.get("output", [])
                if isinstance(output, list):
                    texts: List[str] = []
                    for item in output:
                        if isinstance(item, dict) and item.get("type") in (
                            "output_text",
                            "text",
                        ):
                            text_val = item.get("text") or item.get("content")
                            if isinstance(text_val, str):
                                texts.append(text_val)
                    if texts:
                        return "".join(texts)
        except Exception:
            pass
        return None

    @staticmethod
    def extract_responses_usage(response: Any) -> Optional[Dict[str, Any]]:
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            # Pydantic or dataclass-like
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
            if hasattr(usage, "dict"):
                return usage.dict()
            if isinstance(usage, dict):
                return usage
        except Exception:
            pass
        # Fallback for dict-like response
        if isinstance(response, dict):
            maybe = response.get("usage")
            if isinstance(maybe, dict):
                return maybe
        return None

    @staticmethod
    def extract_responses_text_delta(event: Any) -> Optional[str]:
        try:
            event_type = getattr(event, "type", None)
            if isinstance(event_type, str) and event_type.endswith(".delta"):
                delta = getattr(event, "delta", None)
                if isinstance(delta, str):
                    return delta
        except Exception:
            pass
        return None

    @staticmethod
    def parse_completion(
        completion: Union[ChatCompletion, Stream[ChatCompletionChunk]],
    ) -> Dict[str, Any]:
        if isinstance(completion, Stream):
            # Process the stream of chunks
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
            # Combine all chunks into a single response
            if chunks:
                last_chunk = chunks[-1]
                combined_content = "".join(
                    [
                        choice.get("delta", {}).get("content", "")
                        for chunk in chunks
                        for choice in chunk.get("choices", [])
                    ]
                )
                return {
                    "id": last_chunk.get("id", ""),
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": choice.get("index", 0),
                            "message": {
                                "role": "assistant",
                                "content": combined_content,
                                "tool_calls": choice.get("delta", {}).get("tool_calls"),
                            },
                            "finish_reason": choice.get("finish_reason"),
                        }
                        for choice in last_chunk.get("choices", [])
                    ],
                }
            else:
                return {}
        else:
            # Handle regular ChatCompletion objects
            return {
                "id": completion.id,
                "created": completion.created,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": getattr(choice.message, "tool_calls", None),
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in completion.choices
                ],
                "usage": (
                    {
                        "prompt_tokens": (
                            completion.usage.prompt_tokens if completion.usage else 0
                        ),
                        "completion_tokens": (
                            completion.usage.completion_tokens
                            if completion.usage
                            else 0
                        ),
                        "total_tokens": (
                            completion.usage.total_tokens if completion.usage else 0
                        ),
                    }
                    if completion.usage
                    else {}
                ),
            }

    @staticmethod
    def parse_completion_from_chunks(
        chunks: List[ChatCompletionChunk],
    ) -> Dict[str, Any]:
        """Convert a list of ChatCompletionChunk objects into a combined response format."""
        if not chunks:
            return {}

        # Get the last chunk for metadata
        last_chunk = chunks[-1]

        # Combine all content from chunks
        combined_content = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in chunk.choices
            if choice.delta and choice.delta.content
        )

        # Combine all tool calls from chunks
        tool_calls = []
        current_tool_call = None

        for chunk in chunks:
            for choice in chunk.choices:
                if choice.delta and choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        if current_tool_call is None or (
                            tool_call.index != current_tool_call.index
                        ):
                            if current_tool_call:
                                tool_calls.append(current_tool_call)
                            current_tool_call = tool_call
                        else:
                            # Append to existing tool call
                            if tool_call.function is not None and current_tool_call.function is not None:
                                if tool_call.function.name:
                                    current_tool_call.function.name = (
                                        tool_call.function.name
                                    )
                                if tool_call.function.arguments:
                                    if current_tool_call.function.arguments:
                                        current_tool_call.function.arguments += (
                                            tool_call.function.arguments
                                        )
                                    else:
                                        current_tool_call.function.arguments = (
                                            tool_call.function.arguments
                                        )

        if current_tool_call:
            tool_calls.append(current_tool_call)

        # Construct the final response
        response = {
            "id": last_chunk.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(last_chunk, "model", None),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": combined_content if not tool_calls else None,
                        "tool_calls": tool_calls if tool_calls else None,
                    },
                    "finish_reason": (
                        last_chunk.choices[0].finish_reason
                        if last_chunk.choices
                        else None
                    ),
                }
            ],
            "usage": last_chunk.usage.model_dump() if last_chunk.usage else {},
        }

        return response
