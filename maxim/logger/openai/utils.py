import time
from typing import Any, Dict, Iterable, List, Optional, Union

from openai._streaming import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
                    GenerationRequestMessage(
                        role=role,
                        content=text_content
                    )
                )
            else:
                parsed_messages.append(
                    GenerationRequestMessage(
                        role=role,
                        content=str(content)
                    )
                )
        
        return parsed_messages

    @staticmethod
    def get_model_params(
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_params = {}

        max_tokens = kwargs.get("max_tokens", None)
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens
            
        param_keys = ["temperature", "top_p", "presence_penalty", "frequency_penalty", "response_format"]
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None:
                model_params[key] = kwargs[key]
                
        for key, value in kwargs.items():
            if key not in param_keys and value is not None:
                model_params[key] = value
                
        return model_params

    @staticmethod
    def parse_stream_response(
        chunk: ChatCompletionChunk,
    ) -> Dict[str, Any]:
        return {
            "id": chunk.id,
            "created": int(time.time()),
            "choices": [{
                "index": choice.index,
                "delta": {
                    "role": "assistant",
                    "content": choice.delta.content or "",
                },
                "finish_reason": choice.finish_reason
            } for choice in chunk.choices],
        }

    @staticmethod
    def parse_completion(
        completion: Union[ChatCompletion, Stream[ChatCompletionChunk]],
    ) -> Dict[str, Any]:
        if isinstance(completion, Stream):
            # Process the stream of chunks
            chunks = []
            for chunk in completion:
                chunks.append(OpenAIUtils.parse_stream_response(chunk))
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
                            },
                            "finish_reason": choice.get("finish_reason"),
                        }
                        for choice in last_chunk.get("choices", [])
                    ],
                }
            else:
                return {}

        return {
            "id": completion.id,
            "created": int(time.time()),
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": "assistant",
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in completion.choices
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens
                if completion.usage
                else 0,
                "completion_tokens": completion.usage.completion_tokens
                if completion.usage
                else 0,
                "total_tokens": completion.usage.total_tokens
                if completion.usage
                else 0,
            },
        }