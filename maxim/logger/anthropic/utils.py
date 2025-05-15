import time
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from anthropic.types import MessageParam, MessageStreamEvent, TextDelta

from ..logger import GenerationRequestMessage


class AnthropicUtils:
    @staticmethod
    def parse_message_param(
        message: Iterable[MessageParam],
        override_role: Optional[str] = None,
    ) -> List[GenerationRequestMessage]:
        messages: List[GenerationRequestMessage] = []
        
        for msg in message:
            if isinstance(msg, str):
                messages.append(
                    GenerationRequestMessage(
                        role=override_role or "user",
                        content=msg
                    )
                )
            elif isinstance(msg, dict):
                role = override_role or msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle content blocks
                    text_content = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                    messages.append(
                        GenerationRequestMessage(
                            role=role,
                            content=text_content
                        )
                    )
                else:
                    messages.append(
                        GenerationRequestMessage(
                            role=role,
                            content=str(content)
                        )
                    )
        
        return messages

    @staticmethod
    def get_model_params(
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        model_params = {}
        
        # Check max_tokens parameter
        if max_tokens is not None:
            model_params["max_tokens"] = max_tokens
            
        # Common parameters to check from kwargs
        param_keys = ["system", "metadata", "temperature", "top_p", "top_k"]
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None:
                model_params[key] = kwargs[key]
            
        # Add any additional parameters from kwargs
        for key, value in kwargs.items():
            if key not in param_keys and value is not None:
                model_params[key] = value
                
        return model_params

    @staticmethod
    def parse_message_stream(
        stream: List[MessageStreamEvent],
    ) -> Dict[str, Any]:
        if not stream:
            raise ValueError("No response chunks")
            
        text = ""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        for event in stream:
            if hasattr(event, "type"):
                if event.type == "message_start":
                    usage["prompt_tokens"] = event.message.usage.input_tokens
                    usage["completion_tokens"] = event.message.usage.output_tokens
                    usage["total_tokens"] = event.message.usage.input_tokens + event.message.usage.output_tokens
                elif event.type == "content_block_delta":
                    if isinstance(event.delta, TextDelta):
                        text += event.delta.text
                
        return {
            "id": str(uuid4()),
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop"  # Anthropic doesn't provide this directly
            }],
            "usage": usage
        }

    @staticmethod
    def parse_message(
        message: Any,
    ) -> Dict[str, Any]:
        """
        Parse the Message response from Anthropic API into a standardized format.
        """
        content = ""
        if isinstance(message.content, list):
            for block in message.content:
                # Check if block is a dict and has the expected structure
                if hasattr(block, "type") and hasattr(block, "text") and block.type == "text":
                        content += block.text
                elif isinstance(block, dict) and block.get("type") == "text":
                    content += block.get("text", "")
                elif isinstance(block, dict) and block.get("type") == "message":
                    content += block.get("text", "")
        else:
            content = str(message.content)
            
        return {
            "id": message.id,
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": message.stop_reason or "stop"
            }],
            "usage": {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens
            }
        }