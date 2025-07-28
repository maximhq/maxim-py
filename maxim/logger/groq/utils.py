"""Utility functions for Groq API integration with Maxim.

This module provides utility functions for parsing and processing Groq API
requests and responses to integrate with Maxim's logging and monitoring system.
It handles message format conversion, parameter extraction, response standardization,
and image attachment processing.
"""

import time
from typing import Any, Dict, List, Optional
from collections.abc import Iterable
import uuid

from groq.types.chat.chat_completion import Choice
from groq.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from groq.types import CompletionUsage

from maxim.logger.components.tool_call import ToolCallConfigDict

from ..logger import GenerationRequestMessage, Generation
from ..components.attachment import UrlAttachment
from ...scribe import scribe


class GroqUtils:
    """Utility class for Groq API integration with Maxim.

    This class provides static utility methods for parsing and processing
    Groq API requests and responses to integrate with Maxim's logging
    and monitoring system. It handles message format conversion, parameter
    extraction, response standardization, and image attachment processing.

    All methods are static and can be called directly on the class without
    instantiation. The class follows the same patterns as other provider
    integrations in the Maxim SDK.
    """

    @staticmethod
    def parse_message_param(
        messages: Iterable[ChatCompletionMessageParam],
    ) -> List[GenerationRequestMessage]:
        """Parse Groq message parameters into Maxim format.

        This method converts Groq message parameters into Maxim's
        GenerationRequestMessage format for consistent logging and tracking.
        It handles various message formats including string content and
        structured content blocks with multimodal support.

        Args:
            messages (Iterable[ChatCompletionMessageParam]): Iterable of Groq message
                parameters to be parsed. Each message should have 'role' and
                'content' keys following Groq's message format.

        Returns:
            List[GenerationRequestMessage]: List of parsed messages in Maxim format,
                with role and content extracted and standardized.
        """
        parsed_messages: List[GenerationRequestMessage] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle content blocks for multimodal messages
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=text_content),
                )
            else:
                # Handle simple string content
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=str(content)),
                )

        return parsed_messages

    @staticmethod
    def get_model_params(**kwargs: Any) -> Dict[str, Any]:
        """Extract and normalize model parameters for Maxim logging.

        This method extracts relevant model parameters from Groq API
        calls and formats them for consistent logging in Maxim. It handles
        common parameters like temperature, max_tokens, and Groq specific
        parameters while filtering out internal parameters.

        Args:
            **kwargs (Any): Keyword arguments that may contain model parameters
                from Groq API calls. Can include parameters like temperature,
                max_tokens, top_p, frequency_penalty, etc.

        Returns:
            Dict[str, Any]: Dictionary containing normalized model parameters
                with non-None values only. Internal parameters like 'messages'
                and 'model' are excluded from the result.
        """
        model_params = {}
        skip_keys = ["messages", "model", "extra_headers", "tools"]

        # Common parameters that Groq supports
        param_keys = [
            "temperature",
            "top_p",
            "max_tokens", # deprecated
            "max_completion_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "top_k",
            "min_p",
            "response_format",
            "parallel_tool_calls",
            "tool_choice",
            "stream",
            "n",
            "stop",
        ]

        # Add explicitly known parameters
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None and key not in skip_keys:
                model_params[key] = kwargs[key]

        # Add any other parameters that aren't in skip_keys
        for key, value in kwargs.items():
            if key not in param_keys and key not in skip_keys and value is not None:
                model_params[key] = value

        return model_params

    @staticmethod
    def parse_chunks_to_response(content: str, usage_data: Optional[CompletionUsage]) -> ChatCompletion:
        """Create a response object from streaming chunks for parsing.
        
        This method constructs a response object compatible with the parse_completion
        method from accumulated streaming content and usage data. It creates a
        structured response that mimics the Groq response format.
        
        Args:
            content (str): The accumulated content from streaming chunks that
                represents the complete response text.
            usage_data (Optional[CompletionUsage]): Usage information from the final chunk
                containing token counts and other usage metrics.
            
        Returns:
            Response: Response object with proper attributes that can be processed
                by parse_completion() method. Contains choices, usage, and
                standard response metadata.
        """

        return ChatCompletion(
            id=f"streaming-response-{uuid.uuid4()}",
            created=int(time.time()),
            choices=[Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop"
            )],
            usage=usage_data,
            model="",
            object="chat.completion",
        )

    @staticmethod
    def parse_tool_calls(message: ChatCompletionMessage) -> Optional[List[ToolCallConfigDict]]:
        """Parse tool calls from a Groq message."""

        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return None

        tool_calls: List[ToolCallConfigDict] = []
        for tool_call in message.tool_calls:
            tool_info = ToolCallConfigDict(
                id=tool_call.id,
                name=tool_call.function.name,
                args=tool_call.function.arguments,
            )
            tool_calls.append(tool_info)

        return tool_calls

    @staticmethod
    def parse_completion(completion: ChatCompletion) -> Dict[str, Any]:
        """Parse Groq completion response into standardized format."""

        # Handle Groq ChatCompletion format
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
    def add_image_attachments_from_messages(
            generation: Generation, 
            messages: Iterable[ChatCompletionMessageParam]
        ) -> None:
        """Extract image URLs from messages and add them as attachments to the generation.

        This method scans through Groq messages to find image URLs in content
        blocks and automatically adds them as URL attachments to the generation object.
        It handles the multimodal message format where images are embedded within
        content arrays.

        Args:
            generation (Generation): The Maxim generation object to add attachments to.
                If None, the method will return early without processing.
            messages (Iterable[ChatCompletionMessageParam]): The messages to scan for image URLs.
                Should follow Groq's message format with content arrays
                containing image_url objects.

        Returns:
            None: This method modifies the generation object in-place by adding
                attachments and does not return any value.
        """
        if generation is None or not messages:
            return

        try:
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                                image_url_data = content_item.get("image_url", {})
                                image_url = image_url_data.get("url", "")
                                if image_url:
                                    generation.add_attachment(UrlAttachment(
                                        url=image_url,
                                        name="User Image",
                                        mime_type="image",
                                    ))
        except Exception as e:
            scribe().warning(f"[MaximSDK] Error adding image attachments: {e}")
