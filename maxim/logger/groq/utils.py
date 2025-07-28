"""Utility functions for Groq API integration with Maxim.

This module provides utility functions for parsing and processing Groq API
requests and responses to integrate with Maxim's logging and monitoring system.
It handles message format conversion, parameter extraction, response standardization,
and image attachment processing.
"""

import time
from typing import Any, Dict, List, Optional, Union
from collections.abc import Iterable
import uuid

from groq.types.chat.chat_completion import Choice
from groq.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
)
from groq.types import CompletionUsage
from groq.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from groq.types.chat.chat_completion_message_tool_call import Function

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
    def parse_message(
        message: ChatCompletionMessage,
    ) -> ChatCompletionMessageParam:
        """Parse Groq message into Maxim format."""
        return ChatCompletionMessageParam(role=message.role, content=message.content)

    @staticmethod
    def parse_message_param(
        messages: Iterable[Union[ChatCompletionMessageParam, ChatCompletionMessage]],
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
            if isinstance(msg, ChatCompletionMessage):
                role = msg.role
                content = msg.content
            else:
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
        skip_keys = ["messages", "model", "extra_headers"]

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
    def parse_tool_calls(
        tool_calls: Optional[List[List[ChoiceDeltaToolCall]]],
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Parse and flatten tool calls from Groq streaming chunks.

        This method processes tool calls received from Groq's streaming API chunks,
        flattening the nested structure and converting them into a standardized
        ChatCompletionMessageToolCall format. It handles potential None values
        and empty lists in the input structure.

        Args:
            tool_calls (Optional[List[List[ChoiceDeltaToolCall]]]): A nested list of
                tool calls from streaming chunks. The outer list represents different
                chunks, while the inner list contains the tool calls for each chunk.
                Can be None if no tool calls are present.

        Returns:
            Optional[List[ChatCompletionMessageToolCall]]: A flattened list of
                standardized tool calls, each containing an id, type, and function
                with name and arguments. Returns None if no valid tool calls are found
                or if the input is None.
        """
        if not tool_calls:
            return None

        # Flatten the list and filter out None entries
        flattened_tool_calls = [
            tool_call
            for sublist in tool_calls
            if sublist is not None
            for tool_call in sublist
            if tool_call is not None
        ]

        if not flattened_tool_calls:
            return None

        return [
            ChatCompletionMessageToolCall(
                id=tool_call.id,
                type=tool_call.type,
                function=Function(  
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )
            for tool_call in flattened_tool_calls
        ]

    @staticmethod
    def parse_chunks_to_response(
        content: str, 
        usage_data: Optional[CompletionUsage], 
        tool_calls: Optional[List[List[ChoiceDeltaToolCall]]],
    ) -> ChatCompletion:
        """Create a response object from streaming chunks for parsing.
        
        This method constructs a response object compatible with the parse_completion
        method from accumulated streaming content and usage data. It creates a
        structured response that mimics the Groq response format.
        
        Args:
            content (str): The accumulated content from streaming chunks that
                represents the complete response text.
            usage_data (Optional[CompletionUsage]): Usage information from the final chunk
                containing token counts and other usage metrics.
            tool_calls (Optional[List[ChoiceDeltaToolCall]]): List of tool calls from the chunks.
            
        Returns:
            Response: Response object with proper attributes that can be processed
                by parse_completion() method. Contains choices, usage, and
                standard response metadata.
        """
        converted_tool_calls = GroqUtils.parse_tool_calls(tool_calls)

        return ChatCompletion(
            id=f"streaming-response-{uuid.uuid4()}",
            created=int(time.time()),
            choices=[Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=converted_tool_calls,
                ),
                finish_reason="stop",
            )],
            usage=usage_data,
            model="",
            object="chat.completion",
        )

    @staticmethod
    def parse_completion(completion: ChatCompletion) -> Dict[str, Any]:
        """Parse Groq completion response into standardized format.

        This method converts a Groq ChatCompletion object into
        a standardized dictionary format that can be consistently processed
        by Maxim's logging system. It handles various response formats and
        provides fallback parsing for edge cases.

        Args:
            completion (ChatCompletion): The completion response object
                from Groq API containing the generated content, choices,
                usage information, and other response metadata.

        Returns:
            Dict[str, Any]: Standardized response dictionary with the following structure:
                - id: Unique identifier for the response
                - created: Unix timestamp of creation
                - choices: List of choices with message content and metadata
                - usage: Token usage statistics (prompt, completion, total)
        """
        # Handle Groq ChatCompletion format
        if hasattr(completion, 'choices') and hasattr(completion, 'id'):
            parsed_response = {
                "id": completion.id,
                "created": getattr(completion, 'created', int(time.time())),
                "choices": [],
            }

            for choice in completion.choices:
                choice_data = {
                    "index": getattr(choice, 'index', 0),
                    "message": {
                        "role": getattr(choice.message, 'role', 'assistant'),
                        "content": getattr(choice.message, 'content', ''),
                    },
                    "finish_reason": getattr(choice, 'finish_reason', "stop"),
                }

                # Add tool calls if present
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    choice_data["message"]["tool_calls"] = choice.message.tool_calls

                parsed_response["choices"].append(choice_data)

            # Add usage information if available
            if hasattr(completion, 'usage') and completion.usage:
                parsed_response["usage"] = {
                    "prompt_tokens": getattr(completion.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(completion.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(completion.usage, 'total_tokens', 0),
                }

            return parsed_response

        if isinstance(completion, dict):
            return completion

        return {}

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
            generation.error({ "message": str(e) })
            scribe().warning(f"[MaximSDK] Error adding image attachments: {e}")
