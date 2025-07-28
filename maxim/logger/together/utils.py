"""Utility functions for Together AI API integration with Maxim.

This module provides utility functions for parsing and processing Together AI API
requests and responses to integrate with Maxim's logging and monitoring system.
It handles message format conversion, parameter extraction, response standardization,
and image attachment processing.
"""

import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from together.types.common import FinishReason, ObjectType 
from together.types.chat_completions import ChatCompletionResponse, MessageRole, UsageData, ChatCompletionChoicesData, ChatCompletionMessage

from ..logger import GenerationRequestMessage, Generation
from ..components.attachment import UrlAttachment
from ...scribe import scribe


class TogetherUtils:
    """Utility class for Together AI API integration with Maxim.

    This class provides static utility methods for parsing and processing
    Together AI API requests and responses to integrate with Maxim's logging
    and monitoring system. It handles message format conversion, parameter
    extraction, response standardization, and image attachment processing.

    All methods are static and can be called directly on the class without
    instantiation. The class follows the same patterns as other provider
    integrations in the Maxim SDK.
    """

    @staticmethod
    def parse_message_param(
        messages: Iterable[Dict[str, Any]],
        override_role: Optional[str] = None,
    ) -> List[GenerationRequestMessage]:
        """Parse Together AI message parameters into Maxim format.

        This method converts Together AI message dictionaries into Maxim's
        GenerationRequestMessage format for consistent logging and tracking.
        It handles various message formats including string content and
        structured content blocks with multimodal support.

        Args:
            messages (Iterable[Dict[str, Any]]): Iterable of Together AI message
                dictionaries to be parsed. Each message should have 'role' and
                'content' keys following Together AI's message format.
            override_role (Optional[str]): Optional role to override the message role.
                If provided, all messages will use this role instead of their
                original role. Defaults to None.

        Returns:
            List[GenerationRequestMessage]: List of parsed messages in Maxim format,
                with role and content extracted and standardized.
        """
        parsed_messages: List[GenerationRequestMessage] = []

        for msg in messages:
            role = override_role or msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle content blocks for multimodal messages
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=text_content)
                )
            else:
                # Handle simple string content
                parsed_messages.append(
                    GenerationRequestMessage(role=role, content=str(content))
                )

        return parsed_messages

    @staticmethod
    def get_model_params(**kwargs: Any) -> Dict[str, Any]:
        """Extract and normalize model parameters for Maxim logging.

        This method extracts relevant model parameters from Together AI API
        calls and formats them for consistent logging in Maxim. It handles
        common parameters like temperature, max_tokens, and Together AI specific
        parameters while filtering out internal parameters.

        Args:
            **kwargs (Any): Keyword arguments that may contain model parameters
                from Together AI API calls. Can include parameters like temperature,
                max_tokens, top_p, frequency_penalty, etc.

        Returns:
            Dict[str, Any]: Dictionary containing normalized model parameters
                with non-None values only. Internal parameters like 'messages'
                and 'model' are excluded from the result.
        """
        model_params = {}
        skip_keys = ["messages", "model"]

        # Common parameters that Together AI supports
        param_keys = [
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "top_k",
            "min_p",
            "safety_model",
            "response_format",
            "tools",
            "tool_choice",
            "stream",
            "logprobs",
            "echo",
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
    def parse_chunks_to_response(
        content: str, usage_data: Optional[UsageData] = None
    ) -> ChatCompletionResponse:
        """Create a response object from streaming chunks for parsing.
        
        This method constructs a response object compatible with the parse_completion
        method from accumulated streaming content and usage data. It creates a
        structured response that mimics the Together AI response format.
        
        Args:
            content (str): The accumulated content from streaming chunks that
                represents the complete response text.
            usage_data (UsageData): Usage information from the final chunk
                containing token counts and other usage metrics.
            
        Returns:
            Response: Response object with proper attributes that can be processed
                by parse_completion() method. Contains choices, usage, and
                standard response metadata.
        """
       
        return ChatCompletionResponse(
            id=f"streaming-response-{uuid.uuid4()}",
            created=int(time.time()),
            choices=[ChatCompletionChoicesData(
                index=0,
                message=ChatCompletionMessage(role=MessageRole.ASSISTANT, content=content),
                finish_reason=FinishReason.StopSequence
            )],
            usage=usage_data,
            object=ObjectType.ChatCompletion,
        )

    @staticmethod
    def parse_completion(completion: ChatCompletionResponse) -> Dict[str, Any]:
        """Parse Together AI completion response into standardized format.

        This method converts a Together AI ChatCompletionResponse object into
        a standardized dictionary format that can be consistently processed
        by Maxim's logging system. It handles various response formats and
        provides fallback parsing for edge cases.

        Args:
            completion (ChatCompletionResponse): The completion response object
                from Together AI API containing the generated content, choices,
                usage information, and other response metadata.

        Returns:
            Dict[str, Any]: Standardized response dictionary with the following structure:
                - id: Unique identifier for the response
                - created: Unix timestamp of creation
                - choices: List of choices with message content and metadata
                - usage: Token usage statistics (prompt, completion, total)
        """
        # Handle Together ChatCompletionResponse format
        if hasattr(completion, 'choices') and hasattr(completion, 'id'):
            parsed_response = {
                "id": completion.id,
                "created": getattr(completion, 'created', int(time.time())),
                "choices": [],
            }

            for choice in completion.choices or []:
                choice_data = {
                    "index": getattr(choice, 'index', 0),
                    "message": {
                        "role": getattr(choice.message, 'role', 'assistant'),
                        "content": getattr(choice.message, 'content', ''),
                    },
                    "finish_reason": getattr(choice, 'finish_reason', None),
                }

                # Add tool calls if present
                if (
                    choice.message is not None
                    and hasattr(choice.message, "tool_calls")
                    and choice.message.tool_calls
                ):
                    choice_data["message"]["tool_calls"] = choice.message.tool_calls

                parsed_response["choices"].append(choice_data)

            # Add usage information if available
            if (
                hasattr(completion, "usage")
                and completion.usage is not None
                and completion.usage.prompt_tokens is not None
                and completion.usage.completion_tokens is not None
                and completion.usage.total_tokens is not None
            ):
                parsed_response["usage"] = {
                    "prompt_tokens": getattr(completion.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(completion.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(completion.usage, 'total_tokens', 0),
                }

            return parsed_response

        # Fallback for dict-like responses
        elif isinstance(completion, dict):
            return completion

        return {}

    @staticmethod
    def add_image_attachments_from_messages(generation: Generation, messages: Iterable[Dict[str, Any]]) -> None:
        """Extract image URLs from messages and add them as attachments to the generation.

        This method scans through Together AI messages to find image URLs in content
        blocks and automatically adds them as URL attachments to the generation object.
        It handles the multimodal message format where images are embedded within
        content arrays.

        Args:
            generation (Generation): The Maxim generation object to add attachments to.
                If None, the method will return early without processing.
            messages (Iterable[Dict[str, Any]]): The messages to scan for image URLs.
                Should follow Together AI's message format with content arrays
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
                                        mime_type="image"
                                    ))
        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(f"[MaximSDK] Error adding image attachments: {e}")
