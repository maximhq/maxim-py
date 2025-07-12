"""Utility functions for Fireworks AI API integration with Maxim.

This module provides utility functions for parsing and processing Fireworks AI API
requests and responses to integrate with Maxim's logging and monitoring system.
It handles message format conversion, parameter extraction, response standardization,
and image attachment processing for multimodal support.
"""

import time
from typing import Any, Dict, List 
from collections.abc import Iterable
from dataclasses import dataclass
from fireworks.llm.LLM import ChatCompletion, ChatCompletionMessageParam
from ..logger import GenerationRequestMessage, Generation
from ..components.attachment import UrlAttachment
from ...scribe import scribe

@dataclass
class ResponseMessage:
    """Message object for Fireworks AI response parsing.
    
    This dataclass represents a single message in a Fireworks AI response,
    containing the content and role information needed for logging and
    response handling.
    
    Attributes:
        content (str): The text content of the message response.
        role (str): The role of the message sender, defaults to "assistant".
    """
    content: str
    role: str = "assistant"


@dataclass
class ResponseChoice:
    """Choice object for Fireworks AI response parsing.
    
    This dataclass represents a single choice from a Fireworks AI response,
    containing the message, index, and finish reason information.
    
    Attributes:
        message (ResponseMessage): The message content and role information.
        index (int): The index of this choice in the response, defaults to 0.
        finish_reason (str): The reason why the response finished, defaults to "stop".
    """
    message: ResponseMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass(init=False)
class ResponseUsage:
    """Usage statistics for Fireworks AI response parsing.
    
    This dataclass tracks token usage and other metrics from Fireworks AI
    responses for logging and monitoring purposes. It handles cases where
    usage data might not be available from the API response.
    
    Attributes:
        prompt_tokens (int): Number of tokens used in the prompt.
        completion_tokens (int): Number of tokens used in the completion.
        total_tokens (int): Total number of tokens used.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def __init__(self, usage_data: Any) -> None:
        """Initialize usage statistics from CompletionUsage object.
        
        Args:
            usage_data (Any): Usage data from Fireworks AI API response.
                If None, all token counts will be set to 0.
        """
        if usage_data:
            self.prompt_tokens = getattr(usage_data, 'prompt_tokens', 0)
            self.completion_tokens = getattr(usage_data, 'completion_tokens', 0)
            self.total_tokens = getattr(usage_data, 'total_tokens', 0)
        else:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0


@dataclass(init=False)
class Response:
    """Response object for Fireworks AI response parsing.
    
    This dataclass represents a complete response from Fireworks AI,
    containing all necessary information for logging and processing.
    It's designed to be compatible with the parse_completion method.
    
    Attributes:
        id (str): Unique identifier for the response.
        created (int): Unix timestamp when the response was created.
        choices (List[ResponseChoice]): List of response choices.
        usage (ResponseUsage): Token usage statistics.
    """
    id: str
    created: int
    choices: List[ResponseChoice]
    usage: ResponseUsage
    
    def __init__(self, content: str, usage_data: Any) -> None:
        """Initialize response object from content and usage data.
        
        Args:
            content (str): The response content text.
            usage_data (Any): Usage statistics from the API.
        """
        self.id = "streaming-response"
        self.created = int(time.time())
        self.choices = [ResponseChoice(ResponseMessage(content))]
        self.usage = ResponseUsage(usage_data)


class FireworksUtils:
    """Utility class for Fireworks AI API integration with Maxim.

    This class provides static utility methods for parsing and processing
    Fireworks AI API requests and responses to integrate with Maxim's logging
    and monitoring system. It handles message format conversion, parameter
    extraction, response standardization, and image attachment processing.

    All methods are static and can be called directly on the class without
    instantiation. The class follows the same patterns as other provider
    integrations in the Maxim SDK.
    """

    @staticmethod
    def parse_message_param(messages: Iterable[ChatCompletionMessageParam]) -> List[GenerationRequestMessage]:
        """Parse Fireworks AI message parameters into Maxim format.

        This method converts Fireworks AI message dictionaries into Maxim's
        GenerationRequestMessage format for consistent logging and tracking.
        It handles various message formats including string content and
        structured content blocks with multimodal support.

        Args:
            messages (Iterable[ChatCompletionMessageParam]): Iterable of Fireworks AI
                message dictionaries to be parsed. Each message should have 'role'
                and 'content' keys following Fireworks AI's message format.

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

        This method extracts relevant model parameters from Fireworks AI API
        calls and formats them for consistent logging in Maxim. It handles
        common parameters like temperature, max_tokens, and Fireworks AI specific
        parameters while filtering out internal parameters.

        Args:
            **kwargs (Any): Keyword arguments that may contain model parameters
                from Fireworks AI API calls. Can include parameters like temperature,
                max_tokens, top_p, frequency_penalty, etc.

        Returns:
            Dict[str, Any]: Dictionary containing normalized model parameters
                with non-None values only. Internal parameters like 'messages',
                'model', and 'extra_headers' are excluded from the result.
        """
        model_params = {}
        skip_keys = ["messages", "model", "extra_headers", "tools"]
        
        # Common parameters that Fireworks AI supports
        param_keys = [
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "perf_metrics_in_response",
            "presence_penalty",
            "repetition_penalty",
            "top_k",
            "min_p",
            "response_format",
            "reasoning_effort",
            "stream",
            "n",
            "stop",
            "context_length_exceeded_behavior",
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
    def parse_chunks_to_response(content: str, usage_data: Any) -> Response:
        """Create a response object from streaming chunks for parsing.
        
        This method constructs a response object compatible with the parse_completion
        method from accumulated streaming content and usage data. It creates a
        structured response that mimics the Fireworks AI response format.
        
        Args:
            content (str): The accumulated content from streaming chunks that
                represents the complete response text.
            usage_data (Any): Usage information from the final chunk
                containing token counts and other usage metrics.
                
        Returns:
            Response: A structured response object that can be processed by
                the parse_completion method for consistent logging.
        """
        return Response(content, usage_data)
    
    @staticmethod
    def parse_completion(completion: ChatCompletion) -> Dict[str, Any]:
        """Parse a Fireworks AI completion response into standardized format.

        This method converts a Fireworks AI ChatCompletion response into a
        standardized dictionary format suitable for Maxim logging. It handles
        both structured completion objects and dictionary responses, extracting
        relevant information like choices, usage data, and metadata.

        Args:
            completion (ChatCompletion): The completion response from Fireworks AI
                to be parsed. Can be either a structured ChatCompletion object
                or a dictionary-like response.

        Returns:
            Dict[str, Any]: A standardized dictionary containing parsed response
                data with consistent structure for logging. Includes choices,
                usage information, and metadata when available.
        """
        if hasattr(completion, 'choices') and hasattr(completion, 'id'):
            # Handle structured ChatCompletion objects
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

        # Fallback for dict-like responses
        if isinstance(completion, dict):
            return completion
    
        return {}
        
    @staticmethod
    def add_image_attachments_from_messages(generation: Generation, messages: Iterable[ChatCompletionMessageParam]) -> None:
        """Extract image URLs from messages and add them as attachments to the generation.

        This method scans through Fireworks AI messages to find image URLs in content
        blocks and automatically adds them as URL attachments to the generation object.
        It handles the multimodal message format where images are embedded within
        content arrays, enabling proper tracking and logging of image inputs.

        Args:
            generation (Generation): The Maxim generation object to add attachments to.
                If None, the method will return early without processing.
            messages (Iterable[ChatCompletionMessageParam]): The messages to scan for
                image URLs. Should follow Fireworks AI's message format with content
                arrays containing image_url objects.

        Returns:
            None: This method modifies the generation object in-place by adding
                attachments and does not return any value.

        Note:
            This method is designed to handle Fireworks AI's multimodal message format
            where images are specified as content blocks with type "image_url". It
            gracefully handles cases where no images are present or where the message
            format doesn't contain image data.
        """
        if generation is None or not messages:
            return
            
        try:
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        # Process content blocks for multimodal messages
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                                image_url_data = content_item.get("image_url", {})
                                image_url = image_url_data.get("url", "")
                                if image_url:
                                    # Add the image URL as an attachment to the generation
                                    generation.add_attachment(UrlAttachment(
                                        url=image_url,
                                        name="User Image",
                                        mime_type="image",
                                    ))
        except Exception as e:
            scribe().warning(f"[MaximSDK] Error adding image attachments: {e}")