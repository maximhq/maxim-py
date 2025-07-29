"""Utility functions for Fireworks AI API integration with Maxim.

This module provides utility functions for parsing and processing Fireworks AI API
requests and responses to integrate with Maxim's logging and monitoring system.
It handles message format conversion, parameter extraction, response standardization,
and image attachment processing for multimodal support.
"""

import time
import uuid
from typing import Any, Dict, List, Optional 
from collections.abc import Iterable
from fireworks.client.api import ChatCompletionResponseChoice, ChatMessage, ChatCompletionResponse
from fireworks.llm.LLM import ChatCompletionMessageParam
from ..logger import GenerationRequestMessage, Generation, Trace
from ..components.attachment import UrlAttachment
from ...scribe import scribe

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
    def parse_chunks_to_response(content: str, usage_data: Any) -> ChatCompletionResponse:
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
        return ChatCompletionResponse(
            id=f"streaming-response-{uuid.uuid4()}",
            created=int(time.time()),
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )],
            usage=usage_data,
            model="",
            object="chat.completion",
        )

    @staticmethod
    def parse_completion(completion: ChatCompletionResponse) -> Dict[str, Any]:
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
                                    ))
        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(f"[MaximSDK] Error adding image attachments: {e}")
            
    @staticmethod
    def apply_tags(
        generation: Optional[Generation], 
        trace: Optional[Trace], 
        generation_tags: Optional[Dict[str, str]], 
        trace_tags: Optional[Dict[str, str]]
        ) -> None:
        """
        Apply tags to the generation and trace objects.
        """
        # Apply tags if provided
        if generation_tags is not None and generation is not None:
            for key, value in generation_tags.items():
                generation.add_tag(key, value)
        if trace_tags is not None and trace is not None:
            for key, value in trace_tags.items():
                trace.add_tag(key, value)
    
    @staticmethod
    def map_fireworks_model_name(model: str) -> str:
        """Get the Fireworks model name from the model string.

        This method extracts the Fireworks model name from the model string.

        Returns:
            str: The mapped Fireworks model name.
        """
        
        model_name_mapping = {
            "alpha": "accounts/fireworks/models/alpha",
            "deepseek-r1": "accounts/fireworks/models/deepseek-r1",
            "deepseek-r1-basic": "accounts/fireworks/models/deepseek-r1-basic",
            "deepseek-r1-distill-llama-70b": "accounts/fireworks/models/deepseek-r1-distill-llama-70b",
            "deepseek-v3": "accounts/fireworks/models/deepseek-v3",
            "deepseek-v3-0324": "accounts/fireworks/models/deepseek-v3-0324",
            "firesearch-ocr-v6": "accounts/fireworks/models/firesearch-ocr-v6",
            "llama-guard-3-8b": "accounts/fireworks/models/llama-guard-3-8b",
            "llama-v3p1-405b-instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "llama-v3p1-405b-instruct-long": "accounts/fireworks/models/llama-v3p1-405b-instruct-long",
            "llama-v3p1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "llama-v3p1-8b-instruct": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "llama4-maverick-instruct-basic": "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "llama-v3p3-70b-instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "llama4-scout-instruct-basic": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "mixtral-8x22b-instruct": "accounts/fireworks/models/mixtral-8x22b-instruct",
            "moa": "accounts/fireworks/models/moa",
            "qwen2-vl-72b-instruct": "accounts/fireworks/models/qwen2-vl-72b-instruct",
            "qwen2p5-72b-instruct": "accounts/fireworks/models/qwen2p5-72b-instruct",
            "qwen2p5-vl-32b-instruct": "accounts/fireworks/models/qwen2p5-vl-32b-instruct",
            "qwen3-235b-a22b": "accounts/fireworks/models/qwen3-235b-a22b",
            "qwen3-30b-a3b": "accounts/fireworks/models/qwen3-30b-a3b",
            "qwq-32b": "accounts/fireworks/models/qwq-32b",
            "dobby-unhinged-llama-3-3-70b-new": "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new",
            "yi-large": "accounts/yi-01-ai/models/yi-large",
        }

        return model_name_mapping.get(model, model)
