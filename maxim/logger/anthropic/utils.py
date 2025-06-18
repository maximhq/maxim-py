import time
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4
from anthropic import MessageStreamEvent
from anthropic.types import MessageParam, TextDelta

from ..logger import GenerationRequestMessage


class AnthropicUtils:
    """Utility class for Anthropic API integration with Maxim.

    This class provides static utility methods for parsing and processing
    Anthropic API requests and responses to integrate with Maxim's logging
    and monitoring system. It handles message format conversion, parameter
    extraction, and response standardization.

    All methods are static and can be called directly on the class without
    instantiation.
    """

    @staticmethod
    def parse_message_param(
        message: Iterable[MessageParam],
        override_role: Optional[str] = None,
    ) -> List[GenerationRequestMessage]:
        """Parse Anthropic message parameters into Maxim format.

        This method converts Anthropic MessageParam objects into Maxim's
        GenerationRequestMessage format for consistent logging and tracking.
        It handles various message formats including string content and
        structured content blocks.

        Args:
            message (Iterable[MessageParam]): Iterable of Anthropic message parameters
                to be parsed. Can contain strings, dicts, or MessageParam objects.
            override_role (Optional[str]): Optional role to override the message role.
                If provided, all messages will use this role instead of their original role.

        Returns:
            List[GenerationRequestMessage]: List of parsed messages in Maxim format,
                with role and content extracted and standardized.

        Note:
            - String messages are treated as user messages by default
            - Dict messages should have 'role' and 'content' keys
            - Content blocks are flattened into text content
            - Complex content structures are converted to string representation
        """
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
        """Extract and normalize model parameters for Maxim logging.

        This method extracts relevant model parameters from Anthropic API
        calls and formats them for consistent logging in Maxim. It handles
        common parameters like temperature, top_p, and other generation settings.

        Args:
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs (Any): Additional keyword arguments that may contain
                model parameters such as temperature, top_p, top_k, system, etc.

        Returns:
            Dict[str, Any]: Dictionary containing normalized model parameters
                with non-None values only. Common parameters are extracted
                explicitly while additional parameters are included as-is.

        Note:
            - Only non-None parameters are included in the result
            - System, metadata, temperature, top_p, and top_k are handled explicitly
            - Additional parameters from kwargs are included if they have values
        """
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
        """Parse a list of Anthropic stream events into standardized format.

        This method processes a complete stream of MessageStreamEvent objects
        and converts them into a standardized response format compatible with
        OpenAI-style responses for consistent logging and processing.

        Args:
            stream (List[MessageStreamEvent]): List of stream events from
                Anthropic's streaming API response.

        Returns:
            Dict[str, Any]: Standardized response dictionary with the following structure:
                - id: Unique identifier for the response
                - created: Unix timestamp of creation
                - choices: List with single choice containing message and finish_reason
                - usage: Token usage statistics (prompt, completion, total)

        Raises:
            ValueError: If the stream list is empty.

        Note:
            - Text content is extracted from content_block_delta events
            - Token usage is extracted from message_start events
            - The response format mimics OpenAI's API for compatibility
            - Finish reason is set to "stop" by default (Anthropic doesn't provide this directly)
        """
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
        """Parse an Anthropic Message response into standardized format.

        This method converts an Anthropic Message object into a standardized
        response format compatible with OpenAI-style responses for consistent
        logging and processing across different AI providers.

        Args:
            message (Any): Anthropic Message object containing the API response
                with content, usage statistics, and metadata.

        Returns:
            Dict[str, Any]: Standardized response dictionary with the following structure:
                - id: Message ID from Anthropic
                - created: Unix timestamp of parsing time
                - choices: List with single choice containing message and finish_reason
                - usage: Token usage statistics (input, output, total tokens)

        Note:
            - Content blocks are flattened into a single text string
            - Both structured content blocks and dict-based content are supported
            - Token usage is extracted from the message's usage attribute
            - Stop reason is mapped from Anthropic's stop_reason or defaults to "stop"
            - The response format mimics OpenAI's API for cross-provider compatibility
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
