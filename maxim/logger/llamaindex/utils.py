"""
This module contains utility functions for parsing LlamaIndex messages and model parameters.
"""

from typing import Optional, Any
from llama_index.core.base.llms.types import ContentBlock
from llama_index.core.llms import ChatMessage, MessageRole, LLM

from ..logger import GenerationRequestMessage
from ...scribe import scribe

class LlamaIndexUtils:
    """
    Utility functions for parsing LlamaIndex messages and model parameters.
    """

    @staticmethod
    def parse_messages_to_generation_request(
        messages: list[ChatMessage]
    ) -> list[GenerationRequestMessage]:
        """
        Parse a list of LlamaIndex chat messages to a list of GenerationRequestMessage objects.
        """
        parsed_messages: list[GenerationRequestMessage] = []

        for message in messages:
            parsed_message = LlamaIndexUtils.parse_individual_chat_message(message)
            if parsed_message:
                parsed_messages.append(parsed_message)

        return parsed_messages

    @staticmethod
    def parse_individual_chat_message(message: ChatMessage) -> Optional[GenerationRequestMessage]:
        """
        Parse an individual LlamaIndex chat message to a GenerationRequestMessage object.
        """
        if message.role == MessageRole.USER:
            return GenerationRequestMessage(
                role="user",
                content=parse_message_content(message.blocks)
            )

        if message.role == MessageRole.ASSISTANT:
            return GenerationRequestMessage(
                role="assistant",
                content=parse_message_content(message.blocks)
            )

        if message.role == MessageRole.SYSTEM:
            return GenerationRequestMessage(
                role="system",
                content=parse_message_content(message.blocks)
            )

        if message.role == MessageRole.TOOL:
            return GenerationRequestMessage(
                role="tool",
                content=parse_message_content(message.blocks)
            )

        scribe().warning(f"Unsupported message role: {message.role}")
        return None

    @staticmethod
    def parse_model_parameters(llm: LLM) -> dict[str, Any]:
        """
        Parse the model parameters for a LlamaIndex LLM.
        """
        model_parameters: dict[str, Any] = {}
        if hasattr(llm, "temperature"):
            model_parameters["temperature"] = getattr(llm, "temperature", None)
        if hasattr(llm, "max_tokens"):
            model_parameters["max_tokens"] = getattr(llm, "max_tokens", None)
        if hasattr(llm, "top_p"):
            model_parameters["top_p"] = getattr(llm, "top_p", None)
        if hasattr(llm, "frequency_penalty"):
            model_parameters["frequency_penalty"] = getattr(llm, "frequency_penalty", None)
        if hasattr(llm, "presence_penalty"):
            model_parameters["presence_penalty"] = getattr(llm, "presence_penalty", None)
        if hasattr(llm, "stop"):
            model_parameters["stop"] = getattr(llm, "stop", None)

        return {k: v for k, v in model_parameters.items() if v is not None}

def parse_message_content(blocks: list[ContentBlock]) -> str:
    """
    Parse the content of a list of LlamaIndex content blocks to a string.
    """
    text_content = []
    for block in blocks:
        if block.block_type == "text":
            text_content.append(block.text)
        elif block.block_type == "image":
            if block.url:
                image_url = block.url if isinstance(block.url, str) else str(block.url)
                text_content.append(f"[Image: {image_url}]")
            else:
                raise ValueError(f"Image block has no URL: {block}")
        else:
            raise ValueError(f"Unsupported block type: {block.block_type}")
    return " ".join(text_content)
