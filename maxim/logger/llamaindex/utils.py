from typing import List, Union, Optional, Dict, Any
from llama_index.core.base.llms.types import ContentBlock
from llama_index.core.llms import ChatMessage, MessageRole, LLM

from ..logger import GenerationRequestMessage
from ..components.generation import GenerationRequestTextMessageContent, GenerationRequestImageMessageContent, GenerationResultMessage, TextContent, ImageContent, AudioContent
from ...scribe import scribe

class LlamaIndexUtils:

    @staticmethod
    def parse_messages_to_generation_request(
        messages: List[ChatMessage]
        ) -> List[GenerationRequestMessage]:

        parsed_messages: List[GenerationRequestMessage] = []

        for message in messages:
            parsed_message = LlamaIndexUtils.parse_individual_chat_message(message)
            if parsed_message:
                parsed_messages.append(parsed_message)

        return parsed_messages

    @staticmethod
    def parse_individual_chat_message(message: ChatMessage) -> Optional[GenerationRequestMessage]:
        if message.role == MessageRole.USER:
            return GenerationRequestMessage(role="user", content=parse_message_content(message.blocks))
        elif message.role == MessageRole.ASSISTANT:
            return GenerationRequestMessage(role="assistant", content=parse_message_content(message.blocks))
        elif message.role == MessageRole.SYSTEM:
            return GenerationRequestMessage(role="system", content=parse_message_content(message.blocks))
        elif message.role == MessageRole.TOOL:
            return GenerationRequestMessage(role="tool", content=parse_message_content(message.blocks))
        else:
            scribe().warning(f"Unsupported message role: {message.role}")
            # TODO: Remove this once done and releasing
            raise ValueError(f"Unsupported message role: {message.role}")
            
    @staticmethod
    def parse_message_to_generation_result(
        message: ChatMessage
        ) -> GenerationResultMessage:
        
        parsed_messages = GenerationResultMessage(role=message.role, content=parse_message_content_to_generation_result(message.blocks), tool_calls=[])
        
        return parsed_messages

    @staticmethod
    def parse_model_parameters(llm: LLM) -> Dict[str, Any]:
        # TODO: have a model_parameters empty dict and then fill as we find params
        # TODO: add additional params as we find them
        model_parameters: Dict[str, Any] = {}
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

        return model_parameters

def parse_message_content(blocks: List[ContentBlock]):
    text_content = []
    for block in blocks:
        if block.block_type == "text":
            text_content.append(block.text)
        elif block.block_type == "image":
            if block.url:
                if isinstance(block.url, str):
                    image_url = block.url
                else:
                    image_url = str(block.url)
                text_content.append(f"[Image: {image_url}]")
            else:
                raise ValueError(f"Image block has no URL: {block}")
        else:
            raise ValueError(f"Unsupported block type: {block.block_type}")
    return " ".join(text_content)

def parse_message_content_to_generation_result(blocks: List[ContentBlock]):
    content: List[Union[TextContent, ImageContent, AudioContent]] = []
    for block in blocks:
        if block.block_type == "text":
            content.append(GenerationRequestTextMessageContent(type="text", text=block.text))
        elif block.block_type == "image":
            if block.url:
                if isinstance(block.url, str):
                    image_url = block.url
                else:
                    image_url = str(block.url)
                content.append(GenerationRequestImageMessageContent(type="image", image_url=image_url))
            else:
                raise ValueError(f"Image block has no URL: {block}")
        else:
            raise ValueError(f"Unsupported block type: {block.block_type}")
    return content