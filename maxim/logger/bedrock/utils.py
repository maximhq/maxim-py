import logging
import time
from typing import Any, Dict, Iterable, List, Union
from uuid import uuid4

from maxim.logger import GenerationRequestMessage

BedrockMessageParam = Dict[str, Union[str, List[Dict[str, str]]]]
BedrockMessage = Dict[str, Any]

class BedrockUtils:
    @staticmethod
    def parse_message_param(
        messages: Iterable[BedrockMessageParam],
    ) -> List[GenerationRequestMessage]:
        """
        Parses Bedrock Converse API message format into GenerationRequestMessage format.
        """
        parsed_messages: List[GenerationRequestMessage] = []

        for msg in messages:
            role = msg.get("role")
            content_list = msg.get("content")

            if not isinstance(role, str) or not isinstance(content_list, list):
                # Skip malformed messages or raise an error, depending on desired strictness
                # logging.warning(f"Skipping malformed message: {msg}")
                continue

            combined_text_content = ""
            # Iterate through the content blocks (e.g., [{"text": "..."}, {"image": ...}])
            for content_block in content_list:
                if isinstance(content_block, dict) and "text" in content_block:
                    # Append text from text blocks
                    combined_text_content += content_block.get("text", "")
                # Future: Could add handling for other content types like images if needed

            # Create the GenerationRequestMessage with the extracted role and combined text
            parsed_messages.append(
                GenerationRequestMessage(
                    role=role,
                    content=combined_text_content
                )
            )

        return parsed_messages

    # Placeholder for get_model_parameters if needed by the client code
    @staticmethod
    def get_model_parameters(**kwargs: Any) -> Dict[str, Any]:
         """ Extracts model parameters for logging. """
         # Simple implementation for now, adjust as needed
         return {k: v for k, v in kwargs.items() if v is not None}

    @staticmethod
    def parse_message(
        response: BedrockMessage,
    ) -> Dict[str, Any]:
        """
        Parse the Message response from Bedrock Converse API into a standardized format.
        """
        output_data = response.get("output", {})
        message_data = output_data.get("message", {})
        usage_data = response.get("usage", {})

        # Extract content
        content = ""
        content_list = message_data.get("content", [])
        if isinstance(content_list, list):
            for block in content_list:
                if isinstance(block, dict) and block.get("type", "text") == "text":
                    content += block.get("text", "")
        else:
            content = str(content_list)

        return {
            "id": response.get("ResponseMetadata", {}).get("RequestId", str(uuid4())),
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": message_data.get("role", "assistant"),
                    "content": content,
                },
                "finish_reason": response.get("stopReason", "stop")
            }],
            "usage": {
                "prompt_tokens": usage_data.get("inputTokens", 0),
                "completion_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": usage_data.get("totalTokens", 0)
            }
        }
    
    @staticmethod
    def get_model_name(model_id: str) -> str:
        """
        Extracts the model name from the model ID or ARN.
        """
        try:
            # Check if it's an ARN
            if model_id.startswith("arn:"):
                # Split by '/' and take the last part (after inference-profile/)
                model_part = model_id.split('/')[-1]
            else:
                model_part = model_id

            # Split by '.' to handle the provider prefix (e.g., 'us.anthropic' or 'anthropic')
            parts = model_part.split('.')
            
            # Get the model name part (last part if multiple dots)
            model_with_version = parts[-1]
            
            # Remove version suffix (everything after and including the first '-v')
            model_name = model_with_version.split('-v')[0]
            
            # If it starts with provider name, remove it
            if model_name.startswith('anthropic.'):
                model_name = model_name[len('anthropic.'):]
            
            return model_name

        except Exception as e:
            logging.warning(
                f"[MaximSDK][BedrockUtils] Error parsing model ID: {e}. Returning original model ID."
            )
            return model_id
