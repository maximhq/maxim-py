from typing import Any, Dict, List, Optional 

from ..logger import GenerationRequestMessage


class TogetherUtils:
    """Utility functions for Together API integration.

    This class provides helper methods for parsing and formatting data
    between Together API and Maxim formats.
    """

    @staticmethod
    def get_model_params(**kwargs) -> Dict[str, Any]:
        """Extract model parameters from kwargs.

        Args:
            **kwargs: Keyword arguments passed to the API call.

        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        # Common parameters for both chat and completions
        params = {
            "temperature": kwargs.get("temperature", None),
            "top_p": kwargs.get("top_p", None),
            "top_k": kwargs.get("top_k", None),
            "max_tokens": kwargs.get("max_tokens", None),
            "stop": kwargs.get("stop", None),
            "frequency_penalty": kwargs.get("frequency_penalty", None),
            "presence_penalty": kwargs.get("presence_penalty", None),
            "logprobs": kwargs.get("logprobs", None),
            "logit_bias": kwargs.get("logit_bias", None),
            "stop_sequences": kwargs.get("stop_sequences", None),
        }
        
        # Remove None values
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def parse_message_param(messages: Optional[List[Dict[str, str]]]) -> List[GenerationRequestMessage]:
        """Parse Together message format to Maxim message format.

        Args:
            messages: List of messages in Together format.

        Returns:
            List[GenerationRequestMessage]: List of messages in Maxim format.
        """
        if not messages:
            return []

        maxim_messages = []
        for msg in messages:
            maxim_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        return maxim_messages

    @staticmethod
    def parse_chat_completion(response: Any) -> Dict[str, Any]:
        """Parse Together chat completion response to Maxim format.

        Args:
            response: Together chat completion response.

        Returns:
            Dict[str, Any]: Response in Maxim format.
        """
        return {
            "id": response.id,
            "choices": [
                {
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if hasattr(response, "usage") else None,
        }

    @staticmethod
    def parse_completion(response: Any) -> Dict[str, Any]:
        """Parse Together completion response to Maxim format.

        Args:
            response: Together completion response.

        Returns:
            Dict[str, Any]: Response in Maxim format.
        """
        return {
            "id": response.id,
            "choices": [
                {
                    "text": choice.text,
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if hasattr(response, "usage") else None,
        } 