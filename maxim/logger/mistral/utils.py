import time
from typing import Any, Dict, Iterable, List, Optional

from mistralai.models import ChatCompletionResponse, CompletionChunk, CompletionEvent

from ..logger import GenerationRequestMessage


class MistralUtils:
    @staticmethod
    def parse_message_param(
        messages: Optional[Iterable[Any]],
    ) -> List[GenerationRequestMessage]:
        """Convert Mistral message objects or dictionaries to log-friendly structures."""

        parsed_messages: List[GenerationRequestMessage] = []

        if messages is None:
            return parsed_messages

        for msg in messages:
            role = None
            content = None

            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = MistralUtils._message_content(msg)

            parsed_messages.append(
                GenerationRequestMessage(role=str(role), content=str(content))
            )

        return parsed_messages

    @staticmethod
    def get_model_params(**kwargs: Any) -> Dict[str, Any]:
        model_params: Dict[str, Any] = {}
        param_keys = [
            "temperature",
            "top_p",
            "max_tokens",
            "random_seed",
            "safe_prompt",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        ]
        for key in param_keys:
            if key in kwargs and kwargs[key] is not None:
                model_params[key] = kwargs[key]
        for key, value in kwargs.items():
            if key not in model_params and value is not None:
                model_params[key] = value
        return model_params

    @staticmethod
    def _message_content(message: Any) -> str:
        """Extract textual content from a Mistral message object."""
        if not message:
            return ""

        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        text = ""
        # Newer SDK versions return a list of ContentChunk instances
        if isinstance(content, list):
            for part in content:
                if hasattr(part, "text") and part.text:
                    text += str(part.text)
        else:
            # Fallback for older SDK versions with `.parts`
            parts = getattr(content, "parts", None)
            if parts:
                for part in parts:
                    if getattr(part, "text", None):
                        text += part.text

        return text

    @staticmethod
    def parse_completion(completion: ChatCompletionResponse) -> Dict[str, Any]:
        return {
            "id": completion.id,
            "created": completion.created,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": "assistant",
                        "content": MistralUtils._message_content(choice.message),
                    },
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
                for choice in completion.choices
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
            },
        }

    @staticmethod
    def parse_stream_response(event: CompletionEvent) -> Dict[str, Any]:
        chunk: CompletionChunk = event.data
        return {
            "id": chunk.id,
            "created": chunk.created or int(time.time()),
            "choices": [
                {
                    "index": c.index,
                    "delta": {
                        "role": getattr(c.delta, "role", "assistant"),
                        "content": MistralUtils._message_content(c.delta),
                    },
                    "finish_reason": getattr(c, "finish_reason", None),
                }
                for c in chunk.choices
            ],
        }

    @staticmethod
    def combine_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {}
        last_chunk = chunks[-1]
        text = "".join(
            [
                choice.get("delta", {}).get("content", "")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            ]
        )
        return {
            "id": last_chunk.get("id", ""),
            "created": last_chunk.get("created", int(time.time())),
            "choices": [
                {
                    "index": last_chunk.get("choices", [{}])[0].get("index", 0),
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": last_chunk.get("choices", [{}])[0].get("finish_reason"),
                }
            ],
        }
