from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict, Union
from uuid import uuid4

from typing_extensions import deprecated

from ...scribe import scribe
from ..parsers.generation_parser import parse_model_parameters, parse_result
from ..writer import LogWriter
from .attachment import FileAttachment, FileDataAttachment, UrlAttachment
from .base import BaseContainer
from .types import Entity, GenerationError, GenerationErrorTypedDict, object_to_dict


class GenerationRequestMessageContent(TypedDict):
    type: str
    text: str


class GenerationRequestMessage(TypedDict):
    role: str
    content: Union[str, List[GenerationRequestMessageContent]]


def generation_request_from_gemini_content(content: Any) -> "GenerationRequestMessage":
    if "role" not in content or "parts" not in content:
        raise ValueError("[MaximSDK] Invalid Gemini content")
    if not isinstance(content["parts"], list):
        raise ValueError("[MaximSDK] Invalid parts in Gemini content.")
    parts_content = ""
    for part in content["parts"]:
        parts_content += part["text"]
    return GenerationRequestMessage(role=content["role"], content=parts_content)


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class GenerationConfig:
    id: str
    provider: str
    model: str
    messages: Optional[List[GenerationRequestMessage]] = field(default_factory=list)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    span_id: Optional[str] = None
    name: Optional[str] = None
    maxim_prompt_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class GenerationConfigDict(TypedDict, total=False):
    id: str
    provider: str
    model: str
    messages: Optional[List[GenerationRequestMessage]]
    model_parameters: Dict[str, Any]
    span_id: Optional[str]
    name: Optional[str]
    maxim_prompt_id: Optional[str]
    tags: Optional[Dict[str, str]]


def get_generation_config_dict(
    config: Union[GenerationConfig, GenerationConfigDict],
) -> dict[str, Any]:
    if isinstance(config, GenerationConfig):
        return dict(
            GenerationConfigDict(
                id=config.id,
                provider=config.provider,
                model=config.model,
                messages=config.messages,
                model_parameters=config.model_parameters,
                span_id=config.span_id,
                name=config.name,
                maxim_prompt_id=config.maxim_prompt_id,
                tags=config.tags,
            )
        )
    elif isinstance(config, dict):
        return dict(GenerationConfigDict(**config))


valid_providers = [
    "openai",
    "azure",
    "anthropic",
    "huggingface",
    "together",
    "google",
    "groq",
    "bedrock",
    "cohere",
    "unknown",
]


class GenerationToolCallFunction(TypedDict):
    name: str
    arguments: Optional[str]


class GenerationToolCall(TypedDict):
    id: str
    type: str
    function: GenerationToolCallFunction


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    type: Literal["image"]
    image_url: str


class AudioContent(TypedDict):
    type: Literal["audio"]
    transcript: str


class GenerationResultMessage(TypedDict):
    role: str
    content: Optional[Union[List[Union[TextContent, ImageContent, AudioContent]], str]]
    tool_calls: Optional[List[GenerationToolCall]]


class GenerationResultChoice(TypedDict):
    index: int
    message: GenerationResultMessage
    logprobs: Optional[Any]
    finish_reason: Optional[str]


class TokenDetails(TypedDict):
    text_tokens: int
    audio_tokens: int
    cached_tokens: int


class GenerationUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_token_details: Optional[TokenDetails]
    output_token_details: Optional[TokenDetails]
    cached_token_details: Optional[TokenDetails]


class GenerationResult(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[GenerationResultChoice]
    usage: GenerationUsage


def get_generation_error_config_dict(
    config: Union[GenerationError, GenerationErrorTypedDict],
) -> GenerationErrorTypedDict:
    """
    Convert a TraceConfig object to a TraceConfigDict.

    Args:
        config: Either a TraceConfig object or a TraceConfigDict dictionary.

    Returns:
        A TraceConfigDict dictionary representation of the config.
    """
    return (
        GenerationErrorTypedDict(
            message=config.message,
            code=config.code,
            type=config.type,
        )
        if isinstance(config, GenerationError)
        else config
    )


class Generation(BaseContainer):
    def __init__(
        self, config: Union[GenerationConfig, GenerationConfigDict], writer: LogWriter
    ):
        final_config = get_generation_config_dict(config)
        super().__init__(Entity.GENERATION, final_config, writer)
        self.model = final_config.get("model", None)
        self.maxim_prompt_id = final_config.get("maxim_prompt_id", None)
        self.messages = []
        self.provider = final_config.get("provider", None)
        if self.provider is not None:
            self.provider = self.provider.lower()
            if self.provider not in valid_providers:
                self.provider = "unknown"
        else:
            self.provider = "unknown"
        self.messages.extend([m for m in (final_config.get("messages") or [])])
        self.model_parameters = parse_model_parameters(
            final_config.get("model_parameters", {})
        )

    @staticmethod
    def set_provider_(writer: LogWriter, id: str, provider: str):
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of {', '.join(valid_providers)}."
            )
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"provider": provider}
        )

    def set_provider(self, provider: str):
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. Must be one of {', '.join(valid_providers)}."
            )
        self.provider = provider

    @staticmethod
    def set_model_(writer: LogWriter, id: str, model: str):
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"model": model}
        )

    def set_model(self, model: str):
        self.model = model
        self._commit("update", {"model": model})

    @staticmethod
    def add_message_(writer: LogWriter, id: str, message: GenerationRequestMessage):
        if "content" not in message or "role" not in message:
            scribe.error(
                "[MaximSDK] Invalid message. Must have 'content' and 'role' keys. We are skipping adding this message."
            )
            return
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"messages": [message]}
        )

    def add_message(self, message: GenerationRequestMessage):
        self.messages.append(message)
        self._commit("update", {"messages": [message]})

    @staticmethod
    def set_model_parameters_(
        writer: LogWriter, id: str, model_parameters: Dict[str, Any]
    ):
        model_parameters = parse_model_parameters(model_parameters)
        BaseContainer._commit_(
            writer,
            Entity.GENERATION,
            id,
            "update",
            {"modelParameters": model_parameters},
        )

    def set_model_parameters(self, model_parameters: Dict[str, Any]):
        model_parameters = parse_model_parameters(model_parameters)
        self.model_parameters = model_parameters
        self._commit("update", {"modelParameters": model_parameters})

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        """
        Add an attachment to this trace.

        Args:
            attachment: The attachment to add.
        """
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(
        writer: LogWriter,
        generation_id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Static method to add an attachment to a trace.

        Args:
            writer: The LogWriter instance to use.
            generation_id: The ID of the generation to add the attachment to.
            attachment: The attachment to add.
        """
        Generation._commit_(
            writer,
            Entity.GENERATION,
            generation_id,
            "upload-attachment",
            attachment.to_dict(),
        )

    @staticmethod
    def result_(
        writer: LogWriter, id: str, result: Union[GenerationResult, Dict[str, Any]]
    ):
        try:
            # Checking the type
            result = Generation.convert_result(result)
            # Validating the result
            parse_result(result)
            BaseContainer._commit_(
                writer, Entity.GENERATION, id, "result", {"result": result}
            )
            BaseContainer._end_(
                writer,
                Entity.GENERATION,
                id,
                {
                    "endTimestamp": datetime.now(timezone.utc),
                },
            )
        except Exception as e:
            import traceback

            scribe().error(
                f"[MaximSDK] Invalid result. You can pass OpenAI/Azure ChatCompletion or Langchain LLMResult,AIMessage,ToolMessage or LiteLLM ModelResponse: {str(e)}",
                traceback.format_exc(),
            )

    @staticmethod
    def end_(writer: LogWriter, id: str, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = {}
        BaseContainer._end_(
            writer,
            Entity.GENERATION,
            id,
            {
                "endTimestamp": datetime.now(timezone.utc),
                **data,
            },
        )

    @staticmethod
    def add_tag_(writer: LogWriter, id: str, key: str, value: str):
        BaseContainer._add_tag_(writer, Entity.GENERATION, id, key, value)

    @staticmethod
    def convert_chat_completion(chat_completion: Dict[str, Any]):
        return {
            "id": chat_completion.get("id", str(uuid4())),
            "created": chat_completion.get("created", datetime.now(timezone.utc)),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": choice.get("message").get("role", "assistant"),
                        "content": choice.get("message").get("content", ""),
                        "tool_calls": choice.get("message").get("tool_calls", None),
                        "function_calls": choice.get("message").get(
                            "function_calls", None
                        ),
                    },
                    "finish_reason": choice.get("finish_reason", None),
                    "logprobs": choice.get("logprobs", None),
                }
                for choice in chat_completion.get("choices", [])
            ],
            "usage": chat_completion.get("usage", {}),
        }

    @staticmethod
    def convert_result(
        result: Union[Any, GenerationResult, Dict[str, Any]],
    ) -> Union[Any, GenerationResult, Dict[str, Any]]:
        try:
            parse_result(result)
            return result
        except Exception:
            if isinstance(result, object):
                # trying for langchain first
                try:
                    from langchain.schema import LLMResult
                    from langchain_core.messages import (  # type: ignore
                        AIMessage,
                    )
                    from langchain_core.outputs import (  # type: ignore
                        ChatGeneration,
                        ChatResult,
                    )

                    from ..langchain.utils import (
                        parse_base_message_to_maxim_generation,
                        parse_langchain_chat_generation,
                        parse_langchain_chat_result,
                        parse_langchain_llm_result,
                    )

                    if isinstance(result, AIMessage):
                        return parse_base_message_to_maxim_generation(result)
                    elif isinstance(result, LLMResult):
                        return parse_langchain_llm_result(result)
                    elif isinstance(result, ChatResult):
                        return parse_langchain_chat_result(result)
                    elif isinstance(result, ChatGeneration):
                        return parse_langchain_chat_generation(result)
                except ImportError:
                    pass
                # trying for litellm
                try:
                    from litellm.types.utils import ModelResponse  # type: ignore

                    from ..litellm.parser import parse_litellm_model_response

                    if isinstance(result, ModelResponse):
                        return parse_litellm_model_response(result)
                except ImportError:
                    pass
                # trying gemini response
                try:
                    from google.genai.types import GenerateContentResponse

                    from ..gemini.utils import GeminiUtils

                    if isinstance(result, GenerateContentResponse):
                        return GeminiUtils.parse_gemini_generation_content(result)
                    elif isinstance(result, Iterator):
                        return GeminiUtils.parse_gemini_generation_content_iterator(
                            result
                        )
                except ImportError:
                    pass
                # trying for anthropic
                try:
                    from anthropic.lib.streaming import MessageStopEvent
                    from anthropic.types import Message

                    from ..anthropic import AnthropicUtils

                    if isinstance(result, Message):
                        res = AnthropicUtils.parse_message(result)
                        return res
                except ImportError:
                    pass
                # trying for bedrock
                try:
                    from ..bedrock import BedrockUtils

                    if (
                        isinstance(result, dict)
                        and "output" in result
                        and "input" in result
                    ):
                        res = BedrockUtils.parse_message(result)
                        return res
                except ImportError:
                    pass
            result_dict = object_to_dict(result)
            if isinstance(result_dict, Dict):
                # Checking if its Azure or OpenAI result
                if (
                    "object" in result_dict
                    and result_dict["object"] == "chat.completion"
                ):
                    return Generation.convert_chat_completion(result_dict)
                elif (
                    "object" in result_dict
                    and result_dict["object"] == "text.completion"
                ):
                    raise ValueError("Text completion is not yet supported.")
            return result

    def result(self, result: Any):
        try:
            # Checking the type
            result = Generation.convert_result(result)
            # Validating the result
            parse_result(result)
            # Logging the result
            self._commit("result", {"result": result})
            self.end()
        except ValueError as e:
            import traceback

            scribe().error(
                f"[MaximSDK] Invalid result. You can pass OpenAI/Azure ChatCompletion or Langchain LLMResult, AIMessage, ToolMessage, Gemini result or LiteLLM ModelResponse: {str(e)}",
                traceback.format_exc(),
            )

    def error(self, error: Union[GenerationError, GenerationErrorTypedDict]):
        final_error = get_generation_error_config_dict(error)
        if not final_error.get("code"):
            final_error["code"] = ""
        if not final_error.get("type"):
            final_error["type"] = ""
        self._commit(
            "result",
            {
                "result": {
                    "error": {
                        "message": final_error.get("message", ""),
                        "code": final_error.get("code", ""),
                        "type": final_error.get("type", ""),
                    },
                    "id": str(uuid4()),
                }
            },
        )
        self.end()

    @staticmethod
    def error_(
        writer: LogWriter,
        id: str,
        error: Union[GenerationError, GenerationErrorTypedDict],
    ):
        final_error = get_generation_error_config_dict(error)
        if not final_error.get("code"):
            final_error["code"] = ""
        if not final_error.get("type"):
            final_error["type"] = ""
        BaseContainer._commit_(
            writer,
            Entity.GENERATION,
            id,
            "result",
            {
                "result": {
                    "error": {
                        "message": final_error.get("message", ""),
                        "code": final_error.get("code", ""),
                        "type": final_error.get("type", ""),
                    },
                    "id": str(uuid4()),
                }
            },
        )
        BaseContainer._end_(
            writer,
            Entity.GENERATION,
            id,
            {
                "endTimestamp": datetime.now(timezone.utc),
            },
        )

    def data(self) -> Dict[str, Any]:
        base_data = super().data()
        return {
            **base_data,
            "model": self.model,
            "provider": self.provider,
            "maximPromptId": self.maxim_prompt_id,
            "messages": self.messages,
            "modelParameters": self.model_parameters,
        }
