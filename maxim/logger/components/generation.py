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
from .utils import parse_attachments_from_messages


class GenerationRequestTextMessageContent(TypedDict):
    """
    This class is used to represent a text message in a generation request.
    """

    type: Literal["text"]
    text: str


class GenerationRequestImageMessageContent(TypedDict):
    """
    This class is used to represent an image message in a generation request.
    """

    type: Literal["image"]
    image_url: str


class GenerationRequestMessage(TypedDict):
    """
    This class is used to represent a message in a generation request.
    """

    role: str
    content: Union[
        str,
        List[
            Union[
                GenerationRequestTextMessageContent,
                GenerationRequestImageMessageContent,
            ]
        ],
    ]


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
    """Generation config dict.

    This class represents a generation config dictionary.
    """

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
    """Convert a generation config to a generation config dict else return the config.

    Args:
        config (Union[GenerationConfig, GenerationConfigDict]): The config to get the dict from.

    Returns:
        dict[str, Any]: The generation config dict.
    """
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
    "fireworks",
    "elevenlabs",
    "unknown",
]


class GenerationToolCallFunction(TypedDict):
    """Generation tool call function.

    This class represents a tool call function.
    """

    name: str
    arguments: Optional[str]


class GenerationToolCall(TypedDict):
    """Generation tool call.

    This class represents a tool call.
    """

    id: str
    type: str
    function: GenerationToolCallFunction


class TextContent(TypedDict):
    """Text content.

    This class represents a text content.
    """

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """Image content.

    This class represents an image content.
    """

    type: Literal["image"]
    image_url: str


class AudioContent(TypedDict):
    """Audio content.

    This class represents an audio content.
    """

    type: Literal["audio"]
    transcript: str


class GenerationResultMessage(TypedDict):
    """Generation result message.

    This class represents a generation result message.
    """

    role: str
    content: Optional[Union[List[Union[TextContent, ImageContent, AudioContent]], str]]
    tool_calls: Optional[List[GenerationToolCall]]


class GenerationResultChoice(TypedDict):
    """Generation result choice.

    This class represents a generation result choice.
    """

    index: int
    message: GenerationResultMessage
    logprobs: Optional[Any]
    finish_reason: Optional[str]


class TokenDetails(TypedDict):
    """Token details.

    This class represents token details.
    """

    text_tokens: int
    audio_tokens: int
    cached_tokens: int


class GenerationUsage(TypedDict, total=False):
    """Generation usage.

    This class represents generation usage.
    """

    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    input_audio_duration: Optional[float]
    output_audio_duration: Optional[float]
    input_token_details: Optional[TokenDetails]
    output_token_details: Optional[TokenDetails]
    cached_token_details: Optional[TokenDetails]


class GenerationResult(TypedDict):
    """Generation result.

    This class represents a generation result.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[GenerationResultChoice]
    usage: GenerationUsage


def get_generation_error_config_dict(
    config: Union[GenerationError, GenerationErrorTypedDict],
) -> GenerationErrorTypedDict:
    """Convert a generation error to a generation error dict else return the error.

    Args:
        config: Either a GenerationConfig object or a GenerationConfigDict dictionary.

    Returns:
        A GenerationConfigDict dictionary representation of the config.
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
        """
        Initialize a generation.

        Args:
            config: The config to initialize the generation with.
            writer: The writer to use.
        """
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
        self.messages, attachments = parse_attachments_from_messages(self.messages)
        if len(attachments) > 0:
            for attachment in attachments:
                self.add_attachment(attachment)
        self.model_parameters = parse_model_parameters(
            final_config.get("model_parameters", {})
        )

    @staticmethod
    def set_provider_(writer: LogWriter, id: str, provider: str):
        """
        Static method to set the provider for a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to set the provider for.
            provider: The provider to set.
        """
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of {', '.join(valid_providers)}."
            )
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"provider": provider}
        )

    def set_provider(self, provider: str):
        """
        Set the provider for this generation.

        Args:
            provider: The provider to set.
        """
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. Must be one of {', '.join(valid_providers)}."
            )
        self.provider = provider

    @staticmethod
    def set_model_(writer: LogWriter, id: str, model: str):
        """
        Static method to set the model for a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to set the model for.
            model: The model to set.
        """
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"model": model}
        )

    def set_model(self, model: str):
        """
        Set the model for this generation.

        Args:
            model: The model to set.
        """
        self.model = model
        self._commit("update", {"model": model})
        
    def set_name(self, name: str):
        """
        Set the name for this generation.

        Args:
            name: The name to set.
        """
        self._name = name
        self._commit("update", {"name": name})
        
    @staticmethod
    def set_name_(writer: LogWriter, id: str, name: str):
        """
        Static method to set the name for a generation.
        """
        BaseContainer._commit_(writer, Entity.GENERATION, id, "update", {"name": name})

    def add_metric(self, name: str, value: float) -> None:
        """
        Add a metric to this generation.

        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        self._commit("update", {"metrics": {"name": name, "value": value}})

    @staticmethod
    def add_metric_(writer: LogWriter, id: str, name: str, value: float):
        """
        Static method to add a metric to a generation.
        """
        Generation._commit_(
            writer,
            Entity.GENERATION,
            id,
            "update",
            {"metrics": {"name": name, "value": value}},
        )

    @staticmethod
    def add_message_(writer: LogWriter, id: str, message: GenerationRequestMessage):
        """
        Static method to add a message to a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to add the message to.
            message: The message to add.
        """
        if "content" not in message or "role" not in message:
            scribe().error(
                "[MaximSDK] Invalid message. Must have 'content' and 'role' keys. We are skipping adding this message."
            )
            return
        messages, attachments = parse_attachments_from_messages([message])
        if len(attachments) > 0:
            for attachment in attachments:
                Generation.add_attachment_(writer, id, attachment)
        BaseContainer._commit_(
            writer, Entity.GENERATION, id, "update", {"messages": messages}
        )

    def add_message(self, message: GenerationRequestMessage) -> None:
        """
        Add a message to this generation.

        Args:
            message: The message to add.
        """
        messages, attachments = parse_attachments_from_messages([message])
        if len(attachments) > 0:
            for attachment in attachments:
                self.add_attachment(attachment)
        self._commit("update", {"messages": messages})

    @staticmethod
    def set_model_parameters_(
        writer: LogWriter, id: str, model_parameters: Dict[str, Any]
    ):
        """
        Static method to set the model parameters for a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to set the model parameters for.
            model_parameters: The model parameters to set.
        """
        model_parameters = parse_model_parameters(model_parameters)
        BaseContainer._commit_(
            writer,
            Entity.GENERATION,
            id,
            "update",
            {"modelParameters": model_parameters},
        )

    def set_model_parameters(self, model_parameters: Dict[str, Any]):
        """
        Set the model parameters for this generation.

        Args:
            model_parameters: The model parameters to set.
        """
        model_parameters = parse_model_parameters(model_parameters)
        self.model_parameters = model_parameters
        self._commit("update", {"modelParameters": model_parameters})

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        """
        Add an attachment to this generation.

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
        Static method to add an attachment to a generation.

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
        """
        Static method to add a result to a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to add the result to.
            result: The result to add.
        """
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
        """
        Static method to end a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to end.
            data: The data to add to the generation.
        """
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
        """
        Static method to add a tag to a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to add the tag to.
            key: The key of the tag to add.
            value: The value of the tag to add.
        """
        BaseContainer._add_tag_(writer, Entity.GENERATION, id, key, value)

    @staticmethod
    def convert_chat_completion(chat_completion: Dict[str, Any]):
        """
        Convert a chat completion to a generation result.

        Args:
            chat_completion: The chat completion to convert.

        Returns:
            A generation result.
        """
        return {
            "id": chat_completion.get("id", str(uuid4())),
            "created": chat_completion.get("created", datetime.now(timezone.utc)),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": (choice.get("message", {}) or {}).get(
                            "role", "assistant"
                        ),
                        "content": (choice.get("message", {}) or {}).get("content", ""),
                        "tool_calls": (choice.get("message", {}) or {}).get(
                            "tool_calls", None
                        ),
                        "function_calls": (choice.get("message", {}) or {}).get(
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
        """
        Convert a result to a generation result.

        Args:
            result: The result to convert.

        Returns:
            A generation result.
        """
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
                elif "object" in result_dict and result_dict["object"] == "response":
                    # OpenAI Responses API format - return as-is for logging
                    return result_dict
            return result

    def result(self, result: Any):
        """
        Add a result to this generation.

        Args:
            result: The result to add.
        """
        try:
            # Checking the type
            result = Generation.convert_result(result)
            # Validating the result
            parse_result(result)
            # Taking out attachments from the result
            self._commit("result", {"result": result})
            self.end()
        except ValueError as e:
            scribe().error(
                f"[MaximSDK] Invalid result. You can pass OpenAI/Azure ChatCompletion or Langchain LLMResult, AIMessage, ToolMessage, Gemini result or LiteLLM ModelResponse: {str(e)}",
            )

    def error(self, error: Union[GenerationError, GenerationErrorTypedDict]):
        """
        Add an error to this generation.

        Args:
            error: The error to add.
        """
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
        """
        Static method to add an error to a generation.

        Args:
            writer: The LogWriter instance to use.
            id: The ID of the generation to add the error to.
            error: The error to add.
        """
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
        """
        Get the data for this generation.

        Returns:
            A dictionary containing the data for this generation.
        """
        base_data = super().data()
        return {
            **base_data,
            "model": self.model,
            "provider": self.provider,
            "maximPromptId": self.maxim_prompt_id,
            "messages": self.messages,
            "modelParameters": self.model_parameters,
        }
