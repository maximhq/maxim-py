import logging
from typing import Any, Iterator, List, Optional, TypeVar, Union
from uuid import uuid4

from google.genai import Client
from google.genai.chats import Chat, Chats
from google.genai.models import Models
from google.genai.types import (
    Content,
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    PartUnionDict,
)
from typing_extensions import override

from ..logger import (
    Generation,
    GenerationConfig,
    GenerationRequestMessage,
    Logger,
    Trace,
    TraceConfig,
)
from .async_client import MaximGeminiAsyncClient
from .utils import GeminiUtils

T = TypeVar("T")


class MaximGeminiChatSession(Chat):
    """Maxim Gemini chat session.

    This class represents a Maxim wrapped Gemini chat session.
    """

    def __init__(
        self,
        chat: Chat,
        logger: Logger,
        trace_id: Optional[str] = None,
        is_local_trace: Optional[bool] = False,
    ):
        """Initialize a Maxim wrapped Gemini chat session.

        Args:
            chat: The chat.
            logger: The logger.
            trace_id: The trace id.
            is_local_trace: Whether the trace is local.
        """
        self._chat = chat
        self._logger = logger
        self._trace_id = trace_id
        self._is_local_trace = is_local_trace

    @override
    def send_message(
        self,
        message: Union[list[PartUnionDict], PartUnionDict],
        generation_name: Optional[str] = None,
    ) -> GenerateContentResponse:
        """Send a message to the Maxim wrapped Gemini chat session.

        Args:
            message: The message to send.
            generation_name: The name of the generation.

        Returns:
            GenerateContentResponse: The response from the Maxim wrapped Gemini chat session.
        """
        # Without trace_id we can't do anything
        if self._trace_id is None:
            return super().send_message(message)
        generation: Optional[Generation] = None
        try:
            config = self._chat._config
            messages: List[GenerationRequestMessage] = []
            if config is not None:
                if isinstance(config, GenerateContentConfig):
                    if config.system_instruction is not None:
                        messages.append(
                            GeminiUtils.parse_content(
                                config.system_instruction, "system"
                            )
                        )
                elif isinstance(config, dict):
                    if (
                        instruction := config.get("system_instruction", None)
                    ) is not None:
                        messages.append(
                            GeminiUtils.parse_content(instruction, "system")
                        )
            messages.extend(GeminiUtils.parse_chat_message("user", message))
            gen_config = {
                "id": str(uuid4()),
                "model": self._model,
                "provider": "google",
                "name": generation_name,
                "model_parameters": GeminiUtils.get_model_params(config),
                "messages": messages,
            }
            generation = self._logger.trace_add_generation(self._trace_id, gen_config)
            # Attaching history as metadata
            if self._comprehensive_history is not None:
                generation.add_metadata({"history": self._comprehensive_history})
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in generating content: {str(e)}")
        # Actual call will never fail
        response = super().send_message(message)
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if response is not None:
                self._logger.trace_set_output(self._trace_id, response.text or "")
            if self._is_local_trace:
                self._logger.trace_end(self._trace_id)
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in logging generation: {str(e)}")
        # Returning response
        return response

    @override
    def send_message_stream(
        self,
        message: Union[list[PartUnionDict], PartUnionDict],
        generation_name: Optional[str] = None,
    ):
        """Send a message to the Maxim wrapped Gemini chat session stream.

        Args:
            message: The message to send.
            generation_name: The name of the generation.
        """
        # Without trace_id we can't do anything
        if self._trace_id is None:
            return super().send_message_stream(message)
        generation: Optional[Generation] = None
        try:
            config = self._chat._config
            messages: List[GenerationRequestMessage] = []
            if config is not None:
                if isinstance(config, GenerateContentConfig):
                    if config.system_instruction is not None:
                        messages.append(
                            GeminiUtils.parse_content(
                                config.system_instruction, "system"
                            )
                        )
                elif isinstance(config, dict):
                    if (
                        instruction := config.get("system_instruction", None)
                    ) is not None:
                        messages.append(
                            GeminiUtils.parse_content(instruction, "system")
                        )
            messages.extend(GeminiUtils.parse_chat_message("user", message))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=self._model,
                provider="google",
                name=generation_name,
                model_parameters=GeminiUtils.get_model_params(config),
                messages=messages,
            )
            generation = self._logger.trace_generation(self._trace_id, gen_config)
            # Attaching history as metadata
            if self._curated_history is not None:
                generation.add_metadata({"history": self._curated_history})
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in generating content: {str(e)}")
        # Actual call will never fail
        response = super().send_message_stream(message)
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if self._is_local_trace:
                self._logger.trace_end(self._trace_id)
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in logging generation: {str(e)}")
        # Returning response
        return response

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the chat session.

        Args:
            name: The name of the attribute.

        Returns:
            Any: The attribute.
        """
        return getattr(self._chat, name)        

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the chat session.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        if name in ("_chat", "_logger", "_trace_id", "_is_local_trace"):
            super().__setattr__(name, value)
        else:
            setattr(self._chat, name, value)

    def end_trace(self):
        """End the trace of the chat session.

        This method will end the trace of the chat session if the trace id is not None and the trace is local.
        """
        if self._trace_id is not None and self._is_local_trace:
            self._logger.trace_end(self._trace_id)
            self._trace_id = None


class MaximGeminiChats(Chats):
    """Maxim Gemini chats.

    This class represents a Maxim wrapped Gemini chats.
    """

    def __init__(self, chats: Chats, logger: Logger):
        """Initialize a Maxim wrapped Gemini chats.

        Args:
            chats: The chats.
            logger: The logger.
        """
        self._chats = chats
        self._logger = logger
        self._trace_id = None
        self._is_local_trace = False

    @override
    def create(
        self,
        *,
        model: str,
        config: GenerateContentConfigOrDict = None,  # type: ignore
        history: Optional[list[Content]] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Chat:
        """Create a Maxim wrapped Gemini chat session.

        Args:
            model: The model to use.
            config: The config to use.
            history: The history to use.
            trace_id: The trace id.
            session_id: The session id.

        Returns:
            Chat: The Maxim wrapped Gemini chat session.
        """
        self._is_local_trace = trace_id is None
        self._trace_id = trace_id or str(uuid4())
        if session_id is not None:
            self._logger.session({"id": session_id, "name": "Chat session"})
            self._logger.session_add_trace(
                session_id,
                {
                    "id": self._trace_id,
                    "name": "Chat turn",
                    "session_id": session_id,
                },
            )
        else:
            self._logger.trace(
                {
                    "id": self._trace_id,
                    "name": "Chat session",
                    "session_id": session_id,
                }
            )
        # we start generation here and send it back to chat session
        # every round trip of chat session will be logged in a separate trace
        chat_session = self._chats.create(model=model, config=config, history=history)
        maxim_chat_session = MaximGeminiChatSession(
            chat=chat_session,
            logger=self._logger,
            trace_id=self._trace_id,
            is_local_trace=self._is_local_trace,
        )
        return maxim_chat_session

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the chats.

        Args:
            name: The name of the attribute.

        Returns:
            Any: The attribute.
        """
        result = getattr(self._chats, name)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the chats.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        if name in ("_chats", "_logger", "_trace_id", "_is_local_trace"):
            super().__setattr__(name, value)
        else:
            setattr(self._chats, name, value)


class MaximGeminiModels(Models):
    """Maxim Gemini models.

    This class represents a Maxim wrapped Gemini models.
    """

    def __init__(self, models: Models, logger: Logger):
        """Initialize a Maxim wrapped Gemini models.

        Args:
            models: The models.
            logger: The logger.
        """
        self._models = models
        self._logger = logger

    @override
    def generate_content_stream(
        self,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
    ) -> Iterator[GenerateContentResponse]:
        """Generate content stream.

        Args:
            model: The model to use.
            contents: The contents to use.
            config: The config to use.
            trace_id: The trace id.
            generation_name: The generation name.

        Returns:
            Iterator[GenerateContentResponse]: The content stream.
        """
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            # Checking if there is a system prompt
            messages: List[GenerationRequestMessage] = []
            if config is not None:
                if isinstance(config, GenerateContentConfig):
                    if config.system_instruction is not None:
                        messages.append(
                            GeminiUtils.parse_content(
                                config.system_instruction, "system"
                            )
                        )
                elif isinstance(config, dict):
                    if (
                        instruction := config.get("system_instruction", None)
                    ) is not None:
                        messages.append(
                            GeminiUtils.parse_content(instruction, "system")
                        )
            # Adding contents back
            if contents is not None:
                messages.extend(GeminiUtils.parse_messages(contents))

            gen_config = {
                "id": str(uuid4()),
                "model": model,
                "name": generation_name,
                "provider": "google",
                "model_parameters": GeminiUtils.get_model_params(config),
                "messages": messages,
            }
            generation = trace.generation(gen_config)
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in generating content: {str(e)}")
        # Actual call will never fail
        chunks = super().generate_content_stream(
            model=model, contents=contents, config=config
        )
        # Actual call ends
        try:
            if generation is not None:
                generation.result(chunks)
            if is_local_trace:
                if trace is not None:
                    try:
                        if isinstance(chunks, GenerateContentResponse):
                            trace.set_output(chunks.text or "")
                    except Exception as e:
                        logging.warning(
                            f"[MaximSDK][Gemini] Error in logging generation: {str(e)}"
                        )
                    trace.end()
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in logging generation: {str(e)}")
        # Actual response
        return chunks

    @override
    def generate_content(
        self,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
    ) -> GenerateContentResponse:
        """Generate content.

        Args:
            model: The model to use.
            contents: The contents to use.
            config: The config to use.
            trace_id: The trace id.
            generation_name: The generation name.

        Returns:
            GenerateContentResponse: The content.
        """
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            # Checking if there is a system prompt
            messages: List[GenerationRequestMessage] = []
            if config is not None:
                if isinstance(config, GenerateContentConfig):
                    if config.system_instruction is not None:
                        messages.append(
                            GeminiUtils.parse_content(
                                config.system_instruction, "system"
                            )
                        )
                elif isinstance(config, dict):
                    if (
                        instruction := config.get("system_instruction", None)
                    ) is not None:
                        messages.append(
                            GeminiUtils.parse_content(instruction, "system")
                        )
            # Adding contents back
            if contents is not None:
                messages.extend(GeminiUtils.parse_messages(contents))

            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="google",
                name=generation_name,
                model_parameters=GeminiUtils.get_model_params(config),
                messages=messages,
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in generating content: {str(e)}")

        # Actual call will never fail
        response = self._models.generate_content(
            model=model, contents=contents, config=config
        )
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if is_local_trace:
                if trace is not None:
                    trace.set_output(response.text or "")
                    trace.end()
        except Exception as e:
            logging.warning(f"[MaximSDK][Gemini] Error in logging generation: {str(e)}")
        # Actual response
        return response

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the models.

        Args:
            name: The name of the attribute.

        Returns:
            Any: The attribute.
        """
        return getattr(self._models, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the models.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        if name in ("_models", "_logger"):
            super().__setattr__(name, value)
        else:
            setattr(self._models, name, value)


class MaximGeminiClient(Client):
    """Maxim Gemini client.

    This class represents a Maxim wrapped Gemini client.
    """

    def __init__(self, client: Client, logger: Logger):
        """Initialize a Maxim wrapped Gemini client.

        Args:
            client: The client.
            logger: The logger.
        """
        self._client = client
        self._logger = logger
        self._w_models = MaximGeminiModels(client.models, logger)
        self._w_chats = MaximGeminiChats(client.chats, logger)
        self._w_aio = MaximGeminiAsyncClient(client._aio, logger)

    @property
    def chats(self) -> MaximGeminiChats:
        """Get the Maxim wrapped Gemini chats.

        Returns:
            MaximGeminiChats: The Maxim wrapped Gemini chats.
        """
        return self._w_chats

    @property
    def aio(self) -> MaximGeminiAsyncClient:
        """Get the Maxim wrapped Gemini async client.

        Returns:
            MaximGeminiAsyncClient: The Maxim wrapped Gemini async client.
        """
        return self._w_aio

    @property
    def models(self) -> MaximGeminiModels:
        """Get the Maxim wrapped Gemini models.

        Returns:
            MaximGeminiModels: The Maxim wrapped Gemini models.
        """
        return self._w_models

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the client.

        Args:
            name: The name of the attribute.

        Returns:
            Any: The attribute.
        """
        if name == "_models":
            return self._w_models
        elif name == "_chats":
            return self._w_chats
        elif name == "_aio":
            return self._w_aio
        return getattr(self._client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the client.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        if name in ("_client", "_logger", "_w_models", "_w_chats", "_w_aio"):
            super().__setattr__(name, value)
        else:
            setattr(self._client, name, value)
