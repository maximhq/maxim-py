import logging
from typing import Any, AsyncIterator, Awaitable, List, Optional, Union
from uuid import uuid4

from google.genai.chats import AsyncChat
from google.genai.client import AsyncChats, AsyncClient, AsyncModels
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
from .utils import GeminiUtils


class MaximGeminiAsyncChatSession(AsyncChat):
    def __init__(
        self,
        chat: AsyncChat,
        logger: Logger,
        trace_id: Optional[str] = None,
        is_local_trace: Optional[bool] = False,
    ):
        super().__init__(
            modules=chat._modules,
            model=chat._model,
            config=chat._config,
            history=chat._curated_history,
        )
        self._chat = chat
        self._logger = logger
        self._trace_id = trace_id
        self._is_local_trace = is_local_trace

    @override
    async def send_message(
        self,
        message: Union[list[PartUnionDict], PartUnionDict],
        generation_name: Optional[str] = None,
    ) -> GenerateContentResponse:
        # Without trace_id we can't do anything
        if self._trace_id is None:
            return await super().send_message(message)
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
            logging.warning(
                f"[MaximSDK][GeminiAsyncClient] Error in generating content: {str(e)}"
            )
        # Actual call will never fail
        response = await super().send_message(message)
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if self._is_local_trace:
                self._logger.trace_end(self._trace_id)
        except Exception as e:
            logging.warning(
                f"[MaximSDK][GeminiAsyncClient] Error in logging generation: {str(e)}"
            )
        # Returning response
        return response

    @override
    async def send_message_stream(
        self,
        message: Union[list[PartUnionDict], PartUnionDict],
        generation_name: Optional[str] = None,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]:
        # Without trace_id we can't do anything
        if self._trace_id is None:
            return await super().send_message_stream(message)
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
            logging.warning(
                f"[MaximSDK][GeminiAsyncClient] Error in generating content: {str(e)}"
            )
        # Actual call will never fail
        response = await super().send_message_stream(message)
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if self._is_local_trace:
                self._logger.trace_end(self._trace_id)
        except Exception as e:
            logging.warning(
                f"[MaximSDK][GeminiAsyncClient] Error in logging generation: {str(e)}"
            )
        # Returning response
        return response

    def __getattr__(self, name: str) -> Any:
        result = getattr(self._chats, name)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_chat":
            super().__setattr__(name, value)
        else:
            setattr(self._chats, name, value)

    def end_trace(self):
        if self._trace_id is not None and self._is_local_trace:
            self._logger.trace_end(self._trace_id)
            self._trace_id = None


class MaximGeminiAsyncChats(AsyncChats):
    def __init__(self, chats: AsyncChats, logger: Logger):
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
    ) -> AsyncChat:
        self._is_local_trace = trace_id is None
        self._trace_id = trace_id or str(uuid4())
        # we start generation here and send it back to chat session
        # every round trip of chat session will be logged in a separate trace
        chat_session = self._chats.create(model=model, config=config, history=history)
        maxim_chat_session = MaximGeminiAsyncChatSession(
            chat=chat_session,
            logger=self._logger,
            trace_id=self._trace_id,
            is_local_trace=self._is_local_trace,
        )
        return maxim_chat_session

    def __getattr__(self, name: str) -> Any:
        result = getattr(self._chats, name)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_chats":
            super().__setattr__(name, value)
        else:
            setattr(self._chats, name, value)


class MaximGeminiAsyncModels(AsyncModels):
    def __init__(self, models: AsyncModels, logger: Logger):
        self._models = models
        self._logger = logger

    @override
    async def generate_content_stream(
        self,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]:
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
                name=generation_name,
                provider="google",
                model_parameters=GeminiUtils.get_model_params(config),
                messages=messages,
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            logging.warning(
                f"[MaximSDK][GeminiAsyncModels] Error in generating content: {str(e)}"
            )
        # Actual call will never fail
        chunks = await super().generate_content_stream(
            model=model, contents=contents, config=config
        )
        # Actual call ends
        try:
            if generation is not None:
                generation.result(chunks)
            if is_local_trace:
                if trace is not None:
                    if chunks is not None:
                        a_itr = await chunks
                        async for chunk in a_itr:
                            trace.set_output(" ".join([chunk.text or ""]))
                    trace.end()
        except Exception as e:
            logging.warning(
                f"[MaximSDK][GeminiAsyncModels] Error in logging generation: {str(e)}"
            )
        # Actual response
        return chunks

    @override
    async def generate_content(
        self,
        *,
        model: str,
        contents: Union[ContentListUnion, ContentListUnionDict],
        config: Optional[GenerateContentConfigOrDict] = None,
        trace_id: Optional[str] = None,
        generation_name: Optional[str] = None,
    ) -> GenerateContentResponse:
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
            logging.warning(
                f"[MaximSDK][GeminiAsyncModels] Error in generating content: {str(e)}"
            )

        # Actual call will never fail
        response = await self._models.generate_content(
            model=model, contents=contents, config=config
        )
        # Actual call ends
        try:
            if generation is not None:
                generation.result(response)
            if is_local_trace:
                if trace is not None:
                    if response is not None:
                        trace.set_output(response.text or "")
                    trace.end()
        except Exception as e:
            logging.warning(
                f"[MaximSDK][GeminiAsyncModels] Error in logging generation: {str(e)}"
            )
        # Actual response
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_models":
            super().__setattr__(name, value)
        else:
            setattr(self._models, name, value)


class MaximGeminiAsyncClient(AsyncClient):
    def __init__(self, client: AsyncClient, logger: Logger):
        self._client = client
        self._logger = logger
        self._w_models = MaximGeminiAsyncModels(client.models, logger)
        self._w_chats = MaximGeminiAsyncChats(client.chats, logger)

    def __getattr__(self, name: str) -> Any:
        if name == "_models":
            return self._w_models
        elif name == "_chats":
            return self._w_chats
        result = getattr(self._client, name)
        return result

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_client":
            super().__setattr__(name, value)
        else:
            setattr(self._client, name, value)
