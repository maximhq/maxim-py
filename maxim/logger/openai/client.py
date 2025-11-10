from openai import AsyncOpenAI, OpenAI

from ..logger import (
    Logger,
)
from .async_client import MaximOpenAIAsyncClient
from .chat import MaximOpenAIChat
from .responses import MaximOpenAIResponses
from .realtime import MaximOpenAIRealtime


class MaximOpenAIClient:
    def __init__(self, client: OpenAI, logger: Logger):
        self._client = client
        self._logger = logger
        self._aio = MaximOpenAIAsyncClient(AsyncOpenAI(api_key=client.api_key), logger)

    @property
    def chat(self) -> MaximOpenAIChat:
        return MaximOpenAIChat(self._client, self._logger)

    @property
    def aio(self) -> MaximOpenAIAsyncClient:
        return self._aio

    @property
    def responses(self) -> MaximOpenAIResponses:
        return MaximOpenAIResponses(self._client, self._logger)

    @property
    def realtime(self) -> MaximOpenAIRealtime:
        """
        This property represents the realtime object of MaximOpenAIClient.
        """

        return MaximOpenAIRealtime(self._client, self._logger)
