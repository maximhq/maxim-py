from openai import AsyncOpenAI

from ..logger import (
    Logger,
)
from .async_chat import MaximAsyncOpenAIChat
from .async_responses import MaximAsyncOpenAIResponses
from .async_realtime import MaximOpenAIAsyncRealtime


class MaximOpenAIAsyncClient:
    """
    This class represents a MaximOpenAIAsyncClient.
    """

    def __init__(self, client: AsyncOpenAI, logger: Logger):
        """
        This class represents a MaximOpenAIAsyncClient.

        Args:
            client: The client to use.
            logger: The logger to use.
        """
        self._client = client
        self._logger = logger

    @property
    def chat(self) -> MaximAsyncOpenAIChat:
        """
        This property represents the chat object of MaximOpenAIAsyncClient.
        """

        return MaximAsyncOpenAIChat(self._client, self._logger)

    @property
    def responses(self) -> MaximAsyncOpenAIResponses:
        """
        This property represents the responses object of MaximOpenAIAsyncClient.
        """

        return MaximAsyncOpenAIResponses(self._client, self._logger)

    @property
    def realtime(self) -> MaximOpenAIAsyncRealtime:
        """
        This property represents the realtime object of MaximOpenAIAsyncClient.
        """

        return MaximOpenAIAsyncRealtime(self._client, self._logger)