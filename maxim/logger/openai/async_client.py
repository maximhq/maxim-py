from openai import AsyncOpenAI

from ..logger import (
    Logger,
)
from .async_chat import MaximAsyncOpenAIChat


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
