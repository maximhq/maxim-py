from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat

from ..logger import (
    Logger,
)
from .async_completions import MaximAsyncOpenAIChatCompletions


class MaximAsyncOpenAIChat(AsyncChat):
    """
    This class represents a MaximAsyncOpenAIChat.
    """

    def __init__(self, client: AsyncOpenAI, logger: Logger):
        """
        This class represents a MaximAsyncOpenAIChat.

        Args:
            client: The client to use.
            logger: The logger to use.
        """
        super().__init__(client=client)
        self._logger = logger

    @property
    def completions(self) -> MaximAsyncOpenAIChatCompletions:
        """
        This property represents the completions object of MaximAsyncOpenAIChat.
        """

        return MaximAsyncOpenAIChatCompletions(self._client, self._logger)
