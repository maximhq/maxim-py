from openai import OpenAI
from openai.resources.chat import Chat

from ..logger import (
    Logger,
)
from .completions import MaximOpenAIChatCompletions


class MaximOpenAIChat(Chat):
    """
    This class represents a MaximOpenAIChat.
    """

    def __init__(self, client: OpenAI, logger: Logger):
        """
        This class represents a MaximOpenAIChat.

        Args:
            client: The client to use.
            logger: The logger to use.
        """
        super().__init__(client=client)
        self._logger = logger

    @property
    def completions(self) -> MaximOpenAIChatCompletions:
        """
        This property represents the completions object of MaximOpenAIChat.
        """

        return MaximOpenAIChatCompletions(self._client, self._logger)
