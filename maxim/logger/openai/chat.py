from openai import OpenAI
from openai.resources.chat import Chat

from ..logger import (
    Logger,
)
from .completions import MaximOpenAIChatCompletions


class MaximOpenAIChat(Chat):
    def __init__(self, client: OpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    @property
    def completions(self) -> MaximOpenAIChatCompletions:
        return MaximOpenAIChatCompletions(self._client, self._logger)
