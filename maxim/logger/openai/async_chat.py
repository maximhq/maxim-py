from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat

from ..logger import (
    Logger,
)
from .async_completions import MaximAsyncOpenAIChatCompletions


class MaximAsyncOpenAIChat(AsyncChat):
    def __init__(self, client: AsyncOpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    @property
    def completions(self) -> MaximAsyncOpenAIChatCompletions:
        return MaximAsyncOpenAIChatCompletions(self._client, self._logger)
