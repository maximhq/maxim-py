from openai import AsyncOpenAI

from ..logger import (
    Logger,
)
from .async_chat import MaximAsyncOpenAIChat


class MaximOpenAIAsyncClient:
    def __init__(self, client: AsyncOpenAI, logger: Logger):
        self._client = client
        self._logger = logger

    @property
    def chat(self) -> MaximAsyncOpenAIChat:
        return MaximAsyncOpenAIChat(self._client, self._logger)
