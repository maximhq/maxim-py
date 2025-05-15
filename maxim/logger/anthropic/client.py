from typing import Any

from anthropic import Anthropic

from ..logger import (
    Logger,
)
from .message import MaximAnthropicMessages


class MaximAnthropicClient:
    def __init__(self, client: Anthropic, logger: Logger):
        self._client = client
        self._logger = logger

    @property
    def messages(self) -> Any:
        return MaximAnthropicMessages(self._client, self._logger)
