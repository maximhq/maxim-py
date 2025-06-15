from typing import Any

from ..logger import (
    Logger,
)
from anthropic import Anthropic
from .message import MaximAnthropicMessages


class MaximAnthropicClient:

    def __init__(self, client: Anthropic, logger: Logger):
        self._client = client
        self._logger = logger

    @property
    def messages(self) -> MaximAnthropicMessages:
        return MaximAnthropicMessages(self._client, self._logger)
