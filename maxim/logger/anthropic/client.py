from typing import Any

from ..logger import (
    Logger,
)
from anthropic import Anthropic
from .message import MaximAnthropicMessages


class MaximAnthropicClient:
    """Maxim Anthropic client wrapper.

    This class provides a wrapper around the Anthropic client to integrate
    with Maxim's logging and monitoring capabilities. It allows tracking
    and logging of Anthropic API interactions through the Maxim platform.

    Attributes:
        _client (Anthropic): The underlying Anthropic client instance.
        _logger (Logger): The Maxim logger instance for tracking interactions.
    """

    def __init__(self, client: Anthropic, logger: Logger):
        """Initialize the Maxim Anthropic client.

        Args:
            client (Anthropic): The Anthropic client instance to wrap.
            logger (Logger): The Maxim logger instance for tracking and
                logging API interactions.
        """
        self._client = client
        self._logger = logger

    @property
    def messages(self) -> MaximAnthropicMessages:
        """Get the messages interface with Maxim logging capabilities.

        Returns:
            MaximAnthropicMessages: A wrapped messages interface that provides
                logging and monitoring capabilities for Anthropic message operations.
        """
        return MaximAnthropicMessages(self._client, self._logger)
