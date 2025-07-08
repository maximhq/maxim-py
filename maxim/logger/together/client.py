from together import Together
from together.resources.chat import Chat

from ..logger import Logger
from .completions import MaximTogetherCompletions
from .chat import MaximTogetherChat

# This is a wrapper for the Together chat interface that only intercepts the completions method for logging while keeping other chat methods as-is.
class MaximTogetherChatWrapper:
    def __init__(self, chat: Chat, logger: Logger):
        """Wrapper for Together chat interface that only intercepts completions.
        
        Args:
            chat_interface (Chat): The original Together chat interface
            logger (Logger): The Maxim logger instance
        """
        self._chat = chat
        self._logger = logger
    
    @property
    def completions(self) -> MaximTogetherChat:
        """Get the completions interface with Maxim logging capabilities."""
        return MaximTogetherChat(self._chat.completions, self._logger)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original chat interface."""
        return getattr(self._chat, name)


class MaximTogetherClient:
    def __init__(self, client: Together, logger: Logger):
        """Initialize the Maxim Together client.

        Args:
            client (Together): The Together client instance to wrap.
            logger (Logger): The Maxim logger instance for tracking and
                logging API interactions.
        """
        self._client = client
        self._logger = logger
    
    @property
    def chat(self) -> MaximTogetherChatWrapper:
        """Get the chat interface with selective Maxim logging capabilities.

        Returns:
            MaximTogetherChatWrapper: A wrapped chat interface that only intercepts
                the completions method for logging while keeping other chat methods as-is.
        """
        return MaximTogetherChatWrapper(self._client.chat, self._logger)
        
    @property
    def completions(self) -> MaximTogetherCompletions:
        """Get the completions interface with Maxim logging capabilities.

        Returns:
            MaximTogetherCompletions: A wrapped completions interface that provides
                logging and monitoring capabilities for Together completion operations.
        """
        return MaximTogetherCompletions(self._client, self._logger)

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying Together client.
        
        This allows pass-through access to other Together client methods
        like embeddings, audio, etc. without adding Maxim logging.
        """
        return getattr(self._client, name)
        