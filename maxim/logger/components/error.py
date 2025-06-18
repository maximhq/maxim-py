import logging
from typing import Any, Dict, Optional, TypedDict

from ..writer import LogWriter
from .base import BaseContainer, Entity

logger = logging.getLogger("MaximSDK")


class ErrorConfig(TypedDict, total=False):
    """Error config.

    This class provides functionality to manage error configurations.
    """

    id: Optional[str]
    message: str
    name: Optional[str]
    code: Optional[str]
    type: Optional[str]
    tags: Optional[Dict[str, str]]
    metadata: Optional[Dict[str, Any]]


class Error(BaseContainer):
    """Error.

    This class represents an error event.
    """

    def __init__(self, config: ErrorConfig, writer: LogWriter):
        """Initialize the error.

        Args:
            config (ErrorConfig): The config of the error.
            writer (LogWriter): The writer of the error.
        """
        final_config = dict(config)
        super().__init__(Entity.ERROR, final_config, writer)
        self.message = final_config.get("message", None)
        self.code = final_config.get("code", None)
        self.error_type = final_config.get("type", None)
        self.tags = final_config.get("tags", None)
        self.metadata = final_config.get("metadata", None)

    def data(self) -> Dict[str, Any]:
        """Get the data of the error.

        Returns:
            Dict[str, Any]: The data of the error.
        """
        base_data = super().data()
        return {
            **base_data,
            "message": self.message,
            "code": self.code,
            "errorType": self.error_type,
            "metadata": self.metadata,
        }
