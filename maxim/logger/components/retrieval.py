from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..components import FileAttachment, FileDataAttachment, UrlAttachment
from ..writer import LogWriter
from .base import BaseContainer
from .types import Entity


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class RetrievalConfig:
    """Retrieval config.

    This class provides functionality to manage retrieval configurations.
    """

    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class RetrievalConfigDict(TypedDict, total=False):
    """Retrieval config dict.

    This class provides functionality to manage retrieval config dictionaries.
    """

    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]


def get_retrieval_config_dict(
    config: Union[RetrievalConfig, RetrievalConfigDict],
) -> dict[str, Any]:
    """Get the retrieval config dict.

    Args:
        config (Union[RetrievalConfig, RetrievalConfigDict]): The config to get the dict from.

    Returns:
        dict[str, Any]: The retrieval config dict.
    """
    return (
        dict(
            RetrievalConfigDict(
                id=config.id,
                name=config.name,
                tags=config.tags,
            )
        )
        if isinstance(config, RetrievalConfig)
        else dict(config)
    )


class Retrieval(BaseContainer):
    """Retrieval.

    This class represents a retrieval.
    """

    def __init__(
        self, config: Union[RetrievalConfig, RetrievalConfigDict], writer: LogWriter
    ):
        """
        Initialize a retrieval.

        Args:
            config: The config to initialize the retrieval with.
            writer: The writer to use.
        """
        final_config = get_retrieval_config_dict(config)
        super().__init__(Entity.RETRIEVAL, dict(final_config), writer)
        self.is_output_set = False

    def input(self, query: str):
        """
        Set the input for the retrieval.

        Args:
            query: The query to set for the retrieval.
        """
        if query is None:
            return
        self._commit("update", {"input": query})
        self.end()

    @staticmethod
    def input_(writer: LogWriter, id: str, query: str):
        """
        Set the input for the retrieval.

        Args:
            writer: The writer to use.
            id: The id of the retrieval.
            query: The query to set for the retrieval.
        """
        BaseContainer._commit_(writer, Entity.RETRIEVAL, id, "update", {"input": query})

    def output(self, docs: Union[str, List[str]]):
        """
        Set the output for the retrieval.

        Args:
            docs: The docs to set for the retrieval.
        """
        final_docs = docs if isinstance(docs, list) else [docs]
        self.is_output_set = True
        self._commit(
            "update", {"docs": final_docs, "endTimestamp": datetime.now(timezone.utc)}
        )
        self.end()

    def add_metric(self, name: str, value: float) -> None:
        """
        Add a metric to this retrieval.
        """
        self._commit("update", {"metrics": {"name": name, "value": value}})

    @staticmethod
    def add_metric_(writer: LogWriter, id: str, name: str, value: float):
        """
        Add a metric to the retrieval.
        """
        BaseContainer._commit_(
            writer,
            Entity.RETRIEVAL,
            id,
            "update",
            {"metrics": {"name": name, "value": value}},
        )

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        """
        Add an attachment to the retrieval.

        Args:
            attachment: The attachment to add to the retrieval.
        """
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(
        writer: LogWriter,
        id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        """
        Add an attachment to the retrieval.

        Args:
            writer: The writer to use.
            id: The id of the retrieval.
            attachment: The attachment to add to the retrieval.
        """
        BaseContainer._commit_(
            writer, Entity.RETRIEVAL, id, "upload-attachment", attachment.to_dict()
        )

    @staticmethod
    def output_(writer: LogWriter, id: str, docs: Union[str, List[str]]):
        """
        Set the output for the retrieval.

        Args:
            writer: The writer to use.
            id: The id of the retrieval.
            docs: The docs to set for the retrieval.
        """
        final_docs = docs if isinstance(docs, list) else [docs]
        BaseContainer._commit_(
            writer, Entity.RETRIEVAL, id, "update", {"docs": final_docs}
        )
        BaseContainer._end_(
            writer, Entity.RETRIEVAL, id, {"endTimestamp": datetime.now(timezone.utc)}
        )

    @staticmethod
    def end_(writer: LogWriter, id: str, data: Optional[Dict[str, Any]] = None):
        """
        End the retrieval.

        Args:
            writer: The writer to use.
            id: The id of the retrieval.
            data: The data to set for the retrieval.
        """
        if data is None:
            data = {}
        BaseContainer._end_(
            writer,
            Entity.RETRIEVAL,
            id,
            {
                "endTimestamp": datetime.now(timezone.utc),
                **data,
            },
        )

    @staticmethod
    def add_tag_(writer: LogWriter, id: str, key: str, value: str):
        """
        Add a tag to the retrieval.

        Args:
            writer: The writer to use.
            id: The id of the retrieval.
            key: The key of the tag.
            value: The value of the tag.
        """
        BaseContainer._add_tag_(writer, Entity.RETRIEVAL, id, key, value)
