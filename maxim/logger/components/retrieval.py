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
class RetrievalConfig():
    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class RetrievalConfigDict(TypedDict, total=False):
    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]


def get_retrieval_config_dict(
    config: Union[RetrievalConfig, RetrievalConfigDict],
) -> dict[str, Any]:
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
    def __init__(
        self, config: Union[RetrievalConfig, RetrievalConfigDict], writer: LogWriter
    ):
        final_config = get_retrieval_config_dict(config)
        super().__init__(Entity.RETRIEVAL, dict(final_config), writer)
        self.is_output_set = False

    def input(self, query: str):
        if query is None:
            return
        self._commit("update", {"input": query})
        self.end()

    @staticmethod
    def input_(writer: LogWriter, id: str, query: str):
        BaseContainer._commit_(writer, Entity.RETRIEVAL,
                               id, "update", {"input": query})

    def output(self, docs: Union[str, List[str]]):
        final_docs = docs if isinstance(docs, list) else [docs]
        self.is_output_set = True
        self._commit(
            "update", {"docs": final_docs, "endTimestamp": datetime.now(timezone.utc)})
        self.end()

    def add_attachment(
        self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]
    ):
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(
        writer: LogWriter,
        id: str,
        attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment],
    ):
        BaseContainer._commit_(
            writer, Entity.RETRIEVAL, id, "upload-attachment", attachment.to_dict()
        )

    @staticmethod
    def output_(writer: LogWriter, id: str, docs: Union[str, List[str]]):
        final_docs = docs if isinstance(docs, list) else [docs]
        BaseContainer._commit_(writer, Entity.RETRIEVAL, id, "update", {
                               "docs": final_docs})
        BaseContainer._end_(writer, Entity.RETRIEVAL, id, {
                            "endTimestamp": datetime.now(timezone.utc)})

    @staticmethod
    def end_(writer: LogWriter, id: str, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = {}
        BaseContainer._end_(writer, Entity.RETRIEVAL, id, {
            "endTimestamp": datetime.now(timezone.utc),
            **data,
        })

    @staticmethod
    def add_tag_(writer: LogWriter, id: str, key: str, value: str):
        BaseContainer._add_tag_(writer, Entity.RETRIEVAL, id, key, value)
