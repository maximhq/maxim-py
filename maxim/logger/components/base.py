import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...scribe import scribe
from ..parsers.tags_parser import parse_tags
from ..utils import make_object_serializable
from ..writer import LogWriter
from .types import CommitLog, Entity


class ContainerLister:
    def on_end(self):
        pass


BaseConfig = Dict[str, Any]


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    sanitized_metadata: dict[str, str] = {}
    for key, value in metadata.items():
        serialized_obj = make_object_serializable(value)
        if isinstance(serialized_obj, str):
            sanitized_metadata[key] = serialized_obj
            continue
        sanitized_metadata[key] = json.dumps(serialized_obj)
    return sanitized_metadata


class EvaluateContainerWithVariables:
    def __init__(
        self, id: str, entity: Entity, log_writer: LogWriter, for_evaluators: List[str]
    ) -> None:
        self.entity = entity
        self.writer = log_writer
        self.id = id
        self.for_evaluators = for_evaluators

    def with_variables(self, variables: Dict[str, str]):
        if len(self.for_evaluators) == 0:
            return
        self.writer.commit(
            CommitLog(
                self.entity,
                self.id,
                "evaluate",
                {
                    "with": "variables",
                    "variables": variables,
                    "evaluators": list(set(self.for_evaluators)),
                    "timestamp": datetime.now(timezone.utc),
                },
            )
        )


class EvaluateContainer:
    """
    A class to manage evaluators for a specific entity.

    This class provides functionality to initialize and manage a set of evaluators
    associated with a particular entity and writer.

    Attributes:
        entity (Entity): The entity associated with these evaluators.
        writer (LogWriter): The log writer used for committing evaluator actions.
        evaluators (List[str]): A list of evaluator identifiers.
        id (str): A unique identifier for this set of evaluators.

    Methods:
        with_variables: Allows adding variables to be used by the evaluators.
    """

    def __init__(self, id: str, entity: Entity, log_writer: LogWriter) -> None:
        self.entity = entity
        self.writer = log_writer
        self.id = id

    def with_variables(self, variables: Dict[str, str], for_evaluators: List[str]):
        if len(for_evaluators) == 0:
            raise ValueError("At least one evaluator must be provided")

        self.writer.commit(
            CommitLog(
                self.entity,
                self.id,
                "evaluate",
                {
                    "with": "variables",
                    "variables": variables,
                    "evaluators": list(set(for_evaluators)),
                    "timestamp": datetime.now(timezone.utc),
                },
            )
        )

    def with_evaluators(self, *evaluators: str) -> EvaluateContainerWithVariables:
        if len(evaluators) == 0:
            raise ValueError("At least one evaluator must be provided")

        self.writer.commit(
            CommitLog(
                self.entity,
                self.id,
                "evaluate",
                {
                    "with": "evaluators",
                    "evaluators": list(set(evaluators)),
                    "timestamp": datetime.now(timezone.utc),
                },
            )
        )

        return EvaluateContainerWithVariables(
            self.id, self.entity, self.writer, list(set(evaluators))
        )


class BaseContainer:
    def __init__(self, entity: Entity, config: BaseConfig, writer: LogWriter):
        self.entity = entity
        if "id" not in config:
            self._id = str(uuid.uuid4())
        else:
            self._id = config["id"]
        self._name = config.get("name", None)
        self.span_id = config.get("span_id", None)
        self.start_timestamp = datetime.now(timezone.utc)
        self.end_timestamp = None
        self.tags = parse_tags(config.get("tags", {}))
        self.writer = writer
        # Doing it at the end to avoid problems with regular flow
        # We drop these logs at LogWriter level as well
        # Validate ID format - only allow alphanumeric characters, hyphens, and underscores
        if not re.match(r"^[a-zA-Z0-9_-]+$", self._id):
            if writer.raise_exceptions:
                raise ValueError(
                    f"Invalid ID: {self._id}. ID must only contain alphanumeric characters, hyphens, and underscores. Event will not be logged."
                )
            else:
                scribe().error(
                    f"[MaximSDK] Invalid ID: {config['id']}. ID must only contain alphanumeric characters, hyphens, and underscores. Event will not be logged."
                )

    @property
    def id(self) -> str:
        return self._id

    def evaluate(self) -> EvaluateContainer:
        return EvaluateContainer(self._id, self.entity, self.writer)

    @staticmethod
    def _evaluate_(writer: LogWriter, entity: Entity, id: str) -> EvaluateContainer:
        return EvaluateContainer(id, entity, writer)

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
        self._commit("update", {"metadata": sanitized_metadata})

    @staticmethod
    def add_metadata_(
        writer: LogWriter, entity: Entity, id: str, metadata: Dict[str, Any]
    ) -> None:
        sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
        writer.commit(CommitLog(entity, id, "update", {"metadata": sanitized_metadata}))

    def add_tag(self, key: str, value: str):
        if self.tags is None:
            self.tags = {}
        if not isinstance(value, str):
            raise ValueError("Tag value must be a string")
        # Validate if value is str and not None
        if not value:
            raise ValueError("Tag value must not be empty")
        self.tags[key] = value
        self.tags = parse_tags(self.tags)
        self._commit("update", {"tags": {key: value}})

    @staticmethod
    def _add_tag_(writer: LogWriter, entity: Entity, id: str, key: str, value: str):
        writer.commit(CommitLog(entity, id, "update", {"tags": {key: value}}))

    def end(self):
        self.end_timestamp = datetime.now(timezone.utc)
        self._commit("end", {"endTimestamp": self.end_timestamp})

    @staticmethod
    def _end_(
        writer: LogWriter,
        entity: Entity,
        id: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        if data is None:
            data = {}
        data = {k: v for k, v in data.items() if v is not None}
        writer.commit(CommitLog(entity, id, "end", data))

    def data(self) -> Dict[str, Any]:
        data = {
            "name": self._name,
            "spanId": self.span_id,
            "tags": self.tags,
            "startTimestamp": self.start_timestamp,
            "endTimestamp": self.end_timestamp,
        }
        # removing none values
        data = {k: v for k, v in data.items() if v is not None}
        return data

    @staticmethod
    def _commit_(
        writer: LogWriter,
        entity: Entity,
        id: str,
        action: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        # Removing all null values from data dict
        if data is not None:
            data = {k: v for k, v in data.items() if v is not None}
        writer.commit(CommitLog(entity, id, action, data))

    def _commit(self, action: str, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = self.data()
        # Removing all null values from data dict
        data = {k: v for k, v in data.items() if v is not None}
        self.writer.commit(CommitLog(self.entity, self._id, action, data))


class EventEmittingBaseContainer(BaseContainer):
    @staticmethod
    def _event_(
        writer: LogWriter,
        entity: Entity,
        entity_id: str,
        id: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if metadata is not None:
            sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
            BaseContainer._commit_(
                writer,
                entity,
                entity_id,
                "add-event",
                {
                    "id": id,
                    "name": name,
                    "timestamp": datetime.now(timezone.utc),
                    "tags": tags,
                    "metadata": sanitized_metadata,
                },
            )
        else:
            BaseContainer._commit_(
                writer,
                entity,
                entity_id,
                "add-event",
                {
                    "id": id,
                    "name": name,
                    "timestamp": datetime.now(timezone.utc),
                    "tags": tags,
                },
            )

    def event(
        self,
        id: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if metadata is not None:
            sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
            self._commit(
                "add-event",
                {
                    "id": id,
                    "name": name,
                    "timestamp": datetime.now(timezone.utc),
                    "tags": tags,
                    "metadata": sanitized_metadata,
                },
            )
        else:
            self._commit(
                "add-event",
                {
                    "id": id,
                    "name": name,
                    "timestamp": datetime.now(timezone.utc),
                    "tags": tags,
                },
            )
