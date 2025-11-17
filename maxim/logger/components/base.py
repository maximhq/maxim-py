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
    """Sanitize the metadata.

    Args:
        metadata (Dict[str, Any]): The metadata to sanitize.

    Returns:
        Dict[str, str]: The sanitized metadata.
    """
    sanitized_metadata: dict[str, str] = {}
    for key, value in metadata.items():
        serialized_obj = make_object_serializable(value)
        if isinstance(serialized_obj, str):
            sanitized_metadata[key] = serialized_obj
            continue
        sanitized_metadata[key] = json.dumps(serialized_obj)
    return sanitized_metadata


class EvaluateContainerWithVariables:
    """Evaluate container with variables.

    This class provides functionality to manage evaluators with variables.
    """

    def __init__(
        self, id: str, entity: Entity, log_writer: LogWriter, for_evaluators: List[str]
    ) -> None:
        """Initialize the evaluate container with variables.

        Args:
            id (str): The ID of the evaluate container.
            entity (Entity): The entity of the evaluate container.
            log_writer (LogWriter): The log writer of the evaluate container.
            for_evaluators (List[str]): The evaluators of the evaluate container.
        """
        self.entity = entity
        self.writer = log_writer
        self.id = id
        self.for_evaluators = for_evaluators

    def with_variables(self, variables: Dict[str, str]):
        """With variables.

        Args:
            variables (Dict[str, str]): The variables to use for the evaluate container.
        """
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
    """Evaluate container.

    This class provides functionality to manage evaluators for a specific entity.

    Attributes:
        entity (Entity): The entity associated with these evaluators.
        writer (LogWriter): The log writer used for committing evaluator actions.
        evaluators (List[str]): A list of evaluator identifiers.
        id (str): A unique identifier for this set of evaluators.

    Methods:
        with_variables: Allows adding variables to be used by the evaluators.
    """

    def __init__(self, id: str, entity: Entity, log_writer: LogWriter) -> None:
        """Initialize the evaluate container.

        Args:
            id (str): The ID of the evaluate container.
            entity (Entity): The entity of the evaluate container.
            log_writer (LogWriter): The log writer of the evaluate container.
        """
        self.entity = entity
        self.writer = log_writer
        self.id = id

    def with_variables(self, variables: Dict[str, str], for_evaluators: List[str]):
        """With variables.

        Args:
            variables (Dict[str, str]): The variables to use for the evaluate container.
            for_evaluators (List[str]): The evaluators of the evaluate container.
        """
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
        """With evaluators.

        Args:
            *evaluators (str): The evaluators to use for the evaluate container.

        Returns:
            EvaluateContainerWithVariables: The evaluate container with variables.
        """
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
    """Base container.

    This class provides functionality to manage containers for a specific entity.
    """

    def __init__(self, entity: Entity, config: BaseConfig, writer: LogWriter):
        """Initialize the base container.

        Args:
            entity (Entity): The entity of the base container.
            config (BaseConfig): The config of the base container.
            writer (LogWriter): The writer of the base container.
        """
        self.entity = entity
        if "id" not in config:
            self._id = str(uuid.uuid4())
        else:
            self._id = config["id"]
        self._name = config.get("name", None)
        self.span_id = config.get("span_id", None)
        if config.get("start_timestamp", None) is not None:
            ts = config["start_timestamp"]
            if ts is not None:
                if not isinstance(ts, datetime):
                    scribe().warning(
                        f"[MaximSDK] Invalid start timestamp: {ts} for {self.entity.value}. Reverting to current time."
                    )
                    self.start_timestamp = datetime.now(timezone.utc)
                else:
                    self.start_timestamp = ts
        else:
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
        """Get the ID of the base container.

        Returns:
            str: The ID of the base container.
        """
        return self._id

    def evaluate(self) -> EvaluateContainer:
        """Evaluate the base container.

        Returns:
            EvaluateContainer: The evaluate container.
        """
        return EvaluateContainer(self._id, self.entity, self.writer)

    @staticmethod
    def _evaluate_(writer: LogWriter, entity: Entity, id: str) -> EvaluateContainer:
        """Evaluate the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            id (str): The ID of the base container.
        """
        return EvaluateContainer(id, entity, writer)

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the base container.

        Args:
            metadata (Dict[str, Any]): The metadata to add to the base container.
        """
        sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
        self._commit("update", {"metadata": sanitized_metadata})

    @staticmethod
    def add_metadata_(
        writer: LogWriter, entity: Entity, id: str, metadata: Dict[str, Any]
    ) -> None:
        """Add metadata to the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            id (str): The ID of the base container.
            metadata (Dict[str, Any]): The metadata to add to the base container.
        """
        sanitized_metadata: dict[str, str] = _sanitize_metadata(metadata)
        writer.commit(CommitLog(entity, id, "update", {"metadata": sanitized_metadata}))

    def add_tag(self, key: str, value: str):
        """Add a tag to the base container.

        Args:
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
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
        """Add a tag to the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            id (str): The ID of the base container.
            key (str): The key of the tag.
            value (str): The value of the tag.
        """
        writer.commit(CommitLog(entity, id, "update", {"tags": {key: value}}))

    def set_start_timestamp(self, timestamp: datetime):
        """Set the start timestamp for this base container.

        Args:
            timestamp (datetime): The start timestamp to set.
        """
        if not isinstance(timestamp, datetime):
            scribe().warning(
                f"[MaximSDK] Invalid start timestamp: {timestamp} for {self.entity.value}. Timestamp must be a datetime object."
            )
            return
        self.start_timestamp = timestamp
        self._commit("update", {"startTimestamp": timestamp})

    def set_end_timestamp(self, timestamp: datetime):
        """Set the end timestamp for this base container.

        Args:
            timestamp (datetime): The end timestamp to set.
        """
        if not isinstance(timestamp, datetime):
            scribe().warning(
                f"[MaximSDK] Invalid end timestamp: {timestamp} for {self.entity.value}. Timestamp must be a datetime object."
            )
            return
        self.end_timestamp = timestamp
        self._commit("update", {"endTimestamp": timestamp})

    def end(self):
        """End the base container.

        This method is used to end the base container.
        """
        self.end_timestamp = datetime.now(timezone.utc)
        self._commit("end", {"endTimestamp": self.end_timestamp})

    @staticmethod
    def _end_(
        writer: LogWriter,
        entity: Entity,
        id: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """End the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            id (str): The ID of the base container.
            data (Optional[Dict[str, Any]]): The data to add to the base container.
        """
        if data is None:
            data = {}
        data = {k: v for k, v in data.items() if v is not None}
        writer.commit(CommitLog(entity, id, "end", data))

    def data(self) -> Dict[str, Any]:
        """Get the data of the base container.

        Returns:
            Dict[str, Any]: The data of the base container.
        """
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
        """Commit the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            id (str): The ID of the base container.
            action (str): The action to commit.
            data (Optional[Dict[str, Any]]): The data to commit.
        """
        # Removing all null values from data dict
        if data is not None:
            data = {k: v for k, v in data.items() if v is not None}
        writer.commit(CommitLog(entity, id, action, data))

    def _commit(self, action: str, data: Optional[Dict[str, Any]] = None):
        """Commit the base container.

        Args:
            action (str): The action to commit.
            data (Optional[Dict[str, Any]]): The data to commit.
        """
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
        """Add an event to the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            entity_id (str): The ID of the entity.
            id (str): The ID of the event.
            name (str): The name of the event.
            tags (Optional[Dict[str, str]]): The tags of the event.
            metadata (Optional[Dict[str, Any]]): The metadata of the event.
        """
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
        """Add an event to the base container.

        Args:
            id (str): The ID of the event.
            name (str): The name of the event.
            tags (Optional[Dict[str, str]]): The tags of the event.
            metadata (Optional[Dict[str, Any]]): The metadata of the event.
        """
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

    @staticmethod
    def _set_start_timestamp_(
        writer: LogWriter,
        entity: Entity,
        entity_id: str,
        timestamp: datetime,
    ):
        """Set the start timestamp for the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            entity_id (str): The ID of the entity.
            timestamp (datetime): The start timestamp to set.
        """
        if not isinstance(timestamp, datetime):
            scribe().warning(
                f"[MaximSDK] Invalid start timestamp: {timestamp} for {entity.value}. Timestamp must be a datetime object."
            )
            return
        BaseContainer._commit_(writer, entity, entity_id, "update", {"startTimestamp": timestamp})

    @staticmethod
    def _set_end_timestamp_(
        writer: LogWriter,
        entity: Entity,
        entity_id: str,
        timestamp: datetime,
    ):
        """Set the end timestamp for the base container.

        Args:
            writer (LogWriter): The writer of the base container.
            entity (Entity): The entity of the base container.
            entity_id (str): The ID of the entity.
            timestamp (datetime): The end timestamp to set.
        """
        if not isinstance(timestamp, datetime):
            scribe().warning(
                f"[MaximSDK] Invalid end timestamp: {timestamp} for {entity.value}. Timestamp must be a datetime object."
            )
            return
        BaseContainer._commit_(writer, entity, entity_id, "update", {"endTimestamp": timestamp})
