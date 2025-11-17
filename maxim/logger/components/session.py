from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..writer import LogWriter
from .attachment import (
    FileAttachment,
    FileDataAttachment,
    UrlAttachment,
)
from .base import EventEmittingBaseContainer
from .feedback import Feedback, FeedbackDict, get_feedback_dict
from .trace import Trace, TraceConfig, TraceConfigDict, get_trace_config_dict
from .types import Entity


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class SessionConfig:
    """Session config.

    This class represents a session config.
    """

    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    start_timestamp: Optional[datetime] = None

class SessionConfigDict(TypedDict, total=False):
    """Session config dict.

    This class represents a session config dictionary.
    """

    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]
    start_timestamp: Optional[datetime]

def get_session_config_dict(
    config: Union[SessionConfig, SessionConfigDict],
) -> dict[str, Any]:
    """Convert a session config to a session config dict else return the config.

    Args:
        config: The config to convert.

    Returns:
        dict[str, Any]: The session config dict.
    """
    return (
        dict(
            SessionConfigDict(
                id=config.id,
                name=config.name,
                tags=config.tags,
                start_timestamp=config.start_timestamp,
            )
        )
        if isinstance(config, SessionConfig)
        else dict(config)
    )


class Session(EventEmittingBaseContainer):
    """
    A session is a collection of traces.

    A session is created when a new session is started.

    A session is ended when the session is stopped.    
    """
    ENTITY = Entity.SESSION

    def __init__(
        self, config: Union[SessionConfig, SessionConfigDict], writer: LogWriter
    ):
        """
        Create a new session.

        Args:
            config: The configuration for the session.
        """
        final_config = get_session_config_dict(config)
        super().__init__(Session.ENTITY, dict(final_config), writer)
        self._commit("create")

    def trace(self, config: Union[TraceConfig, TraceConfigDict]) -> Trace:
        """
        Create a new trace for this session.

        Args:
            config: The configuration for the trace.

        Returns:
            A new Trace instance.
        """
        final_config = get_trace_config_dict(config)
        final_config["session_id"] = self.id
        return Trace(final_config, self.writer)

    @staticmethod
    def trace_(
        writer: LogWriter, session_id: str, config: Union[TraceConfig, TraceConfigDict]
    ) -> Trace:
        """
        Create a new trace for this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to create the trace for.
            config: The configuration for the trace.

        Returns:
            A new Trace instance.
        """
        final_config = get_trace_config_dict(config)
        final_config["session_id"] = session_id
        return Trace(final_config, writer)

    def feedback(self, feedback: Union[Feedback, FeedbackDict]):
        """
        Add feedback to this session.

        Args:
            feedback: The feedback to add.
        """
        self._commit("add-feedback", dict(get_feedback_dict(feedback)))

    def add_attachment(self, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]):
        """
        Add an attachment to this session.

        Args:
            attachment: The attachment to add.
        """
        self._commit("upload-attachment", attachment.to_dict())

    @staticmethod
    def add_attachment_(writer: LogWriter, session_id: str, attachment: Union[FileAttachment, FileDataAttachment, UrlAttachment]):
        """
        Add an attachment to this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to add the attachment to.
            attachment: The attachment to add.
        """
        EventEmittingBaseContainer._commit_(
            writer,
            Entity.SESSION,
            session_id,
            "upload-attachment",
            attachment.to_dict(),
        )

    @staticmethod
    def feedback_(
        writer: LogWriter, session_id: str, feedback: Union[Feedback, FeedbackDict]
    ):
        EventEmittingBaseContainer._commit_(
            writer,
            Entity.SESSION,
            session_id,
            "add-feedback",
            dict(get_feedback_dict(feedback)),
        )

    @staticmethod
    def add_tag_(writer: LogWriter, session_id: str, key: str, value: str):
        """
        Add a tag to this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to add the tag to.
            key: The tag key.
            value: The tag value.
        """
        return EventEmittingBaseContainer._add_tag_(writer, Entity.SESSION, session_id, key, value)

    @staticmethod
    def end_(writer: LogWriter, session_id: str, data: Optional[Dict[str, Any]] = None):
        """
        End this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to end.
            data: Optional data to add to the session.
        """
        if data is None:
            data = {}
        # Only set endTimestamp if it's not already provided in data
        if "endTimestamp" not in data:
            data["endTimestamp"] = datetime.now(timezone.utc)
        return EventEmittingBaseContainer._end_(writer, Entity.SESSION, session_id, data)

    @staticmethod
    def event_(writer: LogWriter, session_id: str, id: str, event: str, data: Dict[str, str]):
        """
        Add an event to this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to add the event to.
            id: The ID of the event.
            event: The event.
            data: Optional data to add to the event.
        """
        return EventEmittingBaseContainer._event_(writer, Entity.SESSION, session_id, id, event, data)

    @staticmethod
    def set_start_timestamp_(writer: LogWriter, session_id: str, timestamp: datetime):
        """
        Set the start timestamp for this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to set the start timestamp for.
            timestamp: The start timestamp.
        """
        return EventEmittingBaseContainer._set_start_timestamp_(writer, Entity.SESSION, session_id, timestamp)

    @staticmethod
    def set_end_timestamp_(writer: LogWriter, session_id: str, timestamp: datetime):
        """
        Set the end timestamp for this session.

        Args:
            writer: The LogWriter instance to use.
            session_id: The ID of the session to set the end timestamp for.
            timestamp: The end timestamp.
        """
        return EventEmittingBaseContainer._set_end_timestamp_(writer, Entity.SESSION, session_id, timestamp)
