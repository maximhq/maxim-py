from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict, Union

from typing_extensions import deprecated

from ..writer import LogWriter
from .base import EventEmittingBaseContainer
from .feedback import Feedback, FeedbackDict, get_feedback_dict
from .trace import Trace, TraceConfig, TraceConfigDict, get_trace_config_dict
from .types import Entity


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class SessionConfig:
    id: str
    name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

class SessionConfigDict(TypedDict, total=False):
    id: str
    name: Optional[str]
    tags: Optional[Dict[str, str]]


def get_session_config_dict(
    config: Union[SessionConfig, SessionConfigDict],
) -> dict[str, Any]:
    return (
        dict(
            SessionConfigDict(
                id=config.id,
                name=config.name,
                tags=config.tags,
            )
        )
        if isinstance(config, SessionConfig)
        else dict(config)
    )


class Session(EventEmittingBaseContainer):
    ENTITY = Entity.SESSION

    def __init__(
        self, config: Union[SessionConfig, SessionConfigDict], writer: LogWriter
    ):
        final_config = get_session_config_dict(config)
        super().__init__(Session.ENTITY, dict(final_config), writer)
        self._commit("create")

    def trace(self, config: Union[TraceConfig, TraceConfigDict]) -> Trace:
        final_config = get_trace_config_dict(config)
        final_config["session_id"] = self.id
        return Trace(final_config, self.writer)

    @staticmethod
    def trace_(
        writer: LogWriter, session_id: str, config: Union[TraceConfig, TraceConfigDict]
    ) -> Trace:
        final_config = get_trace_config_dict(config)
        final_config["session_id"] = session_id
        return Trace(final_config, writer)

    def feedback(self, feedback: Union[Feedback, FeedbackDict]):
        self._commit("add-feedback", dict(get_feedback_dict(feedback)))

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
        return EventEmittingBaseContainer._add_tag_(writer, Entity.SESSION, session_id, key, value)

    @staticmethod
    def end_(writer: LogWriter, session_id: str, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = {}
        return EventEmittingBaseContainer._end_(writer, Entity.SESSION, session_id, {
            "endTimestamp": datetime.now(timezone.utc),
            **data,
        })

    @staticmethod
    def event_(writer: LogWriter, session_id: str, id: str, event: str, data: Dict[str, str]):
        return EventEmittingBaseContainer._event_(writer, Entity.SESSION, session_id, id, event, data)
