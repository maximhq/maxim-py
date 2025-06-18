import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from threading import Lock
from typing import Any, Callable, Optional, Union

from livekit.agents import AgentSession
from livekit.agents.llm import RealtimeSession

from ...scribe import scribe
from ..components import FileDataAttachment
from ..logger import Logger, Trace
from ..utils import pcm16_to_wav_bytes


class SessionState(Enum):
    INITIALIZED = 0
    GREETING = 1
    STARTED = 2


@dataclass
class Turn:
    turn_id: str
    turn_sequence: int
    turn_timestamp: datetime
    is_interrupted: bool
    turn_input_transcription: str
    turn_output_transcription: str
    turn_input_audio_buffer: BytesIO
    turn_output_audio_buffer: BytesIO
    trace_id: Optional[str] = None

@dataclass
class SessionStoreEntry:
    room_id: str
    state: SessionState
    provider: str
    user_speaking: bool
    conversation_buffer_index: int
    conversation_buffer: BytesIO
    current_turn: Turn
    llm_config: Optional[dict[str, Any]] = None
    agent_id: Optional[int] = None
    room_name: Optional[str] = None
    agent_session_id: Optional[int] = None
    agent_session: Optional[weakref.ref[AgentSession]] = None
    rt_session_id: Optional[int] = None
    rt_session: Optional[weakref.ref[RealtimeSession]] = None
    mx_session_id: Optional[str] = None
    mx_current_trace_id: Optional[str] = None
    rt_session_info: Optional[dict] = None


MaximLiveKitCallback = Callable[[str, dict], None]
livekit_callback: Optional[MaximLiveKitCallback] = None

maxim_logger: Union[Logger, None] = None


def set_livekit_callback(callback: MaximLiveKitCallback) -> None:
    global livekit_callback
    livekit_callback = callback


def get_livekit_callback() -> Optional[MaximLiveKitCallback]:
    global livekit_callback
    return livekit_callback


def get_maxim_logger() -> Logger:
    """Get the global maxim logger instance."""
    if maxim_logger is None:
        raise ValueError("Maxim logger is not set")
    return maxim_logger


def set_maxim_logger(logger: Logger) -> None:
    """Set the global maxim logger instance."""
    global maxim_logger
    maxim_logger = logger


class LiveKitSessionStore:
    def __init__(self):
        self.mx_live_kit_session_store: list[SessionStoreEntry] = []
        self._lock = Lock()

    def get_session_by_room_id(self, room_id: str) -> Union[SessionStoreEntry, None]:
        with self._lock:
            for entry in self.mx_live_kit_session_store:
                if entry.room_id == room_id:
                    return entry
            return None

    def get_session_by_agent_session_id(
        self, session_id: int
    ) -> Union[SessionStoreEntry, None]:
        with self._lock:
            for entry in self.mx_live_kit_session_store:
                if (
                    entry.agent_session_id is not None
                    and entry.agent_session_id == session_id
                ):
                    return entry
            return None

    def get_session_by_rt_session_id(
        self, rt_session_id: int
    ) -> Union[SessionStoreEntry, None]:
        with self._lock:
            for entry in self.mx_live_kit_session_store:
                if (
                    entry.rt_session_id is not None
                    and entry.rt_session_id == rt_session_id
                ):
                    return entry
            return None

    def set_session(self, entry: SessionStoreEntry):
        with self._lock:
            if entry.agent_session_id is not None:
                # find the entry and replace
                for i, e in enumerate(self.mx_live_kit_session_store):
                    if (
                        e.agent_session_id is not None
                        and e.agent_session_id == entry.agent_session_id
                    ):
                        self.mx_live_kit_session_store[i] = entry
                        return
            self.mx_live_kit_session_store.append(entry)

    def delete_session(self, agent_session_id: int):
        with self._lock:
            # Use list comprehension to avoid modifying list while iterating
            self.mx_live_kit_session_store = [
                entry
                for entry in self.mx_live_kit_session_store
                if not (
                    entry.agent_session_id is not None
                    and entry.agent_session_id == agent_session_id
                )
            ]

    def clear_all_sessions(self):
        with self._lock:
            self.mx_live_kit_session_store.clear()

    def get_current_trace_for_agent_session(
        self, agent_session_id: int
    ) -> Union[Trace, None]:
        session = self.get_session_by_agent_session_id(agent_session_id)
        if session is None:
            return None
        trace_id = session.mx_current_trace_id
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_current_trace_for_room_id(self, room_id: str) -> Union[Trace, None]:
        session = self.get_session_by_room_id(room_id)
        if session is None:
            return None
        trace_id = session.mx_current_trace_id
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_current_trace_from_rt_session_id(
        self, rt_session_id: int
    ) -> Union[Trace, None]:
        session = self.get_session_by_rt_session_id(rt_session_id)
        if session is None:
            return None
        trace_id = session.mx_current_trace_id
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_all_sessions(self):
        with self._lock:
            return self.mx_live_kit_session_store.copy()

    def close_session(self, session_info: SessionStoreEntry):
        with self._lock:
            scribe().debug(f"[MaximSDK] Closing session {session_info.mx_session_id}")
            session_id = session_info.mx_session_id
            index = session_info.conversation_buffer_index

            get_maxim_logger().session_add_attachment(
                session_id,
                FileDataAttachment(
                    data=pcm16_to_wav_bytes(
                        session_info.conversation_buffer.getvalue()
                    ),
                    tags={"attach-to": "input"},
                    name=f"Conversation part {index}",
                    timestamp=int(time.time()),
                ),
            )
            get_maxim_logger().session_end(session_id=session_id)


# Create a thread-local storage for the session store
_session_store = LiveKitSessionStore()


def get_session_store():
    """Get the global session store instance."""
    return _session_store
