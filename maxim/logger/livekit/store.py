import threading
import weakref
from datetime import datetime
from enum import Enum
from typing import List, Optional, TypedDict, Union

from livekit.agents import AgentSession
from livekit.agents.llm import RealtimeSession

from ..logger import GenerationRequestMessage, Logger, Trace


class SessionState(Enum):
    INITIALIZED = 0
    GREETING = 1
    STARTED = 2


class Turn(TypedDict):
    turn_id: str
    turn_sequence: int
    turn_timestamp: datetime
    turn_audio_buffer: bytes
    messages: List[GenerationRequestMessage]


class LLMConfig(TypedDict):
    id: str
    object: str
    expires_at: int
    input_audio_noise_reduction: Optional[bool]
    turn_detection: dict
    input_audio_format: str
    input_audio_transcription: Optional[dict]
    client_secret: Optional[str]
    include: Optional[list]
    model: str
    modalities: list[str]
    instructions: str
    voice: str
    output_audio_format: str
    tool_choice: str
    temperature: float
    max_response_output_tokens: Union[str, int]
    speed: float
    tools: list


class SessionStoreEntry(TypedDict):
    room_id: int
    state: SessionState
    llm_config: Optional[LLMConfig]
    agent_id: Optional[int]
    session_id: Optional[int]
    session: Optional[weakref.ref[AgentSession]]
    rt_session_id: Optional[int]
    rt_session: Optional[weakref.ref[RealtimeSession]]
    mx_current_trace_id: Optional[str]
    mx_session_id: Optional[str]
    rt_session_info: Optional[dict]
    current_turn: Optional[Turn]


maxim_logger: Union[Logger, None] = None


def get_maxim_logger() -> Logger:
    """Get the global maxim logger instance."""
    if maxim_logger is None:
        raise ValueError("Maxim logger is not set")
    return maxim_logger


def set_maxim_logger(logger: Logger) -> None:
    """Set the global maxim logger instance."""
    global maxim_logger
    maxim_logger = logger


class LivekitSessionStore:
    def __init__(self):
        self.mx_livekit_session_store: list[SessionStoreEntry] = []

    def get_session_by_room_id(self, room_id: int) -> Union[SessionStoreEntry, None]:
        for entry in self.mx_livekit_session_store:
            if "room_id" in entry and entry["room_id"] == room_id:
                return entry
        return None

    def get_session_by_session_id(
        self, session_id: int
    ) -> Union[SessionStoreEntry, None]:
        for entry in self.mx_livekit_session_store:
            if "session_id" in entry and entry["session_id"] == session_id:
                return entry
        return None

    def get_session_by_rt_session_id(
        self, rt_session_id: int
    ) -> Union[SessionStoreEntry, None]:
        for entry in self.mx_livekit_session_store:
            if "rt_session_id" in entry and entry["rt_session_id"] == rt_session_id:
                return entry
        return None

    def set_session(self, entry: SessionStoreEntry):
        if "room_id" in entry:
            # find the entry and replace
            for i, e in enumerate(self.mx_livekit_session_store):
                if e["room_id"] == entry["room_id"]:
                    self.mx_livekit_session_store[i] = entry
                    return
        self.mx_livekit_session_store.append(entry)

    def delete_session(self, room_id):
        for entry in self.mx_livekit_session_store:
            if "room_id" in entry and entry["room_id"] == room_id:
                self.mx_livekit_session_store.remove(entry)

    def clear_all_sessions(self):
        self.mx_livekit_session_store.clear()

    def get_current_trace_for_session(self, session_id: int) -> Union[Trace, None]:
        session = self.get_session_by_session_id(session_id)
        if session is None:
            return None
        trace_id = session["mx_current_trace_id"]
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_current_trace_for_room_id(self, room_id: int) -> Union[Trace, None]:
        session = self.get_session_by_room_id(room_id)
        if session is None:
            return None
        trace_id = session["mx_current_trace_id"]
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_current_trace_from_rt_session_id(
        self, rt_session_id: int
    ) -> Union[Trace, None]:
        session = self.get_session_by_rt_session_id(rt_session_id)
        if session is None:
            return None
        trace_id = session["mx_current_trace_id"]
        if trace_id is None:
            return None
        return get_maxim_logger().trace({"id": trace_id})

    def get_all_sessions(self):
        return self.mx_livekit_session_store


# Create a thread-local storage for the session store
_thread_local = threading.local()


def get_session_store():
    """Get the thread-local session store instance."""
    if not hasattr(_thread_local, "session_store"):
        _thread_local.session_store = LivekitSessionStore()
    return _thread_local.session_store
