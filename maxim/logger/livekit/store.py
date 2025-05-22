import threading
from typing import Optional, TypedDict, Union

from livekit.agents import AgentSession
from livekit.agents.llm import RealtimeSession


# TODO convert to weak references
class SessionStoreEntry(TypedDict):
    room_id: str
    agent_id: Optional[int]
    session_id: Optional[int]
    session: Optional[AgentSession]
    rt_session_id: Optional[int]
    rt_session: Optional[RealtimeSession]
    mx_current_trace_id: Optional[str]
    mx_session_id: Optional[str]
    rt_session_info: Optional[dict]


class LivekitSessionStore:
    def __init__(self):
        self.mx_livekit_session_store: list[SessionStoreEntry] = []

    def get_session_by_room_id(self, room_id: str) -> Union[SessionStoreEntry, None]:
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

    def get_all_sessions(self):
        return self.mx_livekit_session_store


# Create a thread-local storage for the session store
_thread_local = threading.local()


def get_session_store():
    """Get the thread-local session store instance."""
    if not hasattr(_thread_local, "session_store"):
        _thread_local.session_store = LivekitSessionStore()
    return _thread_local.session_store
