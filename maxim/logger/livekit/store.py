import threading
from typing import Optional, TypedDict, Union


class SessionStoreEntry(TypedDict, total=False):
    room_id: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]
    rt_session_id: Optional[str]
    current_trace_id: Optional[str]


class LivekitSessionStore:
    def __init__(self):
        self.mx_livekit_session_store: dict[str, SessionStoreEntry] = {}

    def get_session_by_room_id(self, room_id: str) -> Union[SessionStoreEntry, None]:
        return self.mx_livekit_session_store.get(room_id, None)

    def get_session_by_session_id(
        self, session_id: str
    ) -> Union[SessionStoreEntry, None]:
        for entry in self.mx_livekit_session_store.values():
            if "session_id" in entry and entry["session_id"] == session_id:
                return entry
        return None

    def get_session_by_rt_session_id(
        self, rt_session_id: str
    ) -> Union[SessionStoreEntry, None]:
        for entry in self.mx_livekit_session_store.values():
            if "rt_session_id" in entry and entry["rt_session_id"] == rt_session_id:
                return entry
        return None

    def set_session(self, room_id, entry: SessionStoreEntry):
        self.mx_livekit_session_store[room_id] = entry

    def delete_session(self, room_id):
        del self.mx_livekit_session_store[room_id]

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
