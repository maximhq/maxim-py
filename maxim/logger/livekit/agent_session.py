import functools
import inspect
import traceback
import uuid
import weakref
from datetime import datetime, timezone

from livekit.agents import Agent, AgentSession
from livekit.protocol.models import Room

from ...scribe import scribe
from .store import (
    SessionState,
    SessionStoreEntry,
    Turn,
    get_livekit_callback,
    get_maxim_logger,
    get_session_store,
)


def intercept_session_start(self: AgentSession, *args, **kwargs):
    """
    This function is called when a session starts.
    This is the point where we create a new session for Maxim.
    The session info along with room_id, agent_id, etc is stored in the thread-local store.
    """
    maxim_logger = get_maxim_logger()
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] Session started; args={args}, kwargs={kwargs}"
    )
    # getting the room_id
    room = kwargs.get("room", None)
    if isinstance(room, str):
        room_id = room
    elif isinstance(room, Room):
        room_id = room.sid
    else:
        room_id = id(room)
    agent: Agent = kwargs.get("agent", None)
    scribe().debug(f"[Internal]session key:{id(self)}")
    scribe().debug(f"[Internal]Room: {room_id}")
    scribe().debug(f"[Internal]Agent: {agent.instructions}")
    # creating trace as well
    session_id = str(uuid.uuid4())
    session = maxim_logger.session({"id": session_id, "name": "livekit-session"})
    # If callback is set, emit the session started event
    if get_livekit_callback() is not None:
        get_livekit_callback()(
            "maxim.session.started", {"session_id": session_id, "session": session}
        )    
    trace_id = str(uuid.uuid4())
    trace = session.trace(
        {
            "id": trace_id,
            "input": "",
            "name": "Greeting turn",
            "session_id": session_id,
            "tags": {
                "room_id": room_id,
                "agent_id": str(id(agent)),
            },
        }
    )
    if get_livekit_callback() is not None:
        get_livekit_callback()(
            "maxim.trace.started", {"trace_id": trace_id, "trace": trace}
        )
    current_turn = Turn(
        turn_id=str(uuid.uuid4()),
        turn_sequence=0,
        turn_timestamp=datetime.now(timezone.utc),
        is_interrupted=False,
        turn_input_transcription="",
        turn_output_transcription="",
        turn_input_audio_buffer=bytes(),
        turn_output_audio_buffer=bytes(),
    )
    get_session_store().set_session(
        SessionStoreEntry(
            room_id=room_id,
            user_speaking=False,
            provider="unknown",
            conversation_buffer=bytes(),
            conversation_buffer_index=1,
            state=SessionState.INITIALIZED,
            agent_id=id(agent),
            agent_session_id=id(self),
            agent_session=weakref.ref(self),
            rt_session_id=None,
            rt_session=None,
            llm_config=None,
            rt_session_info={},
            mx_current_trace_id=trace_id,
            mx_session_id=session_id,
            current_turn=current_turn,
        ),
    )


def intercept_update_agent_state(self, *args, **kwargs):
    """
    This function is called when the agent state is updated.
    """
    if not args or len(args) == 0:
        return
    new_state = args[0]
    if new_state is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] Agent state updated; new_state={new_state}"
    )    
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(str(uuid.uuid4()), "agent_state_updated", {"new_state": new_state})


def intercept_generate_reply(self, *args, **kwargs):
    """
    This function is called when the agent generates a reply.
    """
    instructions = kwargs.get("instructions", None)
    if instructions is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] Generate reply; instructions={instructions} kwargs={kwargs}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.set_input(instructions)


def intercept_user_state_changed(self, *args, **kwargs):
    """
    This function is called when the user state is changed.
    """
    if not args or len(args) == 0:
        return
    new_state = args[0]
    if new_state is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] User state changed; new_state={new_state}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(str(uuid.uuid4()), "user_state_changed", {"new_state": new_state})


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            intercept_session_start(self, *args, **kwargs)
        elif hook_name == "_update_agent_state":
            intercept_update_agent_state(self, *args, **kwargs)
        elif hook_name == "generate_reply":
            intercept_generate_reply(self, *args, **kwargs)
        elif hook_name == "_update_user_state":
            intercept_user_state_changed(self, *args, **kwargs)
        elif hook_name == "emit":
            if args[0] == "metrics_collected":
                pass
            scribe().debug(f"[Internal][{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}")
        elif hook_name == "end":
            pass
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            if args[0] == "metrics_collected":
                pass
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_session(orig, name):
    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = await orig(self, *args, **kwargs)
                return result
            finally:
                post_hook(self, result, name, args, kwargs)            

        wrapper = async_wrapper
    else:

        def sync_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = orig(self, *args, **kwargs)
                return result
            finally:
                post_hook(self, result, name, args, kwargs)            

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
