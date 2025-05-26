import functools
import inspect
import traceback
import uuid
import weakref
from datetime import datetime, timezone

from livekit.agents import Agent, AgentSession

from ...scribe import scribe
from .store import (
    SessionState,
    SessionStoreEntry,
    Turn,
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
        f"[{self.__class__.__name__}] Session started; args={args}, kwargs={kwargs}"
    )
    # getting the room_id
    room_id = kwargs.get("room", None)
    agent: Agent = kwargs.get("agent", None)
    scribe().debug(f"session key:{id(self)}")
    scribe().debug(f"Room: {room_id}")
    scribe().debug(f"Agent: {agent.instructions}")
    # creating trace as well
    session_id = str(uuid.uuid4())
    session = maxim_logger.session({"id": session_id, "name": "livekit-session"})
    trace_id = str(uuid.uuid4())
    session.trace(
        {
            "id": trace_id,
            "input": "",
            "session_id": session_id,
            "tags": {
                "room_id": room_id,
                "agent_id": str(id(agent)),
            },
        }
    )
    current_turn = Turn(
        turn_id=str(uuid.uuid4()),
        turn_sequence=0,
        turn_timestamp=datetime.now(timezone.utc),
        turn_audio_buffer=bytes(),
        messages=[],
    )
    get_session_store().set_session(
        SessionStoreEntry(
            room_id=room_id,
            state=SessionState.INITIALIZED,
            agent_id=id(agent),
            session_id=id(self),
            session=weakref.ref(self),
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
    new_state = args[0]
    scribe().debug(
        f"[{self.__class__.__name__}] Agent state updated; new_state={new_state}"
    )
    trace = get_session_store().get_current_trace_for_session(id(self))
    if trace is not None:
        trace.event(str(uuid.uuid4()), "agent_state_updated", {"new_state": new_state})


def intercept_generate_reply(self, *args, **kwargs):
    """
    This function is called when the agent generates a reply.
    """
    instructions = kwargs.get("instructions", None)
    scribe().debug(
        f"[{self.__class__.__name__}] Generate reply; instructions={instructions} kwargs={kwargs}"
    )
    trace = get_session_store().get_current_trace_for_session(id(self))
    if trace is not None:
        trace.set_input(instructions)


def intercept_user_state_changed(self, *args, **kwargs):
    """
    This function is called when the user state is changed.
    """
    new_state = args[0]
    scribe().debug(
        f"[{self.__class__.__name__}] User state changed; new_state={new_state}"
    )
    trace = get_session_store().get_current_trace_for_session(id(self))
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
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
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
                f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_session(orig, name):
    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = await orig(self, *args, **kwargs)
            post_hook(self, result, name, args, kwargs)
            return result

        wrapper = async_wrapper
    else:

        def sync_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = orig(self, *args, **kwargs)
            post_hook(self, result, name, args, kwargs)
            return result

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
