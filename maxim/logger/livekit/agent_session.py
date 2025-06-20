import functools
import inspect
import traceback
import uuid
import weakref
from datetime import datetime, timezone
from io import BytesIO

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.events import FunctionToolsExecutedEvent
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
from .utils import get_thread_pool_executor


def intercept_session_start(self: AgentSession, room, room_name, agent: Agent):
    """
    This function is called when a session starts.
    This is the point where we create a new session for Maxim.
    The session info along with room_id, agent_id, etc is stored in the thread-local store.
    """
    maxim_logger = get_maxim_logger()
    scribe().debug(f"[Internal][{self.__class__.__name__}] Session started")
    # getting the room_id
    if isinstance(room, str):
        room_id = room
        room_name = room
    elif isinstance(room, Room):
        room_id = room.sid
        room_name = room.name
    else:
        room_id = id(room)
        if isinstance(room, dict):
            room_name = room.get("name")
    scribe().debug(f"[Internal]session key:{id(self)}")
    scribe().debug(f"[Internal]Room: {room_id}")
    scribe().debug(f"[Internal]Agent: {agent.instructions}")
    # creating trace as well
    session_id = str(uuid.uuid4())
    session = maxim_logger.session({"id": session_id, "name": "livekit-session"})
    # adding tags to the session
    if room_id is not None:
        session.add_tag("room_id", str(room_id))
    if room_name is not None:
        session.add_tag("room_name", str(room_name))
    if session_id is not None:
        session.add_tag("session_id", str(session_id))
    if agent is not None:
        session.add_tag("agent_id", str(id(agent)))
    # If callback is set, emit the session started event
    callback = get_livekit_callback()
    if callback is not None:
        try:
            callback(
                "maxim.session.started", {"session_id": session_id, "session": session}
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}"
            )
    trace_id = str(uuid.uuid4())
    tags: dict[str, str] = {}
    if room_id is not None:
        tags["room_id"] = str(room_id)
    if room_name is not None:
        tags["room_name"] = room_name
    tags["session_id"] = str(id(self))
    if agent is not None:
        tags["agent_id"] = str(id(agent))
    trace = session.trace(
        {
            "id": trace_id,
            "input": agent.instructions,
            "name": "Greeting turn",
            "session_id": session_id,
            "tags": tags,
        }
    )
    callback = get_livekit_callback()
    if callback is not None:
        try:
            callback("maxim.trace.started", {"trace_id": trace_id, "trace": trace})
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] An error was captured during LiveKit callback execution: {e!s}"
            )
    current_turn = Turn(
        turn_id=str(uuid.uuid4()),
        turn_sequence=0,
        turn_timestamp=datetime.now(timezone.utc),
        is_interrupted=False,
        turn_input_transcription="",
        turn_output_transcription="",
        turn_input_audio_buffer=BytesIO(),
        turn_output_audio_buffer=BytesIO(),
    )
    get_session_store().set_session(
        SessionStoreEntry(
            room_id=room_id,
            user_speaking=False,
            provider="unknown",
            conversation_buffer=BytesIO(),
            conversation_buffer_index=1,
            state=SessionState.INITIALIZED,
            agent_id=id(agent),
            room_name=room_name,
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


def intercept_update_agent_state(self, new_state):
    """
    This function is called when the agent state is updated.
    """
    if new_state is None:
        return
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(
            str(uuid.uuid4()),
            f"agent_{new_state}",
            {"new_state": new_state, "platform": "livekit"},
        )


def intercept_generate_reply(self, instructions):
    """
    This function is called when the agent generates a reply.
    """
    if instructions is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] Generate reply; instructions={instructions}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.set_input(instructions)


def intercept_user_state_changed(self, new_state):
    """
    This function is called when the user state is changed.
    """
    if new_state is None:
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] User state changed; new_state={new_state}"
    )
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is not None:
        trace.event(
            str(uuid.uuid4()),
            f"user_{new_state}",
            {"new_state": new_state, "platform": "livekit"},
        )


def handle_tool_call_executed(self, event: FunctionToolsExecutedEvent):
    """
    This function is called when the agent executes a tool call.
    """
    trace = get_session_store().get_current_trace_for_agent_session(id(self))
    if trace is None:
        return
    # this we consider as a tool call result event
    # tool call creation needs to be done at each provider level
    for function_call in event.function_calls:
        tool_call = trace.tool_call(
            {
                "id": function_call.call_id,
                "name": function_call.name,
                "description": "",
                "args": (
                    str(function_call.arguments)
                    if function_call.arguments is not None
                    else ""
                ),
            }
        )
        tool_output = ""
        for output in event.function_call_outputs or []:
            if output.call_id == function_call.call_id:
                tool_output = output.output
                break
        tool_call.result(tool_output)


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            room = kwargs.get("room")
            room_name = kwargs.get("room_name")
            agent = kwargs.get("agent")
            get_thread_pool_executor().submit(
                intercept_session_start, self, room, room_name, agent
            )
        elif hook_name == "_update_agent_state":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(
                intercept_update_agent_state, self, args[0]
            )
        elif hook_name == "generate_reply":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(intercept_generate_reply, self, args[0])
        elif hook_name == "_update_user_state":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(
                intercept_user_state_changed, self, args[0]
            )
        elif hook_name == "emit":
            if args[0] == "metrics_collected":
                pass
            elif args[0] == "function_tools_executed":
                if not args or len(args) == 0:
                    return
                get_thread_pool_executor().submit(
                    handle_tool_call_executed, self, args[1]
                )
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
            )
        elif hook_name == "end":
            pass
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
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
        scribe().debug(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
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
