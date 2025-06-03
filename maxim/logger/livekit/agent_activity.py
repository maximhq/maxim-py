import functools
import inspect
import traceback
from uuid import uuid4

from livekit.agents.voice.agent_activity import AgentActivity

from ...scribe import scribe
from .store import get_session_store
from .utils import start_new_turn

agent_activity_f_skip_list = []


def post_start(self: AgentActivity, *args, **kwargs):
    scribe().debug(f"[Internal][{self.__class__.__name__}] post start called")
    # Trying to get AgentSession and RealtimeSession handles
    rt_session_id = id(self._rt_session)
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session start")
        return
    scribe().debug(
        f"[Internal][{self.__class__.__name__}] session info: {session_info}"
    )
    session_info["rt_session_id"] = rt_session_id
    session_info["rt_session"] = self._rt_session
    get_session_store().set_session(session_info)


def handle_interrupt(self: AgentActivity, *args, **kwargs):
    trace = get_session_store().get_current_trace_for_agent_session(id(self._session))
    if trace is None:
        scribe().error("[MaximSDK] trace is none at realtime session interrupt")
        return
    trace.event(id=str(uuid4()), name="interrupted")
    # here we will need to end the turn


def handle_input_speech_started(self: AgentActivity, *args, **kwargs):
    scribe().debug(f"[Internal][{self.__class__.__name__}] input speech started called")
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    start_new_turn(session_info)


def handle_create_speech_task(self: AgentActivity, args, kwargs):
    if self.agent.session.agent_state != "listening":
        return
    scribe().debug(f"[Internal][{self.__class__.__name__}] create speech task called")
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    if session_info["provider"] != "google-realtime":
        return
    # This is currently hack as Gemini does not support server side interruptions
    current_turn = session_info["current_turn"]
    if current_turn is None:
        scribe().error("[MaximSDK] current turn is none at realtime session emit")
        return
    input_buffer = None
    # Check if there is data and if its there copy it
    if len(current_turn["turn_input_audio_buffer"]) > 0:
        input_buffer = bytes(current_turn["turn_input_audio_buffer"])
    start_new_turn(session_info)
    if input_buffer is not None and len(input_buffer) > 0:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self._session)
        )
        if session_info is not None and input_buffer is not None:
            session_info["current_turn"]["turn_input_audio_buffer"] = bytes(
                input_buffer
            )
            get_session_store().set_session(session_info)


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "interrupt":
            handle_interrupt(self, args, kwargs)
        elif hook_name == "_on_input_speech_started":
            handle_input_speech_started(self, args, kwargs)
        elif hook_name == "_create_speech_task":
            handle_create_speech_task(self, args, kwargs)
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
        if hook_name == "start":
            post_start(self, *args, **kwargs)    
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )    
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_agent_activity(orig, name):
    if name in agent_activity_f_skip_list:
        return orig

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
            except Exception:
                pass
            post_hook(self, result, name, args, kwargs)            

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
