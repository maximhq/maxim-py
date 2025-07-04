import functools
import inspect
import traceback
from uuid import uuid4
import weakref

from livekit.agents.voice.agent_activity import AgentActivity

from ...scribe import scribe
from .store import get_session_store
from .utils import get_thread_pool_executor, start_new_turn

agent_activity_f_skip_list = []


def handle_interrupt(self: AgentActivity):
    trace = get_session_store().get_current_trace_for_agent_session(id(self._session))
    if trace is None:
        scribe().warning(
            "[MaximSDK] trace is none at realtime session interrupt. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    trace.event(id=str(uuid4()), name="interrupted")
    # here we will need to end the turn


def post_start(self: AgentActivity):
    if self._rt_session is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] rt_session is none at realtime session start. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    # Trying to get AgentSession and RealtimeSession handles
    rt_session_id = id(self._rt_session)
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session start. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    session_info.rt_session_id = rt_session_id
    session_info.rt_session = weakref.ref(self._rt_session)
    get_session_store().set_session(session_info)


def handle_input_speech_started(self: AgentActivity):
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    if session_info.provider == "google-realtime":
        trace = get_session_store().get_current_trace_for_agent_session(
            id(self._session)
        )
        if trace is not None:
            trace.event(str(uuid4()), "user_speaking", {"platform": "livekit"})
        session_info.user_speaking = True
        session_info.current_turn.is_interrupted = True
        get_session_store().set_session(session_info)
        return
    # here we can check if the current turn is interrupted
    if (
        session_info.current_turn is not None
        and session_info.current_turn.turn_input_audio_buffer.tell() == 0
        and session_info.current_turn.turn_output_audio_buffer.tell() == 0
    ):
        # we will reuse the same turn
        return
    start_new_turn(session_info)


def handle_create_speech_task(self: AgentActivity):
    if self.agent.session.agent_state != "listening":
        return
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self._session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    if session_info.provider != "google-realtime":
        return
    # This is currently hack as Gemini does not support server side interruptions
    if session_info.current_turn is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] current turn is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    input_buffer = None
    # Check if there is data and if its there copy it
    if (
        session_info.current_turn.turn_input_audio_buffer is not None
        and session_info.current_turn.turn_input_audio_buffer.tell() > 0
    ):
        # Take only the last 5 seconds of audio from the buffer
        audio_bytes = session_info.current_turn.turn_input_audio_buffer.getvalue()
        sample_rate = getattr(
            session_info, "sample_rate", 16000
        )  # default to 16kHz if not set
        sample_width = getattr(
            session_info, "sample_width", 2
        )  # default to 2 bytes (16-bit PCM) if not set
        channels = getattr(session_info, "channels", 1)  # default to mono if not set

        bytes_per_second = sample_rate * sample_width * channels
        last_5_sec_bytes = bytes_per_second * 5

        if len(audio_bytes) > last_5_sec_bytes:
            input_buffer = audio_bytes[-last_5_sec_bytes:]
        else:
            input_buffer = audio_bytes
    start_new_turn(session_info)
    if input_buffer is not None and len(input_buffer) > 0:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self._session)
        )
        if session_info is not None and input_buffer is not None:
            session_info.current_turn.turn_input_audio_buffer.write(input_buffer)
            get_session_store().set_session(session_info)


def pre_hook(self, hook_name, args, kwargs):
    ignored_hooks = ["push_audio"]
    try:
        if hook_name == "interrupt":
            get_thread_pool_executor().submit(handle_interrupt, self)
        elif hook_name == "_on_input_speech_started":
            get_thread_pool_executor().submit(handle_input_speech_started, self)
        elif hook_name == "_create_speech_task":
            get_thread_pool_executor().submit(handle_create_speech_task, self)
        else:
            if hook_name in ignored_hooks:
                return
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    try:
        if hook_name == "start":
            get_thread_pool_executor().submit(post_start, self)
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
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
            except Exception as e:
                scribe().debug(
                    f"[Internal][{self.__class__.__name__}] {name} failed; error={e!s}\n{traceback.format_exc()}"
                )
            post_hook(self, result, name, args, kwargs)

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
