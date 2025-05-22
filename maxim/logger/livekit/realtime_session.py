import functools
import inspect
import traceback

from livekit.agents.llm import RealtimeSession

from ...scribe import scribe
from .openai.realtime.handler import handle_openai_server_event_received
from .store import get_session_store


def intercept_realtime_session_emit(self: RealtimeSession, *args, **kwargs):
    """
    This function is called when the realtime session emits an event.
    """
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    get_session_store().set_session(session_info)
    event = args[0]
    if event == "openai_client_event_queued":
        pass
    elif event == "openai_server_event_received":
        handle_openai_server_event_received(session_info, args[1])
    else:
        scribe().debug(
            f"[{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
        )

def intercept_realtime_session_on(self, *args, **kwargs):
    action = args[0]
    activity = args[1]
    if action == "generation_created":
        print(f"Activity: {activity}")
        scribe().debug(
            f"[{self.__class__.__name__}] generation_created called; args={args}, kwargs={kwargs}"
        )
    elif action == "input_speech_started":
        scribe().debug(
            f"[{self.__class__.__name__}] input_speech_started called; args={args}, kwargs={kwargs}"
        )
    elif action == "input_speech_stopped":
        scribe().debug(
            f"[{self.__class__.__name__}] input_speech_stopped called; args={args}, kwargs={kwargs}"
        )
    elif action == "input_audio_transcription_completed":
        scribe().debug(
            f"[{self.__class__.__name__}] input_audio_transcription_completed called; args={args}, kwargs={kwargs}"
        )


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            intercept_realtime_session_emit(self, *args, **kwargs)
        elif hook_name == "on":
            intercept_realtime_session_on(self, *args, **kwargs)
        elif hook_name == "interrupt":
            print("###########INTERRUPTED EVENT############")
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
            pass
        else:
            scribe().debug(
                f"[{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )

def instrument_realtime_session(orig, name):
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
