import functools
import inspect
import time
import traceback
from uuid import uuid4

from livekit.agents.llm import (
    RealtimeModelError,
    RealtimeSession,
)

from ...scribe import scribe
from ..components import FileDataAttachment
from ..utils import pcm16_to_wav_bytes
from .gemini.gemini_realtime_session import handle_google_input_transcription_completed
from .openai.realtime.handler import (
    handle_openai_client_event_queued,
    handle_openai_input_transcription_completed,
    handle_openai_server_event_received,
)
from .store import get_maxim_logger, get_session_store


def intercept_realtime_session_emit(self: RealtimeSession, *args, **kwargs):
    """
    This function is called when the realtime session emits an event.
    """
    if not args or len(args) == 0:
        return
    event = args[0]
    if event == "openai_client_event_queued":
        # Here we are buffering the session level audio buffer first
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().error("[MaximSDK] session info is none at realtime session emit")
            return
        if session_info["user_speaking"]:
            handle_openai_client_event_queued(session_info, args[1])
    elif event == "openai_server_event_received":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().error("[MaximSDK] session info is none at realtime session emit")
            return
        handle_openai_server_event_received(session_info, args[1])
    elif event == "input_speech_stopped":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().error("[MaximSDK] session info is none at realtime session emit")
            return
        session_info["user_speaking"] = False
        get_session_store().set_session(session_info)
    elif event == "metrics_collected":
        pass
    elif event == "generation_created":
        pass
    elif event == "input_audio_transcription_completed":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().error("[MaximSDK] session info is none at realtime session emit")
            return
        if session_info["provider"] == "openai-realtime":
            handle_openai_input_transcription_completed(session_info, args[1])
        elif session_info["provider"] == "google-realtime":
            handle_google_input_transcription_completed(session_info, args[1])
    elif event == "error":
        scribe().debug(
            f"[Internal]=====[{self.__class__.__name__}] error; args={args}, kwargs={kwargs}"
        )
        if args[1] is not None and isinstance(args[1], RealtimeModelError):
            main_error: RealtimeModelError = args[1]
            trace = get_session_store().get_current_trace_from_rt_session_id(id(self))
            if trace is not None:
                trace.add_error(
                    {
                        "id": str(uuid4()),
                        "name": main_error.type,
                        "type": main_error.label,
                        "message": main_error.error.__str__(),
                        "metadata": {
                            "recoverable": main_error.recoverable,
                            "trace": main_error.error.__traceback__,
                        },
                    }
                )
        else:
            scribe().error(f"[{self.__class__.__name__}] error; error={args[1]}")
    else:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
        )


def handle_interrupt(self, *args, **kwargs):
    scribe().debug(f"[Internal][{self.__class__.__name__}] interrupt called;")
    rt_session_id = id(self)
    session_info = get_session_store().get_session_by_rt_session_id(rt_session_id)
    if session_info is None:
        return
    turn = session_info.get("current_turn", None)
    if turn is not None:
        turn["is_interrupted"] = True
        session_info["current_turn"] = turn
        get_session_store().set_session(session_info)
    trace = get_session_store().get_current_trace_from_rt_session_id(rt_session_id)
    if trace is None:
        return
    trace.event(id=str(uuid4()), name="Interrupt", tags={"type": "interrupt"})


def handle_off(self, *args, **kwargs):
    scribe().debug(f"[Internal][{self.__class__.__name__}] off called;")
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        return
    session_id = session_info["mx_session_id"]
    index = session_info["conversation_buffer_index"]
    get_maxim_logger().session_add_attachment(
        session_id,
        FileDataAttachment(
            data=pcm16_to_wav_bytes(session_info["conversation_buffer"]),
            tags={"attach-to": "input"},
            name=f"Conversation part {index}",
            timestamp=int(time.time()),
        ),
    )
    get_maxim_logger().session_end(session_id=session_id)


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            intercept_realtime_session_emit(self, *args, **kwargs)
        elif hook_name == "interrupt":
            handle_interrupt(self, *args, **kwargs)
        elif hook_name == "off":
            handle_off(self, *args, **kwargs)
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
            pass
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_realtime_session(orig, name):
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
