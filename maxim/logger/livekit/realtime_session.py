"""Cross-provider realtime session instrumentation.

This module provides generic pre/post hooks and event interception that work
across different realtime providers (e.g., OpenAI and Gemini). It delegates
provider-specific handling to the corresponding submodules:

- OpenAI handlers: `maxim.logger.livekit.openai.realtime.handler`
- Gemini handlers: `maxim.logger.livekit.gemini.gemini_realtime_session`

When and why this module is used
--------------------------------
- Methods of LiveKit's `RealtimeSession` are wrapped (see each provider's
  instrumenter) to call into these hooks.
- We centralize app-level concerns like buffering audio at the session level,
  marking interruptions, and forwarding SDK events to the provider-specific
  path based on `session_info.provider`.
"""

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
# Import Gemini handler conditionally to avoid dependency issues
try:
    from .gemini.gemini_realtime_session import handle_google_input_transcription_completed
    GEMINI_HANDLER_AVAILABLE = True
except (ImportError, NameError):
    # Gemini dependencies not available
    handle_google_input_transcription_completed = None
    GEMINI_HANDLER_AVAILABLE = False
from .openai.realtime.handler import (
    handle_openai_client_event_queued,
    handle_openai_input_transcription_completed,
    handle_openai_server_event_received,
)
from .store import get_maxim_logger, get_session_store
from .utils import get_thread_pool_executor


def intercept_realtime_session_emit(self: RealtimeSession, event, data):
    """
    This function is called when the realtime session emits an event.
    """
    if event == "openai_client_event_queued":
        # Here we are buffering the session level audio buffer first
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().debug("[MaximSDK] session info is none at realtime session emit")
            return
        if session_info.user_speaking:
            handle_openai_client_event_queued(session_info, data)
    elif event == "openai_server_event_received":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().debug("[MaximSDK] session info is none at realtime session emit")
            return
        handle_openai_server_event_received(session_info, data)
    elif event == "input_speech_stopped":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().debug("[MaximSDK] session info is none at realtime session emit")
            return
        session_info.user_speaking = False
        get_session_store().set_session(session_info)
    elif event == "input_audio_transcription_completed":
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().debug("[MaximSDK] session info is none at realtime session emit")
            return
        if session_info.provider == "openai-realtime":
            handle_openai_input_transcription_completed(session_info, data)
        elif session_info.provider == "google-realtime":
            if GEMINI_HANDLER_AVAILABLE and handle_google_input_transcription_completed:
                handle_google_input_transcription_completed(session_info, data)
    elif event == "error":
        scribe().debug(f"[Internal][{self.__class__.__name__}] error;")
        if data is not None and isinstance(data, RealtimeModelError):
            main_error: RealtimeModelError = data
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
            scribe().warning(f"[{self.__class__.__name__}] error; error={data}")
    else:
        scribe().debug(
            f"[Internal][{self.__class__.__name__}] emit called; args={data}"
        )


def handle_interrupt(self):
    scribe().debug(f"[Internal][{self.__class__.__name__}] interrupt called;")
    rt_session_id = id(self)
    session_info = get_session_store().get_session_by_rt_session_id(rt_session_id)
    if session_info is None:
        return
    turn = session_info.current_turn
    if turn is not None:
        turn.is_interrupted = True
        session_info.current_turn = turn
        get_session_store().set_session(session_info)
    trace = get_session_store().get_current_trace_from_rt_session_id(rt_session_id)
    if trace is None:
        return
    trace.event(id=str(uuid4()), name="Interrupt", tags={"type": "interrupt"})


def handle_off(self):
    scribe().debug(f"[Internal][{self.__class__.__name__}] off called;")
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        return
    session_id = session_info.mx_session_id
    index = session_info.conversation_buffer_index
    if session_info.conversation_buffer.tell() == 0:
        return
    get_maxim_logger().session_add_attachment(
        session_id,
        FileDataAttachment(
            data=pcm16_to_wav_bytes(session_info.conversation_buffer.getvalue()),
            tags={"attach-to": "input"},
            name=f"Conversation part {index}",
            timestamp=int(time.time()),
        ),
    )
    get_maxim_logger().session_end(session_id=session_id)


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            if not args or len(args) == 0:
                return
            event = args[0]
            get_thread_pool_executor().submit(
                intercept_realtime_session_emit, self, event, args[1]
            )
        elif hook_name == "interrupt":
            get_thread_pool_executor().submit(handle_interrupt, self)
        elif hook_name == "off":
            get_thread_pool_executor().submit(handle_off, self)
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
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
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
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
