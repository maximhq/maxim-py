import functools
import inspect
import traceback
import uuid
from datetime import datetime, timezone
from uuid import uuid4

from livekit.agents.llm import (
    InputTranscriptionCompleted,
    RealtimeModelError,
    RealtimeSession,
)

from ...scribe import scribe
from .openai.realtime.handler import (
    handle_openai_client_event_queued,
    handle_openai_server_event_received,
)
from .store import Turn, get_maxim_logger, get_session_store


def intercept_realtime_session_emit(self: RealtimeSession, *args, **kwargs):
    """
    This function is called when the realtime session emits an event.
    """
    event = args[0]
    if event == "openai_client_event_queued":
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
        # adding a new generation to the current trace
        session_info = get_session_store().get_session_by_rt_session_id(id(self))
        if session_info is None:
            scribe().error("[MaximSDK] session info is none at realtime session emit")
            return
        trace = get_session_store().get_current_trace_from_rt_session_id(id(self))
        if trace is None:
            return
        # adding a new generation
        turn = session_info["current_turn"]
        if turn is None:
            return
        model = "unknown"
        llm_config = session_info.get("llm_config", None)
        if llm_config is not None:
            model = llm_config.get("model", "unknown")
        model_parameters = {}
        if llm_config is not None:
            model_parameters = llm_config.get("model_parameters", {})
        input_transcription: InputTranscriptionCompleted = args[1]
        trace.generation(
            {
                "id": turn["turn_id"],
                "model": model,
                "provider": "openai",
                "model_parameters": model_parameters,
                "messages": [
                    {"role": "user", "content": input_transcription.transcript}
                ],
            }
        )

    elif event == "input_speech_started":
        pass
    elif event == "error":
        scribe().debug(
            f"=====[{self.__class__.__name__}] error; args={args}, kwargs={kwargs}"
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
                            "trace": main_error.error.with_traceback,
                        },
                    }
                )
        else:
            scribe().error(f"[{self.__class__.__name__}] error; error={args[1]}")
    else:
        scribe().debug(
            f"[{self.__class__.__name__}] emit called; args={args}, kwargs={kwargs}"
        )


def handle_interrupt(self, args, **kwargs):
    rt_session_id = id(self)
    trace = get_session_store().get_current_trace_from_rt_session_id(rt_session_id)
    if trace is None:
        return
    trace.event(id=str(uuid4()), name="Interrupt", tags={"type": "interrupt"})


def pre_hook(self, hook_name, args, kwargs):
    try:
        if hook_name == "emit":
            intercept_realtime_session_emit(self, *args, **kwargs)
        elif hook_name == "interrupt":
            handle_interrupt(self, *args, **kwargs)
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
