import functools
import inspect
import time
import traceback
from io import BytesIO
from uuid import uuid4

from ...scribe import scribe
from ..components import FileDataAttachment
from ..utils import pcm16_to_wav_bytes
from .store import get_session_store, get_maxim_logger
from .utils import get_thread_pool_executor

stt_f_skip_list = []


def handle_stt_transcribe(self, audio_frames, language=None):
    """Handle STT transcription with audio buffering"""
    try:
        # Get session info from agent session
        session_info = None
        if hasattr(self, "_session") and self._session is not None:
            session_info = get_session_store().get_session_by_agent_session_id(
                id(self._session)
            )

        if session_info is None:
            scribe().debug(
                "[Internal][STT] No session info found for STT transcription"
            )
            return

        turn = session_info.current_turn
        if turn is None:
            scribe().debug("[Internal][STT] No current turn for STT transcription")
            return

        # Buffer audio frames
        if audio_frames is not None:
            for frame in audio_frames:
                if hasattr(frame, "data"):
                    if turn.turn_input_audio_buffer is None:
                        turn.turn_input_audio_buffer = BytesIO()
                    turn.turn_input_audio_buffer.write(frame.data)
                    session_info.conversation_buffer.write(frame.data)

        # Create generation for STT if we have audio
        if (
            turn.turn_input_audio_buffer is not None
            and turn.turn_input_audio_buffer.tell() > 0
        ):
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self._session)
            )
            if trace is not None:
                # Create STT generation
                stt_gen_id = str(uuid4())
                trace.generation(
                    {
                        "id": stt_gen_id,
                        "model": getattr(self, "_model", "unknown-stt"),
                        "name": "STT Transcription",
                        "provider": "openai",
                        "model_parameters": {"language": language} if language else {},
                        "messages": [{"role": "user", "content": "[Audio Input]"}],
                    }
                )

                # Add audio attachment to generation
                get_maxim_logger().generation_add_attachment(
                    stt_gen_id,
                    FileDataAttachment(
                        data=pcm16_to_wav_bytes(
                            turn.turn_input_audio_buffer.getvalue()
                        ),
                        tags={"attach-to": "input"},
                        name="STT Audio Input",
                        timestamp=int(time.time()),
                    ),
                )

        session_info.current_turn = turn
        get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][STT] Audio handling failed: {e!s}\n{traceback.format_exc()}"
        )


def handle_stt_result(self, result, generation_id=None):
    """Handle STT transcription result"""
    try:
        if result is None:
            return

        # Get session info from agent session
        session_info = None
        if hasattr(self, "_session") and self._session is not None:
            session_info = get_session_store().get_session_by_agent_session_id(
                id(self._session)
            )

        if session_info is None:
            return

        turn = session_info.current_turn
        if turn is None:
            return

        # Extract transcript from result
        transcript = ""
        if hasattr(result, "transcript"):
            transcript = result.transcript
        elif hasattr(result, "text"):
            transcript = result.text
        elif isinstance(result, str):
            transcript = result

        if transcript:
            turn.turn_input_transcription = transcript
            session_info.current_turn = turn
            get_session_store().set_session(session_info)

            # Update trace input if available
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self._session)
            )
            if trace is not None:
                trace.set_input(transcript)

    except Exception as e:
        scribe().warning(
            f"[Internal][STT] Result handling failed: {e!s}\n{traceback.format_exc()}"
        )


def pre_hook(self, hook_name, args, kwargs):
    """Pre-hook for STT methods"""
    try:
        if hook_name in ["transcribe", "arecognize", "recognize"]:
            # Handle audio input for transcription
            audio_frames = args[0] if args else None
            language = kwargs.get("language") or (args[1] if len(args) > 1 else None)
            get_thread_pool_executor().submit(
                handle_stt_transcribe, self, audio_frames, language
            )
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} pre-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    """Post-hook for STT methods"""
    try:
        if hook_name in ["transcribe", "arecognize", "recognize"]:
            # Handle transcription result
            get_thread_pool_executor().submit(handle_stt_result, self, result)
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} completed; result_type={type(result).__name__}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} post-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_stt_init(orig, name, class_name):
    """Special instrumentation for STT __init__ method"""
    if name in stt_f_skip_list:
        return orig

    def wrapper(self, *args, **kwargs):
        pre_hook(self, name, args, kwargs)
        result = None
        try:
            result = orig(self, *args, **kwargs)
            scribe().debug(f"[Internal][{class_name}] initialized")
            return result
        except Exception as e:
            scribe().warning(
                f"[Internal][{class_name}] {name} failed; error={e!s}\n{traceback.format_exc()}"
            )
            raise
        finally:
            post_hook(self, result, name, args, kwargs)

    return functools.wraps(orig)(wrapper)


def instrument_stt(orig, name):
    """General instrumentation for STT methods"""
    if name in stt_f_skip_list:
        return orig

    if inspect.iscoroutinefunction(orig):

        async def async_wrapper(self, *args, **kwargs):
            pre_hook(self, name, args, kwargs)
            result = None
            try:
                result = await orig(self, *args, **kwargs)
                return result
            except Exception as e:
                scribe().warning(
                    f"[Internal][{self.__class__.__name__}] {name} failed; error={e!s}\n{traceback.format_exc()}"
                )
                raise
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
                scribe().warning(
                    f"[Internal][{self.__class__.__name__}] {name} failed; error={e!s}\n{traceback.format_exc()}"
                )
                raise
            finally:
                post_hook(self, result, name, args, kwargs)

        wrapper = sync_wrapper
    return functools.wraps(orig)(wrapper)
