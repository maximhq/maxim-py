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

tts_f_skip_list = []


def handle_tts_text_input(self, text):
    """Handle TTS text input - store transcription to current turn (created by start_new_turn)"""
    try:
        # Get session info from agent session
        session_info = None
        if hasattr(self, "_session") and self._session is not None:
            session_info = get_session_store().get_session_by_agent_session_id(
                id(self._session)
            )

        if session_info is None:
            scribe().debug("[Internal][TTS] No session info found for text input")
            return

        turn = session_info.current_turn
        if turn is None:
            # TTS should happen after a turn is established by start_new_turn
            scribe().debug(
                "[Internal][TTS] No current turn for text input - turn should be created by start_new_turn"
            )
            return

        # Store output transcription to the turn created by start_new_turn
        if text:
            turn.turn_output_transcription = text

        session_info.current_turn = turn
        get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][TTS] Text input handling failed: {e!s}\n{traceback.format_exc()}"
        )


def handle_tts_result(self, result):
    """Handle TTS synthesis result with audio output"""
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

        # Extract audio data from result
        audio_data = None
        if hasattr(result, "data"):
            audio_data = result.data
        elif hasattr(result, "audio"):
            audio_data = result.audio
        elif hasattr(result, "content"):
            audio_data = result.content

        if audio_data:
            # Buffer audio output
            if turn.turn_output_audio_buffer is None:
                turn.turn_output_audio_buffer = BytesIO()
            turn.turn_output_audio_buffer.write(audio_data)
            session_info.conversation_buffer.write(audio_data)

            # Audio attachment will be handled in agent_activity.py when turn completes

            session_info.current_turn = turn
            get_session_store().set_session(session_info)

            # Update trace output if available
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self._session)
            )
            if trace is not None and turn.turn_output_transcription:
                trace.set_output(turn.turn_output_transcription)

    except Exception as e:
        scribe().warning(
            f"[Internal][TTS] Result handling failed: {e!s}\n{traceback.format_exc()}"
        )


def pre_hook(self, hook_name, args, kwargs):
    """Pre-hook for TTS methods"""
    try:
        if hook_name in ["synthesize", "asynthesize", "speak"]:
            # Handle text input for synthesis
            text = args[0] if args else kwargs.get("text", "")
            get_thread_pool_executor().submit(handle_tts_text_input, self, text)
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} pre-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self, result, hook_name, args, kwargs):
    """Post-hook for TTS methods"""
    try:
        if hook_name in ["synthesize", "asynthesize", "speak"]:
            # Handle synthesis result
            get_thread_pool_executor().submit(handle_tts_result, self, result)
        else:
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] {hook_name} completed; result_type={type(result).__name__}"
            )
    except Exception as e:
        scribe().warning(
            f"[{self.__class__.__name__}] {hook_name} post-hook failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_tts_init(orig, name, class_name):
    """Special instrumentation for TTS __init__ method"""
    if name in tts_f_skip_list:
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


def instrument_tts(orig, name):
    """General instrumentation for TTS methods"""
    if name in tts_f_skip_list:
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
