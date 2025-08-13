import asyncio
import functools
import inspect
import traceback
from typing import Optional

from livekit.agents.tts import TTS, ChunkedStream, SynthesizedAudio
from livekit.agents.utils.aio import Chan

from ...scribe import scribe
from .store import get_session_store, get_tts_store
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


def handle_tts_result(self: TTS, result: Optional[ChunkedStream]) -> Optional[bytes]:
    """Handle TTS synthesis result with audio output"""
    if result is None:
        return None

    tts_id = id(self)
    event_ch = result._event_ch

    try:
        target_loop = None
        if hasattr(event_ch, '_loop') and event_ch._loop.is_running():
            target_loop = event_ch._loop

        if target_loop is None:
            return None
        return _run_channel_processing(result, target_loop, tts_id)

    except Exception as e:
        scribe().error(f"[Internal][TTS] Result handling failed: {e!s}\n{traceback.format_exc()}")
        return None

def _run_channel_processing(
        result: ChunkedStream,
        loop: asyncio.AbstractEventLoop, tts_id: int
    ) -> Optional[bytes]:
    """Run channel processing using an existing event loop"""

    try:
        future = asyncio.run_coroutine_threadsafe(
            _process_audio_events(result._event_ch, tts_id),
            loop
        )
        return future.result(timeout=60.0)

    except asyncio.TimeoutError:
        scribe().warning("[Internal][TTS] Processing timed out after 60 seconds")
        future.cancel()
        return None
    except Exception as e:
        scribe().error(f"[Internal][TTS] Async processing failed: {e!s}")
        return None


async def _process_audio_events(
        event_channel: Chan[SynthesizedAudio],
        tts_id: int
    ) -> Optional[bytes]:
    """Process audio events from the TTS stream"""
    event_frames = []

    try:
        async for event in event_channel:
            if event.frame is not None and event.frame.data:
                event_frames.append(event.frame)

        if event_frames and len(event_frames) > 0:
            get_tts_store().add_tts_audio_data(tts_id, event_frames)
        else:
            scribe().warning("[Internal][TTS] No audio data collected from events")
            return None

    except asyncio.CancelledError:
        scribe().debug("[Internal][TTS] Audio processing cancelled")
        raise  # Re-raise to properly handle cancellation
    except Exception as e:
        scribe().error(f"[Internal][TTS] Error processing audio events: {e!s}")
        return None


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
            new_stream = result._tee
            get_thread_pool_executor().submit(handle_tts_result, self, new_stream)
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
