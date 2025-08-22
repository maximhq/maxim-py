import functools
import inspect
from io import BytesIO
import time
import traceback
from uuid import uuid4
import weakref

from livekit.agents.metrics import AgentMetrics 
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.vad import VADEvent, VADEventType
from livekit.plugins.openai.llm import _LLMOptions

from ...scribe import scribe
from ..components import FileDataAttachment
from ..utils import pcm16_to_wav_bytes
from .store import get_session_store, get_maxim_logger
from .utils import extract_llm_model_parameters, get_thread_pool_executor, start_new_turn

agent_activity_f_skip_list = []


def handle_interrupt(self: AgentActivity):
    trace = get_session_store().get_current_trace_for_agent_session(id(self.session))
    if trace is None:
        scribe().warning(
            "[MaximSDK] trace is none at realtime session interrupt. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    trace.event(id=str(uuid4()), name="interrupted")


def post_realtime_session_start(self: AgentActivity):
    # Trying to get AgentSession and RealtimeSession handles
    rt_session_id = id(self.realtime_llm_session)
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self.session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session start. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    session_info.rt_session_id = rt_session_id
    if self.realtime_llm_session is not None:
        session_info.rt_session = weakref.ref(self.realtime_llm_session)
        get_session_store().set_session(session_info)


def post_start(self: AgentActivity):
    if self._rt_session is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] rt_session is none at realtime session start. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    post_realtime_session_start(self)


def handle_input_speech_started(self: AgentActivity):
    scribe().debug(f"[Internal][{self.__class__.__name__}] input speech started")
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self.session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    if session_info.provider == "google-realtime":
        trace = get_session_store().get_current_trace_for_agent_session(
            id(self.session)
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
    scribe().debug(f"[Internal][{self.__class__.__name__}] create speech task")
    if self.agent.session.agent_state != "listening":
        return
    session_info = get_session_store().get_session_by_agent_session_id(
        id(self.session)
    )
    if session_info is None:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
        )
        return
    if session_info.provider == "google-realtime":
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
            # Take only the last 2 seconds of audio from the buffer
            audio_bytes = session_info.current_turn.turn_input_audio_buffer.getvalue()
            sample_rate = getattr(
                session_info, "sample_rate", 16000
            )  # default to 16kHz if not set
            sample_width = getattr(
                session_info, "sample_width", 2
            )  # default to 2 bytes (16-bit PCM) if not set
            channels = getattr(
                session_info, "channels", 1
            )  # default to mono if not set

            bytes_per_second = sample_rate * sample_width * channels
            last_2_sec_bytes = bytes_per_second * 2

            if len(audio_bytes) > last_2_sec_bytes:
                input_buffer = audio_bytes[-last_2_sec_bytes:]
            else:
                input_buffer = audio_bytes
        start_new_turn(session_info)
        if input_buffer is not None and len(input_buffer) > 0:
            session_info = get_session_store().get_session_by_agent_session_id(
                id(self.session)
            )
            if session_info is not None and input_buffer is not None:
                session_info.current_turn.turn_input_audio_buffer.write(input_buffer)
                get_session_store().set_session(session_info)


def handle_vad_inference_done(self: AgentActivity, event: VADEvent):
    """
    This function is called when the VAD inference is done.
    """
    if event.speaking:
        scribe().debug(f"[Internal][{self.__class__.__name__}] VAD inference done")


def handle_final_transcript(self, event):
    """Handle final transcript event and create generation with audio attachments"""
    try:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self.session)
        )
        if session_info is None:
            scribe().warning(
                f"[Internal][{self.__class__.__name__}] session info is none at final transcript. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues."
            )
            return

        turn = session_info.current_turn
        if turn is None:
            scribe().warning(
                f"[Internal][{self.__class__.__name__}] current turn is none at final transcript."
            )
            return

        llm_opts: _LLMOptions = self.llm._opts if self.llm is not None else self._agent.llm._opts
        model = self.llm.model if self.llm is not None else self._agent.llm.model
        if llm_opts is not None:
            model_parameters = extract_llm_model_parameters(llm_opts)
        else:
            model_parameters = None

        # Extract transcript
        transcript = ""
        if hasattr(event, "alternatives") and event.alternatives:
            transcript = event.alternatives[0].text
        elif hasattr(event, "transcript"):
            transcript = event.transcript
        elif hasattr(event, "text"):
            transcript = event.text

        if transcript:
            turn.turn_input_transcription = transcript

            # Create/update generation for this turn
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self.session)
            )
            if trace is not None:
                # Create generation for the conversation turn
                trace.generation(
                    {
                        "id": turn.turn_id,
                        "model": model if model is not None else "unknown",
                        "name": "Conversation Turn",
                        "provider": "livekit",
                        "model_parameters": model_parameters if model_parameters is not None else {},
                        "messages": [{"role": "user", "content": transcript}],
                    }
                )

                # Add input audio attachment if available
                if (
                    turn.turn_input_audio_buffer is not None
                    and turn.turn_input_audio_buffer.tell() > 0
                ):
                    get_maxim_logger().generation_add_attachment(
                        turn.turn_id,
                        FileDataAttachment(
                            data=pcm16_to_wav_bytes(
                                turn.turn_input_audio_buffer.getvalue()
                            ),
                            tags={"attach-to": "input"},
                            name="User Audio Input",
                            timestamp=int(time.time()),
                        ),
                    )

                # Clear the input buffer after attaching the audio data
                turn.turn_input_audio_buffer = None
                trace.set_input(transcript)

            session_info.current_turn = turn
            get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] final transcript handling failed; error={e!s}\n{traceback.format_exc()}"
        )


def handle_agent_response_complete(self, response_text):
    """Handle agent response completion and attach output audio"""
    try:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self.session)
        )
        if session_info is None:
            return

        turn = session_info.current_turn
        if turn is None:
            return

        if response_text:
            turn.turn_output_transcription = response_text

            # Add output audio attachment if we have a generation and audio
            if (
                turn.turn_output_audio_buffer is not None
                and turn.turn_output_audio_buffer.tell() > 0
            ):
                get_maxim_logger().generation_add_attachment(
                    turn.turn_id,
                    FileDataAttachment(
                        data=pcm16_to_wav_bytes(
                            turn.turn_output_audio_buffer.getvalue()
                        ),
                        tags={"attach-to": "output"},
                        name="Agent Audio Response",
                        timestamp=int(time.time()),
                    ),
                )

            # Update trace output
            trace = get_session_store().get_current_trace_for_agent_session(
                id(self.session)
            )
            if trace is not None:
                trace.set_output(response_text)

            session_info.current_turn = turn
            get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] agent response handling failed; error={e!s}\n{traceback.format_exc()}"
        )


def handle_agent_speech_finished(self: AgentActivity):
    """Handle when agent finishes speaking - attach any remaining audio"""
    try:
        session_info = get_session_store().get_session_by_agent_session_id(
            id(self.session)
        )
        if session_info is None:
            return

        turn = session_info.current_turn
        if turn is None:
            return
        # Add output audio attachment if we have audio that hasn't been attached yet
        if (
            turn.turn_output_audio_buffer is not None
            and turn.turn_output_audio_buffer.tell() > 0
        ):
            get_maxim_logger().generation_add_attachment(
                turn.turn_id,
                FileDataAttachment(
                    data=pcm16_to_wav_bytes(turn.turn_output_audio_buffer.getvalue()),
                    tags={"attach-to": "output"},
                    name="Agent Audio Response",
                    timestamp=int(time.time()),
                ),
            )

        session_info.current_turn = turn
        get_session_store().set_session(session_info)

    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] agent speech finished handling failed; error={e!s}\n{traceback.format_exc()}"
        )

def handle_metrics_collected(self: AgentActivity, metrics: AgentMetrics):
    """Handle metrics collected event"""
    if metrics.type != "llm_metrics":
        # Will only get the LLM metrics
        return

    session_info = get_session_store().get_session_by_agent_session_id(
        id(self.session)
    )
    if session_info is None:
        return

    turn = session_info.current_turn
    if turn is None:
        return

    usage = {}
    usage["completion_tokens"] = 0 if metrics.completion_tokens is None else metrics.completion_tokens
    usage["prompt_tokens"] = 0 if metrics.prompt_tokens is None else metrics.prompt_tokens
    usage["total_tokens"] = 0 if metrics.total_tokens is None else metrics.total_tokens

    turn.usage = usage

    
def handle_end_of_speech(self: AgentActivity, event: VADEvent):
    try:
        if event.type != VADEventType.END_OF_SPEECH:
            return

        session_info = get_session_store().get_session_by_agent_session_id(
            id(self.session)
        )
        if session_info is None:
            scribe().warning(
                f"[Internal][{self.__class__.__name__}] session info is none at end of speech."
            )
            return
        
        turn = session_info.current_turn
        if turn is None:
            scribe().warning(
                f"[Internal][{self.__class__.__name__}] current turn is none at end of speech."
            )
            return
        
        frames = event.frames[0] if len(event.frames) > 0 else None
        if frames is None:
            scribe().warning(
                f"[Internal][{self.__class__.__name__}] frames is none at end of speech."
            )
            return
        
        turn = start_new_turn(session_info)
        if turn.turn_input_audio_buffer is None:
            turn.turn_input_audio_buffer = BytesIO()
        turn.turn_input_audio_buffer.write(frames.data)
        session_info.current_turn = turn
        session_info.conversation_buffer.write(frames.data)
        
        if turn.turn_input_audio_buffer is not None and turn.turn_input_audio_buffer.tell() > 0:
            get_maxim_logger().generation_add_attachment(
                turn.turn_id,
                FileDataAttachment(
                    data=pcm16_to_wav_bytes(
                        turn.turn_input_audio_buffer.getvalue()
                    ),
                    tags={"attach-to": "input"},
                    name="User Audio Input",
                    timestamp=int(time.time()),
                ),
            )
        turn.turn_input_audio_buffer = None
        get_session_store().set_session(session_info)
    except Exception as e:
        scribe().warning(
            f"[Internal][{self.__class__.__name__}] handle_end_of_speech failed in function; error={e!s}\n{traceback.format_exc()}"
        )

def pre_hook(self, hook_name, args, kwargs):
    ignored_hooks = ["push_audio"]
    try:
        if hook_name == "interrupt":
            get_thread_pool_executor().submit(handle_interrupt, self)
        elif hook_name == "_on_input_speech_started":
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] agent speech started handling"
            )
            get_thread_pool_executor().submit(handle_input_speech_started, self)
        elif hook_name == "_create_speech_task":
            scribe().debug(
                f"[Internal][{self.__class__.__name__}] agent speech task creation handling {args}"
            )
            get_thread_pool_executor().submit(handle_create_speech_task, self)
        elif hook_name == "on_end_of_speech":
            try:
                get_thread_pool_executor().submit(handle_end_of_speech, self, args[0])
            except Exception as e:
                scribe().warning(
                    f"[Internal][{self.__class__.__name__}] handle_end_of_speech failed; error={e!s}\n{traceback.format_exc()}"
                )
        elif hook_name == "_on_metrics_collected":
            get_thread_pool_executor().submit(handle_metrics_collected, self, args[0])
        elif hook_name == "on_vad_inference_done":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(handle_vad_inference_done, self, args[0])
        elif hook_name == "on_final_transcript":
            if not args or len(args) == 0:
                return
            get_thread_pool_executor().submit(handle_final_transcript, self, args[0])
        elif hook_name == "_conversation_item_added":
            if args and len(args) > 0:
                item = args[0]
                if (
                    hasattr(item, "role")
                    and item.role == "assistant"
                    and hasattr(item, "content")
                    and item.content
                ):
                    content = (
                        item.content[0]
                        if isinstance(item.content, list)
                        else str(item.content)
                    )
                    get_thread_pool_executor().submit(
                        handle_agent_response_complete, self, content
                    )
        elif hook_name == "_on_output_speech_ended":
            # When agent finishes speaking, attach any buffered audio
            get_thread_pool_executor().submit(handle_agent_speech_finished, self)
        elif hook_name == "_generate_reply":
            pass
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
