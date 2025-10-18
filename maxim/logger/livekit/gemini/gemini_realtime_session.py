"""Gemini Realtime Session instrumentation and handlers.

This module wires into Google Gemini's Realtime SDK (`RealtimeSession`) by
wrapping selected lifecycle methods with pre/post hooks and by handling SDK
callbacks. The goal is to:

- Capture model inputs/outputs (audio and text) and persist them to Maxim logs
    as turns and generations.
- Record token usage and provider metadata for cost/latency analysis.
- Surface tool-call activity as structured trace events.
- Buffer long-running, streaming audio without blocking the realtime thread.

When and why this module is used
--------------------------------
- It is activated by `instrument_gemini()` (see `instrumenter.py`), which
    monkey-patches the `RealtimeSession` methods to call `pre_hook`/`post_hook`.
- `pre_hook` dispatches heavy logging work to a background thread pool so that
    realtime audio and events keep flowing smoothly.
- `post_hook` captures final results after specific SDK calls complete (e.g.,
    `_build_connect_config`).

This decoupled design ensures accurate logging and tracing while minimizing
impact on the realtime media path.
"""

import copy
import functools
import inspect
import json
import time
import traceback
from io import BytesIO
from typing import Union

try:
    from google.genai.types import (
        LiveConnectConfig,
        LiveServerContent,
        LiveServerToolCall,
        UsageMetadata,
        Content,
    )
except ImportError:
    pass
from livekit.agents.llm import InputTranscriptionCompleted
try:
    from livekit.plugins.google.realtime.realtime_api import RealtimeSession
except ImportError:
    from livekit.plugins.google.beta.realtime.realtime_api import RealtimeSession
from livekit.rtc import AudioFrame

from maxim.scribe import scribe
from maxim.logger.components import (
    AudioContent,
    FileDataAttachment,
    GenerationResult,
    GenerationResultChoice,
    ImageContent,
    TextContent,
    TokenDetails,
)
from maxim.logger.utils import pcm16_to_wav_bytes, trim_silence_edges
from maxim.logger.livekit.store import (
    SessionStoreEntry,
    get_maxim_logger,
    get_session_store,
)
from maxim.logger.livekit.utils import get_thread_pool_executor


def handle_build_connect_config_result(
    self: RealtimeSession, result: LiveConnectConfig
):
    """Handle connection configuration after the SDK builds it.

    When: Invoked from `post_hook` after `_build_connect_config` completes.

    Why: Initializes a new Maxim generation with the resolved model,
    system instructions, and tool configuration so that subsequent streaming
    events are associated with the right turn and trace.
    """
    # here we will start the new generation
    # this is same as session started
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().warning(
            "[MaximSDK] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues"
        )
        return
    session_info.provider = "google-realtime"
    llm_config = {}
    if result.generation_config is not None:
        llm_config = result.generation_config.model_dump()
    if result.speech_config is not None:
        for key, value in result.speech_config.model_dump().items():
            if value is not None:
                llm_config[key] = value
    # removing speech_config from the llm_config
    llm_config.pop("speech_config", None)
    llm_config["model"] = self._opts.model
    # checking if there are any tools, add adding them in the llm_config
    if result.tools is not None and len(result.tools) > 0:
        llm_config["tools"] = []
        # Third party tools
        for tool in result.tools:
            if tool.function_declarations is None:
                continue
            for func_declaration in tool.function_declarations:
                if func_declaration is None:
                    continue
                llm_config["tools"].append(
                    {
                        "name": func_declaration.name,
                        "description": func_declaration.description,
                        "parameters": func_declaration.parameters,
                    }
                )
    # saving back llm_config
    session_info.llm_config = llm_config
    # saving back the session
    get_session_store().set_session(session_info)
    trace_id = session_info.mx_current_trace_id
    if trace_id is None:
        return
    trace = get_maxim_logger().trace({"id": trace_id})
    turn = session_info.current_turn
    if turn is None:
        return
    system_prompt = ""
    if (
        result.system_instruction is not None
        and isinstance(result.system_instruction, Content)
        and result.system_instruction.parts is not None
    ):
        if result.system_instruction.parts is not None:
            # We have to iterate through the system instructions
            for part in result.system_instruction.parts:
                if part.text is not None:
                    system_prompt += part.text
    trace.generation(
        {
            "id": turn.turn_id,
            "model": self._opts.model,
            "name": "LLM call",
            "provider": "google",
            "model_parameters": llm_config,
            "messages": [{"role": "system", "content": system_prompt}],
        }
    )
    session_info.user_speaking = False
    session_info.current_turn = turn
    get_session_store().set_session(session_info)


def handle_server_content(self: RealtimeSession, content: LiveServerContent):
    """Process streamed server content from Gemini (text/audio parts).

    When: Dispatched from `pre_hook` upon `_handle_server_content` calls.

    Why: Accumulates assistant output transcription and audio bytes into the
    current turn and a session-level conversation buffer. The conversation
    buffer is periodically flushed to the session as attachments to avoid large
    in-memory growth during long conversations.
    """
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().warning(
            "[MaximSDK] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues"
        )
        return
    turn = session_info.current_turn
    if turn is None:
        return

    if content.output_transcription is not None and session_info.user_speaking:
        session_info.user_speaking = False
        get_session_store().set_session(session_info)
    # adding transcription
    if (
        content.output_transcription is not None
        and content.output_transcription.text is not None
    ):
        turn.turn_output_transcription += content.output_transcription.text
    # getting audio bytes from the payload
    if (
        content.model_turn is not None
        and content.model_turn.parts is not None
        and len(content.model_turn.parts) > 0
    ):
        for part in content.model_turn.parts:
            if part.inline_data is None or part.inline_data.data is None:
                continue
            turn.turn_output_audio_buffer.write(part.inline_data.data)
            if (
                session_info.conversation_buffer.tell() + len(part.inline_data.data)
                > 10 * 1024 * 1024
            ):
                session_id = session_info.mx_session_id
                if session_id is None:
                    return
                index = session_info.conversation_buffer_index
                get_maxim_logger().session_add_attachment(
                    session_id,
                    FileDataAttachment(
                        data=pcm16_to_wav_bytes(
                            session_info.conversation_buffer.getvalue()
                        ),
                        tags={"attach-to": "input"},
                        name=f"Conversation part {index}",
                        timestamp=int(time.time()),
                    ),
                )
                session_info.conversation_buffer = BytesIO()
                session_info.conversation_buffer_index = index + 1
            session_info.conversation_buffer.write(part.inline_data.data)
    session_info.current_turn = turn
    get_session_store().set_session(session_info)


def handle_google_input_transcription_completed(
    session_info: SessionStoreEntry, input_transcription: InputTranscriptionCompleted
):
    """Persist the user's input transcription for the current turn.

    When: Triggered by the base realtime interceptor on
    `input_audio_transcription_completed` events for the Google provider.

    Why: Stores the user's speech-to-text transcript on the active turn so it
    can be included in the generation request and traces.
    """
    turn = session_info.current_turn
    if turn is None:
        return
    turn.turn_input_transcription = input_transcription.transcript
    get_session_store().set_session(session_info)


def handle_usage_metadata(self: RealtimeSession, usage: UsageMetadata):
    """Finalize the generation and log usage/tokens and media attachments.

    When: Dispatched from `pre_hook` upon `_handle_usage_metadata` calls when
    Gemini emits usage statistics for a completed model turn.

    Why: Completes the current generation by attaching input/output audio,
    constructing a `GenerationResult` with token details, updating the active
    trace, and persisting the assistant's transcript as output.
    """
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().warning(
            "[MaximSDK] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues"
        )
        return
    # Here if the interrupted turn is present, we need to use that instead of the current turn
    turn = session_info.current_turn
    if turn is None:
        return
    # Writing final Generation result
    if (
        turn.turn_input_audio_buffer is not None
        and turn.turn_input_audio_buffer.tell() > 0
    ):
        buffer = turn.turn_input_audio_buffer.getvalue()
        # Trim leading and trailing silence (assumes 16kHz mono PCM16)
        buffer = trim_silence_edges(buffer, sample_rate=16000, frame_ms=10, threshold=300)
        get_maxim_logger().generation_add_attachment(
            turn.turn_id,
            FileDataAttachment(
                data=pcm16_to_wav_bytes(buffer),
                tags={"attach-to": "input"},
                name="User Input",
                timestamp=int(time.time()),
            ),
        )
    if (
        turn.turn_output_audio_buffer is not None
        and turn.turn_output_audio_buffer.tell() > 0
    ):
        get_maxim_logger().generation_add_attachment(
            turn.turn_id,
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn.turn_output_audio_buffer.getvalue()),
                tags={"attach-to": "output"},
                name="Assistant Response",
                timestamp=int(time.time()),
            ),
        )
    contents: list[Union[TextContent, ImageContent, AudioContent]] = []
    contents.append({"type": "audio", "transcript": turn.turn_output_transcription})
    choices: list[GenerationResultChoice] = []
    choices.append(
        {
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": contents, "tool_calls": []},
        }
    )
    get_maxim_logger().generation_set_provider(turn.turn_id, "google")
    # Parsing token details
    input_token_details: dict[str, int] = {}
    output_token_details: dict[str, int] = {}
    cached_token_details: dict[str, int] = {}
    if usage.prompt_tokens_details is not None:
        for detail in usage.prompt_tokens_details:
            if detail.modality == "TEXT" and detail.token_count is not None:
                input_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO" and detail.token_count is not None:
                input_token_details["audio_tokens"] = detail.token_count
    if usage.response_tokens_details is not None:
        for detail in usage.response_tokens_details:
            if detail.modality == "TEXT" and detail.token_count is not None:
                output_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO" and detail.token_count is not None:
                output_token_details["audio_tokens"] = detail.token_count
    if usage.cache_tokens_details is not None:
        for detail in usage.cache_tokens_details:
            if detail.modality == "TEXT" and detail.token_count is not None:
                cached_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO" and detail.token_count is not None:
                cached_token_details["audio_tokens"] = detail.token_count
    if session_info.rt_session_id is None or session_info.mx_current_trace_id is None:
        return
    trace = get_session_store().get_current_trace_from_rt_session_id(
        session_info.rt_session_id
    )
    get_maxim_logger().trace_add_generation(
        session_info.mx_current_trace_id,
        {
            "id": turn.turn_id,
            "model": self._opts.model,
            "name": "LLM call",
            "provider": "google",
            "messages": [{"role": "user", "content": turn.turn_input_transcription}],
            "model_parameters": session_info.llm_config,  # type: ignore
        },
    )
    result: GenerationResult = {
        "id": turn.turn_id,
        "object": "",
        "created": int(time.time()),
        "model": self._opts.model,
        "usage": {
            "completion_tokens": (
                usage.response_token_count
                if usage.response_token_count is not None
                else 0
            ),
            "prompt_tokens": (
                usage.prompt_token_count if usage.prompt_token_count is not None else 0
            ),
            "total_tokens": (
                usage.total_token_count if usage.total_token_count is not None else 0
            ),
            "input_token_details": TokenDetails(**input_token_details),
            "output_token_details": TokenDetails(**output_token_details),
            "cached_token_details": TokenDetails(**cached_token_details),
        },
        "choices": choices,
    }
    # Setting up the generation
    get_maxim_logger().generation_result(turn.turn_id, result)
    # Setting the output to the trace
    if session_info.rt_session_id is not None:
        trace = get_session_store().get_current_trace_from_rt_session_id(
            session_info.rt_session_id
        )
        if (
            trace is not None
            and len(choices) > 0
            and choices[0]["message"]["content"] is not None
            and isinstance(choices[0]["message"]["content"], list)
            and len(choices[0]["message"]["content"]) > 0
            and choices[0]["message"]["content"][0] is not None
            and "transcript" in choices[0]["message"]["content"][0]
        ):
            trace.set_output(choices[0]["message"]["content"][0]["transcript"])


def handle_push_audio(self: RealtimeSession, data_buffer: bytes):
    """Buffer inbound user audio for the current turn and session.

    When: Dispatched from `pre_hook` whenever `push_audio` is called by the
    SDK with a new `AudioFrame`.

    Why: Maintains a precise record of the user's audio input for the turn and
    for the longer session conversation buffer, while skipping leading silence
    so empty buffers are not logged.
    """
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().warning(
            "[MaximSDK] session info is none at realtime session emit. If you are seeing this frequently, please report issue at https://github.com/maximhq/maxim-py/issues"
        )
        return
    turn = session_info.current_turn
    if turn is None:
        return
    # This will help us skip silence before the turn starts
    if (data_buffer is None or len(data_buffer) == 0) and (
        turn.turn_input_audio_buffer is None or turn.turn_input_audio_buffer.tell() == 0
    ):
        return
    # Ensure we have a valid audio buffer
    if turn.turn_input_audio_buffer is None:
        turn.turn_input_audio_buffer = BytesIO()
    pcm_bytes = trim_silence_edges(data_buffer, sample_rate=16000, frame_ms=10, threshold=300)
    turn.turn_input_audio_buffer.write(pcm_bytes)
    session_info.conversation_buffer.write(pcm_bytes)
    session_info.current_turn = turn
    get_session_store().set_session(session_info)


def handle_tool_calls(self, tool_calls: LiveServerToolCall):
    """Record tool calls issued by the model as trace events.

    When: Dispatched from `pre_hook` upon `_handle_tool_calls` callbacks.

    Why: Surfaces model-initiated function/tool invocations in the trace with
    their names and serialized arguments for auditability and debugging.
    """
    trace = get_session_store().get_current_trace_from_rt_session_id(id(self))
    if trace is None:
        return
    if tool_calls is None:
        return
    if tool_calls.function_calls is None:
        return
    for function_call in tool_calls.function_calls:
        if function_call is None:
            continue
        if function_call.id is None:
            continue
        raw_args = function_call.args
        if isinstance(raw_args, (dict, list)):
            raw_args = json.dumps(raw_args)
        trace.tool_call(
            {
                "id": function_call.id,
                "name": (
                    function_call.name
                    if function_call.name is not None
                    else "tool_call"
                ),
                "args": raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
            }
        )


ignored_hooks = ["_resample_audio", "_send_client_event", "_start_new_generation"]


def pre_hook(self: RealtimeSession, hook_name, args, kwargs):
    """Lightweight pre-hook executed before selected SDK methods.

    When: Called by wrappers installed via `instrument_gemini_session`.

    Why: Routes specific realtime callbacks to background workers to avoid
    blocking the audio/media pipeline. Only a subset of hooks produce logging
    work; the rest are emitted as debug messages to aid troubleshooting.
    """
    try:
        if hook_name in ignored_hooks:
            return
        elif hook_name == "_handle_server_content":
            # can we do deep copy of args[0]
            # and then pass it to the thread pool executor
            get_thread_pool_executor().submit(
                handle_server_content, self, copy.deepcopy(args[0])
            )
        elif hook_name == "_handle_tool_calls":
            get_thread_pool_executor().submit(handle_tool_calls, self, args[0])
        elif hook_name == "_handle_usage_metadata":
            get_thread_pool_executor().submit(
                handle_usage_metadata, self, copy.deepcopy(args[0])
            )
        elif hook_name == "push_audio":
            if not args or len(args) == 0:
                return
            audio_frame: AudioFrame = args[0]
            data_buffer = memoryview(audio_frame._data).cast("B")
            get_thread_pool_executor().submit(handle_push_audio, self, data_buffer)
        elif hook_name == "_mark_current_generation_done":
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] _mark_current_generation_done called; args={args}, kwargs={kwargs}"
            )
        else:
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().warning(
            f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def post_hook(self: RealtimeSession, result, hook_name, args, kwargs):
    """Post-hook executed after selected SDK methods complete.

    When: Called by wrappers installed via `instrument_gemini_session`.

    Why: Performs completion-time work for particular hooks (e.g., capturing
    model configuration after `_build_connect_config`) and emits debug details
    for others. Expensive actions are avoided here to keep teardown fast.
    """
    try:
        if hook_name in ignored_hooks:
            return
        if hook_name in (
            "_resample_audio",
            "push_audio",
            "_send_client_event",
            "_send_task",
            "_start_new_generation",
            "_handle_server_content",
            "_handle_usage_metadata",
        ):
            return
        if hook_name == "_build_connect_config":
            handle_build_connect_config_result(self, result)
        else:
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().warning(
            f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} failed; error={e!s}\n{traceback.format_exc()}"
        )


def instrument_gemini_session(orig, name):
    """Wrap a `RealtimeSession` method with pre/post hooks.

    When: Used during instrumentation to monkey-patch every callable attribute
    on `RealtimeSession` (except dunders) with a wrapper that executes
    `pre_hook` and `post_hook` around the original method.

    Why: Centralizes logging/observability without modifying the upstream SDK
    implementation. Supports both sync and async methods.
    """
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
