import functools
import inspect
import time
import traceback
from typing import Union

from google.genai.types import (
    LiveConnectConfig,
    LiveServerContent,
    UsageMetadata,
)
from livekit.agents.llm import InputTranscriptionCompleted
from livekit.plugins.google.beta.realtime.realtime_api import RealtimeSession
from livekit.rtc import AudioFrame

from ....scribe import scribe
from ...components import (
    AudioContent,
    FileDataAttachment,
    GenerationResult,
    GenerationResultChoice,
    ImageContent,
    TextContent,
)
from ...utils import pcm16_to_wav_bytes
from ..store import (
    SessionStoreEntry,
    get_maxim_logger,
    get_session_store,
)


def handle_build_connect_config_result(
    self: RealtimeSession, result: LiveConnectConfig
):
    # here we will start the new generation
    # this is same as session started
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    session_info["provider"] = "google-realtime"
    llm_config = result.generation_config.model_dump()
    llm_config["speech_config"] = result.speech_config.model_dump()
    llm_config["model"] = self._opts.model
    # checking if there are any tools, add adding them in the llm_config
    if result.tools is not None and len(result.tools) > 0:
        llm_config["tools"] = []
        # Third party tools
        for tool in result.tools:
            for func_declaration in tool.function_declarations:
                llm_config["tools"].append(
                    {
                        "name": func_declaration.name,
                        "description": func_declaration.description,
                        "parameters": func_declaration.parameters,
                    }
                )
    # saving back llm_config
    session_info["llm_config"] = llm_config
    # saving back the session
    get_session_store().set_session(session_info)
    trace_id = session_info["mx_current_trace_id"]
    if trace_id is None:
        return
    trace = get_maxim_logger().trace({"id": trace_id})
    turn = session_info["current_turn"]
    if turn is None:
        return
    system_prompt = ""
    if result.system_instruction is not None:
        # We have to iterate through the system instructions
        for part in result.system_instruction.parts:
            system_prompt += part.text
    trace.generation(
        {
            "id": turn["turn_id"],
            "model": self._opts.model,
            "name": "LLM call",
            "provider": "google",
            "model_parameters": llm_config,
            "messages": [{"role": "system", "content": system_prompt}],
        }
    )
    session_info["user_speaking"] = False
    get_session_store().set_session(session_info)


def handle_server_content(self: RealtimeSession, content: LiveServerContent):
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    turn = session_info["current_turn"]
    if turn is None:
        return
    if content.output_transcription is not None:
        if session_info["user_speaking"]:
            session_info["user_speaking"] = False
            get_session_store().set_session(session_info)
    # adding transcription
    if content.output_transcription is not None:
        turn["turn_output_transcription"] += content.output_transcription.text
    # getting audio bytes from the payload
    if content.model_turn is not None:
        if content.model_turn.parts is not None and len(content.model_turn.parts) > 0:
            for part in content.model_turn.parts:
                turn["turn_output_audio_buffer"] += part.inline_data.data
                if (
                    len(session_info["conversation_buffer"])
                    + len(part.inline_data.data)
                    > 10 * 1024 * 1024
                ):
                    session_id = session_info["mx_session_id"]
                    index = session_info["conversation_buffer_index"]
                    get_maxim_logger().session_add_attachment(
                        session_id,
                        FileDataAttachment(
                            data=pcm16_to_wav_bytes(
                                session_info["conversation_buffer"]
                            ),
                            tags={"attach-to": "input"},
                            name=f"Conversation part {index}",
                            timestamp=int(time.time()),
                        ),
                    )
                    session_info["conversation_buffer"] = b""
                    session_info["conversation_buffer_index"] = index + 1
                session_info["conversation_buffer"] += part.inline_data.data
    session_info["current_turn"] = turn
    get_session_store().set_session(session_info)


def handle_google_input_transcription_completed(
    session_info: SessionStoreEntry, input_transcription: InputTranscriptionCompleted
):
    turn = session_info["current_turn"]
    if turn is None:
        return
    turn["turn_input_transcription"] = input_transcription.transcript
    get_session_store().set_session(session_info)


def handle_usage_metadata(self: RealtimeSession, usage: UsageMetadata):
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    turn = session_info["current_turn"]
    if turn is None:
        return
    # Writing final Generation result
    if (
        turn["turn_output_audio_buffer"] is not None
        and len(turn["turn_output_audio_buffer"]) > 0
    ):
        get_maxim_logger().generation_add_attachment(
            turn["turn_id"],
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn["turn_output_audio_buffer"]),
                tags={"attach-to": "output"},
                name="Assistant Response",
                timestamp=int(time.time()),
            ),
        )
    if (
        turn["turn_input_audio_buffer"] is not None
        and len(turn["turn_input_audio_buffer"]) > 0
    ):
        get_maxim_logger().generation_add_attachment(
            turn["turn_id"],
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn["turn_input_audio_buffer"]),
                tags={"attach-to": "input"},
                name="User Input",
                timestamp=int(time.time()),
            ),
        )
        session_info["current_turn"]["turn_input_audio_buffer"] = b""
    contents: list[Union[TextContent, ImageContent, AudioContent]] = []
    contents.append({"type": "audio", "transcript": turn["turn_output_transcription"]})
    choices: list[GenerationResultChoice] = []
    choices.append(
        {
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": contents, "tool_calls": []},
        }
    )
    get_maxim_logger().generation_set_provider(turn["turn_id"], "google")
    # Parsing token details
    input_token_details: dict[str, int] = {}
    output_token_details: dict[str, int] = {}
    cached_token_details: dict[str, int] = {}
    if usage.prompt_tokens_details is not None:
        for detail in usage.prompt_tokens_details:
            if detail.modality == "TEXT":
                input_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO":
                input_token_details["audio_tokens"] = detail.token_count
    if usage.response_tokens_details is not None:
        for detail in usage.response_tokens_details:
            if detail.modality == "TEXT":
                output_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO":
                output_token_details["audio_tokens"] = detail.token_count
    if usage.cache_tokens_details is not None:
        for detail in usage.cache_tokens_details:
            if detail.modality == "TEXT":
                cached_token_details["text_tokens"] = detail.token_count
            elif detail.modality == "AUDIO":
                cached_token_details["audio_tokens"] = detail.token_count
    trace = get_session_store().get_current_trace_from_rt_session_id(
        session_info["rt_session_id"]
    )
    get_maxim_logger().trace_add_generation(
        session_info["mx_current_trace_id"],
        {
            "id": turn["turn_id"],
            "model": self._opts.model,
            "name": "LLM call",
            "provider": "google",
            "model_parameters": session_info["llm_config"],
            "messages": [{"role": "user", "content": turn["turn_input_transcription"]}],
        },
    )
    result: GenerationResult = {
        "id": turn["turn_id"],
        "object": "",
        "created": int(time.time()),
        "model": self._opts.model,
        "usage": {
            "completion_tokens": usage.response_token_count,
            "prompt_tokens": usage.prompt_token_count,
            "total_tokens": usage.total_token_count,
            "input_token_details": input_token_details,
            "output_token_details": output_token_details,
            "cached_token_details": cached_token_details,
        },
        "choices": choices,
    }
    # Setting up the generation
    get_maxim_logger().generation_result(turn["turn_id"], result)
    # Setting the output to the trace
    if session_info["rt_session_id"] is not None:
        trace = get_session_store().get_current_trace_from_rt_session_id(
            session_info["rt_session_id"]
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


def handle_push_audio(self: RealtimeSession, args, kwargs):
    session_info = get_session_store().get_session_by_rt_session_id(id(self))
    if session_info is None:
        scribe().error("[MaximSDK] session info is none at realtime session emit")
        return
    turn = session_info["current_turn"]
    if turn is None:
        return
    if not args or len(args) == 0:
        return
    audio_frame: AudioFrame = args[0]
    # This will help us skip silence before the turn starts
    if len(audio_frame.data) == 0 and len(turn["turn_input_audio_buffer"]) == 0:
        return
    # Checking if the audio buffer is mostly silence
    turn["turn_input_audio_buffer"] += memoryview(audio_frame._data).cast("B")
    session_info["conversation_buffer"] += memoryview(audio_frame._data).cast("B")
    session_info["current_turn"] = turn
    get_session_store().set_session(session_info)


ignored_hooks = ["_resample_audio", "_send_client_event", "_start_new_generation"]


def pre_hook(self: RealtimeSession, hook_name, args, kwargs):
    try:
        if hook_name in ignored_hooks:
            return
        elif hook_name == "_handle_server_content":
            handle_server_content(self, args[0])
        elif hook_name == "_handle_usage_metadata":
            handle_usage_metadata(self, args[0])
        elif hook_name == "push_audio":
            handle_push_audio(self, args, kwargs)
        elif hook_name == "_mark_current_generation_done":
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] _mark_current_generation_done called; args={args}, kwargs={kwargs}"
            )
        else:
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} called; args={args}, kwargs={kwargs}"
            )
    except Exception as e:
        scribe().error(
            f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def post_hook(self: RealtimeSession, result, hook_name, args, kwargs):
    try:
        if hook_name in ignored_hooks:
            return
        elif hook_name == "_resample_audio":
            pass
        elif hook_name == "push_audio":
            pass
        elif hook_name == "_send_client_event":
            pass
        elif hook_name == "_send_task":
            pass
        elif hook_name == "_start_new_generation":
            pass
        elif hook_name == "_handle_server_content":
            # this has audio data but post_hook is not required
            pass
        elif hook_name == "_handle_usage_metadata":
            # this has usage data
            pass
        elif hook_name == "_build_connect_config":
            handle_build_connect_config_result(self, result)
        else:
            scribe().debug(
                f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} completed; result={result}"
            )
    except Exception as e:
        scribe().error(
            f"[Internal][Gemini:{self.__class__.__name__}] {hook_name} failed; error={str(e)}\n{traceback.format_exc()}"
        )


def instrument_gemini_session(orig, name):
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
