import base64
import time
from io import BytesIO
from typing import Any, List, Union

from livekit.agents.llm import InputTranscriptionCompleted

from maxim.logger.components import (
    AudioContent,
    FileDataAttachment,
    GenerationResult,
    GenerationResultChoice,
    ImageContent,
    TextContent,
)
from maxim.logger.utils import pcm16_to_wav_bytes
from maxim.logger.livekit.store import SessionStoreEntry, get_maxim_logger, get_session_store
from maxim.logger.livekit.openai.realtime.events import SessionCreatedEvent, get_model_params


def handle_session_created(session_info: SessionStoreEntry, event: SessionCreatedEvent):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    if event["session"] is not None:
        session_info.llm_config = event["session"]  # type: ignore
    session_info.provider = "openai-realtime"
    # saving back the session
    get_session_store().set_session(session_info)
    # creating the generation
    trace_id = session_info.mx_current_trace_id
    if trace_id is None:
        return
    trace = get_maxim_logger().trace({"id": trace_id})
    turn = session_info.current_turn
    if turn is None:
        return
    llm_config = session_info.llm_config
    system_prompt = ""
    if llm_config is not None:
        system_prompt = llm_config["instructions"]
    trace.generation(
        {
            "id": turn.turn_id,
            "model": event["session"]["model"],
            "name": "LLM call",
            "provider": "openai",
            "model_parameters": get_model_params(event["session"]),
            "messages": [{"role": "system", "content": system_prompt}],
        }
    )


def handle_openai_client_event_queued(session_info: SessionStoreEntry, event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI client.
    """
    event_type = event.get("type")
    if event_type == "input_audio_buffer.append":
        if session_info.current_turn is None:
            return
        turn = session_info.current_turn
        if turn.turn_input_audio_buffer is None:
            turn.turn_input_audio_buffer = BytesIO()
        decoded_audio = None
        try:
            decoded_audio = base64.b64decode(event["audio"])
        except Exception:
            pass
        if decoded_audio is not None:
            turn.turn_input_audio_buffer.write(decoded_audio)
        # Here we are buffering the session level audio buffer first
        if (
            session_info.conversation_buffer.tell() + len(event["audio"])
            > 10 * 1024 * 1024
        ):
            session_id = session_info.mx_session_id
            index = session_info.conversation_buffer_index
            get_maxim_logger().session_add_attachment(
                session_id,
                FileDataAttachment(
                    data=pcm16_to_wav_bytes(
                        session_info.conversation_buffer.getvalue()
                    ),
                    tags={"attach-to": "session"},
                    name=f"Conversation part {index}",
                    timestamp=int(time.time()),
                ),
            )
            session_info.conversation_buffer = BytesIO()
            session_info.conversation_buffer_index = index + 1
        if decoded_audio is not None:
            session_info.conversation_buffer.write(decoded_audio)
        session_info.current_turn = turn
        get_session_store().set_session(session_info)


def buffer_audio(entry: SessionStoreEntry, event):
    # Buffering audio to the current session_entry
    if entry.current_turn is None:
        return
    turn = entry.current_turn
    if turn.turn_output_audio_buffer is None:
        turn.turn_output_audio_buffer = BytesIO()
    turn.turn_output_audio_buffer.write(base64.b64decode(event["delta"]))
    entry.current_turn = turn
    entry.conversation_buffer.write(base64.b64decode(event["delta"]))
    get_session_store().set_session(entry)


def handle_openai_input_transcription_completed(
    session_info: SessionStoreEntry, event: InputTranscriptionCompleted
):
    # adding a new generation to the current trace
    trace = get_session_store().get_current_trace_from_rt_session_id(
        session_info.rt_session_id
    )
    if trace is None:
        return
    # adding a new generation
    turn = session_info.current_turn
    if turn is None:
        return
    model = "unknown"

    llm_config = session_info.llm_config
    if llm_config is not None:
        model = llm_config["model"] if llm_config["model"] is not None else "unknown"
    model_parameters = {}
    if llm_config is not None:
        model_parameters = llm_config

    turn.turn_input_transcription += event.transcript
    trace.generation(
        {
            "id": turn.turn_id,
            "model": model,
            "name": "LLM call",
            "provider": "unknown",
            "model_parameters": model_parameters,
            "messages": [{"role": "user", "content": turn.turn_input_transcription}],
        }
    )
    trace.set_input(event.transcript)


def handle_openai_server_event_received(session_info: SessionStoreEntry, event: Any):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    event_type = event.get("type")
    if event_type == "session.created":
        handle_session_created(session_info, event)
    elif event_type == "session.updated":
        pass
    elif event_type == "response.created":
        pass
    elif event_type == "rate_limits.updated":
        # fire as an event
        pass
    elif event_type == "response.output_item.added":
        # response of the LLM call
        pass
    elif event_type == "conversation.item.created":
        pass
    elif event_type == "response.content_part.added":
        pass
    elif event_type == "response.audio_transcript.delta":
        # we can skip this as at the end it gives the entire transcript
        # and we can use that
        pass
    elif event_type == "response.audio.delta":
        # buffer this audio data against the response id
        # use index as well
        buffer_audio(session_info, event)
    elif event_type == "response.audio_transcript.done":
        pass
    elif event_type == "response.output_item.done":
        pass
    elif event_type == "response.content_part.done":
        pass
    elif event_type == "response.done":
        # compute tokens
        # push audio buffer data to the server
        # mark the LLM call complete
        # Attaching the audio buffer as attachment to the generation
        turn = session_info.current_turn
        if turn is None:
            return
        get_maxim_logger().generation_add_attachment(
            turn.turn_id,
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn.turn_input_audio_buffer.getvalue()),
                tags={"attach-to": "input"},
                name="User Input",
                timestamp=int(time.time()),
            ),
        )
        get_maxim_logger().generation_add_attachment(
            turn.turn_id,
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn.turn_output_audio_buffer.getvalue()),
                tags={"attach-to": "output"},
                name="Assistant Response",
                timestamp=int(time.time()),
            ),
        )
        response = event["response"]
        # Adding result to the generation
        usage = response["usage"]
        choices: List[GenerationResultChoice] = []
        if session_info.llm_config is not None:
            model = (
                session_info.llm_config["model"]
                if session_info.llm_config["model"] is not None
                else "unknown"
            )
        else:
            model = "unknown"
        for index, output in enumerate(response["output"]):
            contents: List[Union[TextContent, ImageContent, AudioContent]] = []
            for content in output["content"]:
                if content is None:
                    continue
                if "type" in content and content["type"] == "audio":
                    contents.append(
                        {
                            "type": "audio",
                            "transcript": content["transcript"],
                        }
                    )

            choice: GenerationResultChoice = {
                "index": index,
                "finish_reason": (
                    response["status"] if response["status"] is not None else "stop"
                ),
                "logprobs": None,
                "message": {
                    "role": "assistant",
                    "content": contents,
                    "tool_calls": [],
                },
            }

            choices.append(choice)
        get_maxim_logger().generation_set_provider(turn.turn_id, "openai")
        result: GenerationResult = {
            "id": response["id"],
            "object": response["object"],
            "created": int(time.time()),
            "model": model,
            "usage": {
                "completion_tokens": usage.get("output_tokens", 0),
                "prompt_tokens": usage.get("input_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "input_token_details": {
                    "text_tokens": usage.get("input_token_details", {}).get(
                        "text_tokens", 0
                    ),
                    "audio_tokens": usage.get("input_token_details", {}).get(
                        "audio_tokens", 0
                    ),
                    "cached_tokens": usage.get("input_token_details", {}).get(
                        "cached_tokens", 0
                    ),
                },
                "output_token_details": {
                    "text_tokens": usage.get("output_token_details", {}).get(
                        "text_tokens", 0
                    ),
                    "audio_tokens": usage.get("output_token_details", {}).get(
                        "audio_tokens", 0
                    ),
                    "cached_tokens": usage.get("output_token_details", {}).get(
                        "cached_tokens", 0
                    ),
                },
                "cached_token_details": {
                    "text_tokens": usage.get("cached_token_details", {}).get(
                        "text_tokens", 0
                    ),
                    "audio_tokens": usage.get("cached_token_details", {}).get(
                        "audio_tokens", 0
                    ),
                    "cached_tokens": usage.get("cached_token_details", {}).get(
                        "cached_tokens", 0
                    ),
                },
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
