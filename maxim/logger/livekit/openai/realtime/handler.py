import base64
import io
import time
import wave
from typing import Any, List, Union

from ....components import (
    AudioContent,
    FileDataAttachment,
    GenerationResult,
    GenerationResultChoice,
    ImageContent,
    TextContent,
)
from ...store import SessionStoreEntry, get_maxim_logger, get_session_store
from .events import SessionCreatedEvent, get_model_params


def pcm16_to_wav_bytes(
    pcm_bytes: bytes, num_channels: int = 1, sample_rate: int = 24000
) -> bytes:
    """
    Convert PCM-16 audio data to WAV format bytes.

    Args:
        pcm_bytes (bytes): Raw PCM-16 audio data
        num_channels (int): Number of audio channels (default: 2)
        sample_rate (int): Sample rate in Hz (default: 44100)

    Returns:
        bytes: WAV format audio data
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wavfile:
        wavfile.setnchannels(num_channels)
        wavfile.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wavfile.setframerate(sample_rate)
        wavfile.writeframes(pcm_bytes)
    return buffer.getvalue()


def handle_session_created(session_info: SessionStoreEntry, event: SessionCreatedEvent):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    session_info["llm_config"] = event["session"]
    # saving back the session
    get_session_store().set_session(session_info)
    # creating the generation
    trace_id = session_info["mx_current_trace_id"]
    if trace_id is None:
        return
    trace = get_maxim_logger().trace({"id": trace_id})
    turn = session_info["current_turn"]
    if turn is None:
        return
    llm_config = session_info["llm_config"]
    system_prompt = ""
    if llm_config is not None:
        system_prompt = llm_config["instructions"]
    trace.generation(
        {
            "id": turn["turn_id"],
            "model": event["session"]["model"],
            "provider": "openai",
            "model_parameters": get_model_params(event["session"]),
            "messages": [{"role": "system", "content": system_prompt}],
        }
    )


def handle_openai_client_event_queued(session_info: SessionStoreEntry, event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI client.
    """
    # We ignore buffer audio type as it just fills it with silence`
    type = event.get("type")
    if type == "input_audio_buffer.append":
        return
    print(f"OpenAI client event queued:type:{type} {event}")


def buffer_audio(entry: SessionStoreEntry, event):
    # Buffering audio to the current session_entry
    if entry["current_turn"] is None:
        return
    print("#############BUFFERING AUDIO############")
    turn = entry["current_turn"]
    if turn["turn_audio_buffer"] is None:
        turn["turn_audio_buffer"] = b""
    turn["turn_audio_buffer"] += base64.b64decode(event["delta"])
    entry["current_turn"] = turn
    get_session_store().set_session(entry)


def handle_openai_server_event_received(session_info: SessionStoreEntry, event: Any):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    type = event.get("type")
    if type == "session.created":
        handle_session_created(session_info, event)
    elif type == "session.updated":
        pass
    elif type == "response.created":
        pass
    elif type == "rate_limits.updated":
        # fire as an event
        pass
    elif type == "response.output_item.added":
        # response of the LLM call
        pass
    elif type == "conversation.item.created":
        pass
    elif type == "response.content_part.added":
        pass
    elif type == "response.audio_transcript.delta":
        # we can skip this as at the end it gives the entire transcript
        # and we can use that
        pass
    elif type == "response.audio.delta":
        # buffer this audio data against the response id
        # use index as well
        buffer_audio(session_info, event)
    elif type == "response.audio_transcript.done":
        pass
    elif type == "response.output_item.done":
        pass
    elif type == "response.content_part.done":
        pass
    elif type == "response.done":
        # compute tokens
        # push audio buffer data to the server
        # mark the LLM call complete
        print(f"#############LLM CALL COMPLETE############ {event}")
        # Attaching the audio buffer as attachment to the generation
        turn = session_info["current_turn"]
        if turn is None:
            return
        get_maxim_logger().generation_add_attachment(
            turn["turn_id"],
            FileDataAttachment(
                data=pcm16_to_wav_bytes(turn["turn_audio_buffer"]),
                tags={"attach-to": "output"},
                name="Assistant Response",
                timestamp=int(time.time()),
                metadata={"attach-to": "output"},
            ),
        )
        response = event["response"]
        # Adding result to the generation
        usage = response["usage"]
        choices: List[GenerationResultChoice] = []
        if session_info["llm_config"] is not None:
            model = session_info["llm_config"]["model"]
        else:
            model = "unknown"
        for index, output in enumerate(response["output"]):
            print(f"#############OUTPUT {output}############")
            contents: List[Union[TextContent, ImageContent, AudioContent]] = []
            for content in output["content"]:
                if content is None:
                    return
                if "type" in content and content["type"] == "audio":
                    contents.append(
                        {
                            "type": "audio",
                            "transcript": content["transcript"],
                        }
                    )

            choice: GenerationResultChoice = {
                "index": index,
                "finish_reason": response["status"]
                if response["status"] is not None
                else "stop",
                "logprobs": None,
                "message": {
                    "role": "assistant",
                    "content": contents,
                    "tool_calls": [],
                },
            }

            choices.append(choice)
        result: GenerationResult = {
            "id": response["id"],
            "object": response["object"],
            "created": int(time.time()),
            "model": model,
            "usage": {
                "completion_tokens": usage["output_tokens"],
                "prompt_tokens": usage["input_tokens"],
                "total_tokens": usage["total_tokens"],
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
        # Technically here new trace should start
