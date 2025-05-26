from typing import Any
from uuid import uuid4

from ...store import SessionStoreEntry, get_maxim_logger, get_session_store
from .events import SessionCreatedEvent, get_model_params


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
    trace.generation(
        {
            "id": str(uuid4()),
            "model": event["session"]["model"],
            "provider": "openai",
            "model_parameters": get_model_params(event["session"]),
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
    print(f"buffer audio {event}")
    # Buffering audio to the current session_entry
    if entry["current_turn"] is None:
        return
    turn = entry["current_turn"]
    if turn["turn_audio_buffer"] is None:
        turn["turn_audio_buffer"] = b""
    turn["turn_audio_buffer"] += event["delta"]
    entry["current_turn"] = turn


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
        pass
