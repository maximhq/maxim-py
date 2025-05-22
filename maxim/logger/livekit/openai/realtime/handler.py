from ...store import SessionStoreEntry


def parse_session_created(event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    print("#############CREATE LLM CALL############")


def handle_openai_client_event_queued(session_info: SessionStoreEntry, event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI client.
    """
    # We ignore buffer audio type as it just fills it with silence`
    type = event.get("type")
    if type == "input_audio_buffer.append":
        return
    print(f"OpenAI client event queued:type:{type} {event}")


def handle_openai_server_event_received(session_info: SessionStoreEntry, event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    type = event.get("type")
    if type == "session.created":
        parse_session_created(event)
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
        print("#############BUFFERING AUDIO############")
        pass
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
        print("#############LLM CALL COMPLETE############")
        pass
