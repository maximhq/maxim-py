def parse_session_created(event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    print(f"OpenAI session created event received: {event}")


def handle_openai_server_event_received(event: dict):
    """
    This function is called when the realtime session receives an event from the OpenAI server.
    """
    print(f"OpenAI server event received: {event}")
    type = event.get("type")
    if type == "session.created":
        parse_session_created(event)
