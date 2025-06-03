from datetime import datetime, timezone
from uuid import uuid4

from .store import (
    SessionStoreEntry,
    Turn,
    get_livekit_callback,
    get_maxim_logger,
    get_session_store,
)


def start_new_turn(session_info: SessionStoreEntry):
    """
    This function will start a new turn and return the current turn.
    If the current turn is interrupted or empty, it will return None.
    If the current turn is not interrupted and not empty, it will return the current turn.
    If the current turn is interrupted and not empty, it will return None.
    If the current turn is empty, it will return None.
    If the current turn is not interrupted and empty, it will return None.
    If the current turn is interrupted and empty, it will return None.

    Args:
        session_info: The session information.

    Returns:
        The new turn or None if the current turn is interrupted or empty.
    """
    turn = session_info.get("current_turn", None)
    if turn is not None:
        if turn["is_interrupted"]:
            # User interrupted before assistant response - continue in current turn
            return None
        if (
            turn["turn_input_transcription"] == ""
            and turn["turn_input_audio_buffer"] == b""
        ):
            return None
    trace = get_session_store().get_current_trace_from_rt_session_id(
        session_info["rt_session_id"]
    )
    if trace is not None:
        trace.end()
        if get_livekit_callback() is not None:
            get_livekit_callback()(
                "maxim.trace.ended", {"trace_id": trace.id, "trace": trace}
            )
    next_turn_sequence = 1
    if turn is not None and "turn_sequence" in turn:
        next_turn_sequence = turn["turn_sequence"] + 1
    # Creating a new turn and new trace
    session_id = session_info["mx_session_id"]
    trace_id = str(uuid4())
    tags = {}
    if session_info["room_id"] is not None:
        tags["room_id"] = session_info["room_id"]
    if session_info["agent_id"] is not None:
        tags["agent_id"] = session_info["agent_id"]
    current_turn = Turn(
        turn_id=str(uuid4()),
        turn_sequence=next_turn_sequence,
        turn_timestamp=datetime.now(timezone.utc),
        turn_input_audio_buffer=bytes(),
        is_interrupted=False,
        turn_input_transcription="",
        turn_output_transcription="",
        turn_output_audio_buffer=bytes(),
    )
    trace = get_maxim_logger().trace(
        {
            "id": trace_id,
            "name": f"Turn {next_turn_sequence}",
            "session_id": session_id,
            "tags": tags,
        }
    )
    session_info["user_speaking"] = True
    session_info["current_turn"] = current_turn
    session_info["mx_current_trace_id"] = trace_id
    get_session_store().set_session(session_info)
    if get_livekit_callback() is not None:
        get_livekit_callback()(
            "maxim.trace.started", {"trace_id": trace_id, "trace": trace}
        )
    return current_turn
