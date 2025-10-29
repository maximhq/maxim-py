import functools
from typing import Any
from uuid import uuid4

from elevenlabs.conversational_ai.conversation import Conversation

from maxim.logger import Session
from maxim.logger.components import TraceConfigDict
from maxim.logger.logger import Logger
from maxim.scribe import scribe
from maxim.types import Trace

_instrumented = False

# Global variables for trace and session
_current_trace: Trace | None = None
_current_session: Session | None = None

# TODO: Instrument execute_tool from ClientTools
# Also check the Events that are present. If they do get called, then perhaps can
# log them as well
# Also check the agent_response_correction callback and interrupt
# check the send_user_message function as well. Agents can be used by
# direct text inputs from users as well
# TODO: AsyncConversation
# Get agent and user audio
#
# Additional TODO: User and simulated user (Conversation simulator)
#


def wrap_conversation_start_session(func, logger: Logger):
    @functools.wraps(func)
    def wrapper(self: Conversation, *args: Any, **kwargs: Any):
        scribe().info(f"ElevenLabs conversation start session {vars(self)}")
        scribe().info(f"ElevenLabs conversation start session {args}")
        scribe().info(f"ElevenLabs conversation start session {kwargs}")

        global _current_session
        _current_session = logger.session(
            {"id": str(uuid4()), "name": "ElevenLabs Default session"}
        )

        return func(self, *args, **kwargs)

    return wrapper


def wrap_conversation_end_session(func):
    @functools.wraps(func)
    def wrapper(self: Conversation, *args: Any, **kwargs: Any):
        scribe().info(f"ElevenLabs conversation end session {vars(self)}")
        scribe().info(f"ElevenLabs conversation end session {args}")
        scribe().info(f"ElevenLabs conversation end session {kwargs}")

        global _current_session
        if _current_session and isinstance(_current_session, Session):
            _current_session.end()
            _current_session = None
        else:
            scribe().warning("[MAXIM SDK] Session is None at end_session")

        return func(self, *args, **kwargs)

    return wrapper


def wrap_init(original_init, logger: Logger):
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Store original callbacks
        original_transcript = self.callback_user_transcript
        original_response = self.callback_agent_response

        # Create transcript callback that includes observability
        def wrapped_transcript(transcript: str):
            # Always do our observability logic
            scribe().info(f"ElevenLabs callback user transcript {vars(self)}")
            scribe().info(f"ElevenLabs callback user transcript {transcript}")

            global _current_session
            global _current_trace
            try:
                if not _current_session:
                    raise ValueError("Session info is not found in transcript")

                if _current_session and not isinstance(_current_session, Session):
                    raise TypeError("Session is of invalid type in user_transcript")
                if _current_trace and not isinstance(_current_trace, Trace):
                    raise TypeError("Trace is of invalid type in user_transcript")

                trace_id = str(uuid4())
                # TODO: Could do a turn based indexing for name

                _current_trace = _current_session.trace(
                    TraceConfigDict(id=trace_id, name="Eleven Labs convo trace")
                )
                _current_trace.set_input(transcript)

                # TODO: Need to get the model and provider and token costs
                # Additionally, will also need to store the generation to push the result
                # when getting agent response
                # generation = trace.generation(GenerationConfigDict(id=str(uuid4())))

            except Exception as e:
                scribe().error(f"ElevenLabs callback user transcript error {e}")
            finally:
                # Call user's callback if it exists in all cases
                if original_transcript:
                    return original_transcript(transcript)

        # Create response callback that includes observability
        def wrapped_response(response: str):
            # Always do our observability logic
            scribe().info(f"ElevenLabs callback agent response {vars(self)}")
            scribe().info(f"ElevenLabs callback agent response {response}")

            global _current_trace
            try:
                if _current_trace and not isinstance(_current_trace, Trace):
                    raise TypeError("Trace is of invalid type")

                if _current_trace:
                    _current_trace.set_output(response)
                    _current_trace.end()
                    _current_trace = None
            except Exception as e:
                scribe().error(f"ElevenLabs callback agent response error {e}")
            finally:
                # Call user's callback if it exists
                if original_response:
                    return original_response(response)

        # Set up our wrapped callbacks
        self.callback_user_transcript = wrapped_transcript
        self.callback_agent_response = wrapped_response

    return wrapped_init


def instrument_elevenlabs(logger: Logger):
    # Set the scribe logger level based on debug flag
    # scribe().set_level(logging.DEBUG if debug else logging.INFO)
    global _instrumented
    if _instrumented:
        scribe().info("ElevenLabs already instrumented")
        return

    # Wrap methods at class level
    setattr(
        Conversation,
        "start_session",
        wrap_conversation_start_session(Conversation.start_session, logger),
    )
    setattr(
        Conversation,
        "end_session",
        wrap_conversation_end_session(Conversation.end_session),
    )

    # Wrap callbacks at instance level
    Conversation.__init__ = wrap_init(Conversation.__init__, logger)

    _instrumented = True
