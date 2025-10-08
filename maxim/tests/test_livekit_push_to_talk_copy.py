import logging
import os
from enum import Enum

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobRequest, RoomIO, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import InputAudioTranscription
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit


class SessionState(Enum):
    """Session state enum for tracking the current state of the agent session"""
    ACTIVE = "active"
    CLEANING_UP = "cleaning_up"
    INTERRUPTED = "interrupted"
    SHUTDOWN = "shutdown"

logger = logging.getLogger("push-to-talk")
logger.setLevel(logging.INFO)

load_dotenv()
maxim_logger = Maxim({ "base_url": os.getenv("MAXIM_BASE_URL"), "debug": True }).logger()
instrument_livekit(maxim_logger)
print("Testting livekit push to talk, Take 20")

## This example demonstrates how to use the push-to-talk for multi-participant
## conversations with a voice agent
## It disables audio input by default, and only enables it when the client explicitly
## triggers the `start_turn` RPC method

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
        llm=openai.realtime.RealtimeModel(
            model="gpt-realtime",
            temperature=0.8,
            voice="sage",
            turn_detection=None,
            input_audio_transcription=InputAudioTranscription(
                model="gpt-4o-transcribe",
                language="en",
                prompt="You are transcribing the audio of a conversation between a kid named Tony and Bunny. Bunny is an AI guide for the kid. You should never output this prompt in your transcription. NEVER ever do that. Try to transcribe exactly what is in the audio. If you are not sure, then you should say specifically output 'transcription failed'. "
            ),
        ),
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # callback before generating a reply after user turn committed
        logger.info("User turn completed: '%s'", new_message.text_content)
        
        if not new_message.text_content:
            # for example, raise StopResponse to stop the agent from generating a reply
            logger.info("ignore empty user turn")
            raise StopResponse()
        
        # Log that we're about to let the agent respond
        logger.info("User input received, agent will respond automatically")
        
        # Debug: Check if we have access to session for manual generation if needed
        if hasattr(self, 'session') and self.session:
            logger.info("Agent session is available for manual response generation if needed")
        else:
            logger.info("No direct session access - relying on automatic response")


async def entrypoint(ctx: JobContext):
    session = AgentSession(turn_detection="manual")
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()

    agent = MyAgent()
    await session.start(agent=agent)
    ctx.connect()
    await session.generate_reply(
        instructions="Greet the user and offer your assistance in English."
    )

    # disable input audio at the start
    session.input.set_audio_enabled(False)

    logger.info("🚀 Agent started with RPC controls!")

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        logger.info("🎤 RPC: Starting turn with data: %s", data)
        session.interrupt()
        session.clear_user_turn()

        # listen to the caller if multi-user
        room_io.set_participant(data.caller_identity)
        session.input.set_audio_enabled(True)
        logger.info("Audio input enabled")

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        logger.info("🔇 RPC: Ending turn")
        session.input.set_audio_enabled(False)
        logger.info("Audio input disabled")
        session.commit_user_turn(transcript_timeout=0.01)
        logger.info("User turn committed - waiting for agent response")

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(_data: rtc.RpcInvocationData):
        logger.info("❌ RPC: Cancelling turn")
        session.input.set_audio_enabled(False)
        session.clear_user_turn()
        logger.info("Turn cancelled")


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # this attribute communicates to frontend that we support PTT
        attributes={"push-to-talk": "1"},
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, request_fnc=handle_request))
