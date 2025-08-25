import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    google,
    noise_cancellation,
    silero,
)
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit

load_dotenv()

logger = Maxim({ "base_url": os.getenv("MAXIM_BASE_URL") }).logger()
instrument_livekit(logger)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.",
            stt=openai.STT(model="gpt-4o-mini-transcribe"),
            llm=google.LLM(model="gemini-2.0-flash-001", temperature=1),
            tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),
            vad=silero.VAD.load(),
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
