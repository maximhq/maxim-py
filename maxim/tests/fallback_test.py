import logging
import os
from typing import Optional

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import (
    deepgram,
    elevenlabs,
    google,
    noise_cancellation,
    openai,
    silero,
)

# from src.prompts.simple_agent_prompts import SYSTEM_PROMPT_INSTRUCTIONS
# from src.utils.logger import get_logger
from maxim import Maxim
from maxim.logger.livekit import instrument_livekit

load_dotenv()
# Initialize Maxim logger
logger_maxim = Maxim().logger()

# logger = get_logger("simple-voice-agent")


# Optional: Set up event handling for trace lifecycle
def on_event(event: str, data: dict):
    if event == "maxim.trace.started":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.info(f"Trace started - ID: {trace_id}")
    elif event == "maxim.trace.ended":
        trace_id = data["trace_id"]
        trace = data["trace"]
        logging.info(f"Trace ended - ID: {trace_id}")


# Instrument LiveKit with Maxim observability
instrument_livekit(logger_maxim, on_event)


class SimpleVoiceAgent(Agent):
    def __init__(self) -> None:
        self._tasks = []
        super().__init__(
            instructions="System instructions",
            stt=stt.FallbackAdapter(
                [
                    deepgram.STT(
                        model="nova-2-general",
                        language="en",
                        interim_results=False,  # Enable streaming partial results
                        smart_format=True,  # Better punctuation and formatting
                    )
                ]
            ),
            llm=llm.FallbackAdapter(
                [
                    openai.LLM(model="gpt-4.1", temperature=0.25),
                    google.LLM(model="gemini-2.5-pro-preview-05-06"),
                ]
            ),
            tts=tts.FallbackAdapter(
                [
                    elevenlabs.TTS(
                        model="eleven_multilingual_v2",
                        streaming_latency=3,
                        chunk_length_schedule=[120, 160, 250, 290],
                    ),
                    openai.TTS(model="tts-1"),
                ]
            ),
            # Optimized VAD settings for faster end-of-utterance detection
            vad=silero.VAD.load(
                activation_threshold=0.7,
                min_silence_duration=0.5,
            ),
        )


# async def start_recording_of_session_to_s3_via_egress(
#     ctx: JobContext,
#     lkapi: api.LiveKitAPI,
# ) -> Optional[api.EgressInfo]:
#     try:
#         # Start S3 recording
#         req = api.RoomCompositeEgressRequest(
#             room_name=ctx.room.name,
#             layout="speaker",
#             preset=api.EncodingOptionsPreset.H264_720P_30,
#             audio_only=False,
#             file_outputs=[
#                 api.EncodedFileOutput(
#                     file_type=api.EncodedFileType.MP4,
#                     filepath="livekit-poc-data/recordings/recording-{room_name}-{time}.mp4",
#                     s3=api.S3Upload(
#                         access_key=os.getenv("AWS_ACCESS_KEY_ID"),
#                         secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
#                         session_token=os.getenv("AWS_SESSION_TOKEN"),
#                         region=os.getenv("AWS_REGION"),
#                         bucket=os.getenv("AWS_BUCKET_NAME"),
#                     ),
#                 )
#             ],
#         )
#         egress_info = await lkapi.egress.start_room_composite_egress(req)
#         # logger.info(f"Started recording: {egress_info.egress_id}")
#         return egress_info
#     except Exception as e:
#         # logger.error(f"Failed to start recording of the session to be saved to S3: {e}")
#         pass


async def entrypoint(ctx: JobContext):
    session = AgentSession()

    lkapi = api.LiveKitAPI()
    # await start_recording_of_session_to_s3_via_egress(ctx, lkapi)

    # Create session folders and files for metrics and usages
    # session_start, session_folder_path, metrics_file, usage_file = (
    #     create_session_folders_and_files_for_metrics_and_usages(ctx)
    # )

    # usage_collector = metrics.UsageCollector()

    # @session.on("metrics_collected")
    # def on_metrics_collected(ev: MetricsCollectedEvent):
    #     log_metrics_to_file(ev, metrics_file, ctx, usage_collector)

    # async def log_usage():
    #     await log_usage_to_file(
    #         ctx,
    #         session_start,
    #         usage_collector,
    #         session_folder_path,
    #         metrics_file,
    #         usage_file,
    #     )

    # At shutdown, generate and log the summary from the usage collector
    # ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=SimpleVoiceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await lkapi.aclose()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
