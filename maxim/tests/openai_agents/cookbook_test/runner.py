import os
import asyncio
import importlib.util
from pathlib import Path
from uuid import uuid4
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
from agents import Runner, trace
from agents.tracing import add_trace_processor

from maxim import Maxim
from maxim.logger.openai.agents import MaximOpenAIAgentsTracingProcessor

load_dotenv()

logger = Maxim({"base_url": os.getenv("MAXIM_BASE_URL")}).logger()

# Handle import when run as script or as module
current_dir = Path(__file__).parent
cookbook_agents_path = current_dir / "cookbook_agents.py"

# Load the module dynamically
spec = importlib.util.spec_from_file_location("cookbook_agents", cookbook_agents_path)
cookbook_agents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cookbook_agents)


# Import the functions we need
create_knowledge_agent = cookbook_agents.create_knowledge_agent
create_triage_agent = cookbook_agents.create_triage_agent

# Global variable to hold the triage agent
triage_agent = None
vector_store_id = os.getenv("VECTOR_STORE_ID")


def pre_initialization():
    """Initialize vector store and create agents."""
    global triage_agent

    # vector_store_result = create_vector_store("ACME Shop Product Knowledge Base")
    # if not vector_store_result or "id" not in vector_store_result:
    #     raise ValueError("Failed to create vector store")

    # vector_store_id = vector_store_result["id"]
    # # Use absolute path based on script's directory
    # pdf_path = current_dir / "acme_product_catalogue.pdf"
    # upload_file(str(pdf_path), vector_store_id)

    # Create knowledge agent with actual vector store ID
    knowledge_agent = create_knowledge_agent(vector_store_id)

    # Create triage agent with all handoff agents
    triage_agent = create_triage_agent(knowledge_agent)

    add_trace_processor(MaximOpenAIAgentsTracingProcessor(logger))


async def voice_assistant():
    samplerate = sd.query_devices(kind="input")["default_samplerate"]

    while True:
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))

        # Check for input to either provide voice or exit
        cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
        if cmd.lower() == "esc":
            print("Exiting...")
            break
        print("Listening...")
        recorded_chunks = []

        # Start streaming from microphone until Enter is pressed
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            callback=lambda indata, frames, time, status: recorded_chunks.append(
                indata.copy()
            ),
        ):
            input()

        # Concatenate chunks into single buffer
        recording = np.concatenate(recorded_chunks, axis=0)

        # Input the buffer and await the result
        audio_input = AudioInput(buffer=recording)

        with trace("ACME App Voice Assistant", group_id=str(uuid4())):
            result = await pipeline.run(audio_input)

        # Transfer the streamed result into chunks of audio
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)

        response_audio = np.concatenate(response_chunks, axis=0)

        # Play response
        print("Assistant is responding...")
        sd.play(response_audio, samplerate=samplerate)
        sd.wait()
        print("---")


async def test_queries():
    examples = [
        "What's my ACME account balance doc? My user ID is 1234567890",  # Account Agent test
        "Ooh i've got money to spend! How big is the input and how fast is the output of the dynamite dispenser?",  # Knowledge Agent test
        "Hmmm, what about duck hunting gear - what's trending right now?",  # Search Agent test
    ]
    with trace("ACME App Assistant", group_id=str(uuid4())):
        for query in examples:
            result = await Runner.run(triage_agent, query)
            print(f"User: {query}")
            print(result.final_output)
            print("---")


if __name__ == "__main__":
    pre_initialization()
    asyncio.run(test_queries())
    # asyncio.run(voice_assistant())
