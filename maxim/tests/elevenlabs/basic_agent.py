import os
from dotenv import load_dotenv
import logging

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from maxim import Maxim
from maxim.logger.elevenlabs import instrument_elevenlabs

logging.basicConfig(level=logging.INFO)
load_dotenv()

baseUrl = os.getenv("MAXIM_BASE_URL")
agent_id = os.getenv("ELEVENLABS_AGENT_ID")
eleven_labs_api_key = os.getenv("ELEVENLABS_API_KEY")

if not eleven_labs_api_key:
    raise ValueError("ELEVENLABS_API_KEY is not set")
if not agent_id:
    raise ValueError("ELEVENLABS_AGENT_ID is not set")

logger = Maxim({"base_url": baseUrl}).logger()
instrument_elevenlabs(logger)

client = ElevenLabs(api_key=eleven_labs_api_key)

# Create audio interface for real-time audio input/output
audio_interface = DefaultAudioInterface()

# Create conversation
conversation = Conversation(
    client=client,
    agent_id=agent_id,
    requires_auth=True,
    audio_interface=audio_interface,
)

# Start the conversation
conversation.start_session()

print("\nPress Enter to end the conversation...")
input()  # Wait for Enter key

# End the conversation
conversation.end_session()
logger.cleanup()
print("Conversation ended.")
