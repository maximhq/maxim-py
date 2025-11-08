import os
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

from maxim import Maxim
from maxim.logger.elevenlabs import instrument_elevenlabs

load_dotenv()

# Configure logging to see the scribe logs
logging.basicConfig(level=logging.DEBUG)

elevenlabsApiKey = os.getenv("EL_API_KEY")
client = ElevenLabs(api_key=elevenlabsApiKey)

logger = Maxim({"debug": True}).logger()
instrument_elevenlabs(logger)

audio = client.text_to_speech.convert(
    text="The first move is what sets everything in motion.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)
