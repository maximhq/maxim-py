import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from maxim import Maxim
from maxim.logger.elevenlabs import instrument_elevenlabs

load_dotenv()

elevenlabsApiKey = os.getenv("EL_API_KEY")
client = ElevenLabs(api_key=elevenlabsApiKey)

logger = Maxim({"debug": True}).logger()
instrument_elevenlabs(logger)

audio = client.speech_to_text.convert(model_id="eleven_multilingual_v2")
print(audio.text)