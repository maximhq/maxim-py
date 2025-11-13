import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

elevenlabsApiKey = os.getenv("EL_API_KEY")
client = ElevenLabs(api_key=elevenlabsApiKey)

transcription_id = "transcription_id"

audio = client.speech_to_text.convert(model_id="eleven_multilingual_v2")
