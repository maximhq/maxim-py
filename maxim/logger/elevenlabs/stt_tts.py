"""Instrument the ElevenLabs STT and TTS methods"""

import functools
import itertools

from maxim import Logger
from maxim.scribe import scribe
from elevenlabs.text_to_speech.client import TextToSpeechClient

_instrumented = False


# This would be returning an audio
def wrap_text_to_speech_convert(func, logger: Logger):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        scribe().info(f"Converted text to speech: {result}")
        return result

    return wrapper


# This would be returning an audio stream
def wrap_text_to_speech_stream(func, logger: Logger):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        scribe().info(f"Converted text to speech: {result}")
        return result

    return wrapper


def instrument_elevenlabs_stt_tts(logger: Logger):
    # Instrument the STT and TTS methods
    global _instrumented
    if _instrumented:
        return

    # Instrument the STT methods
    setattr(
        TextToSpeechClient,
        "convert",
        wrap_text_to_speech_convert(TextToSpeechClient.convert, logger),
    )

    # Instrument the TTS methods
    setattr(
        TextToSpeechClient,
        "stream",
        wrap_text_to_speech_stream(TextToSpeechClient.stream, logger),
    )
    _instrumented = True
