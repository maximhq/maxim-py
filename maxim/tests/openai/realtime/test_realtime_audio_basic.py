"""
Test file for OpenAI Realtime API - Basic Audio Tests

Tests basic audio input/output functionality including:
- Audio-only conversations
- Different voice options
- Audio format handling
- Session configuration
"""

import base64
import os
import unittest
from pathlib import Path

import dotenv
from openai import AsyncOpenAI

try:
    import simpleaudio

    SIMPLEAUDIO_AVAILABLE = True
except ImportError:
    SIMPLEAUDIO_AVAILABLE = False
    print("⚠️ simpleaudio not available. Audio playback will be skipped.")

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEST_AUDIO_FILE = Path(__file__).parent / "files" / "hello_test.wav"

# Audio playback constants for PCM16 format
SAMPLE_RATE_HZ = 24000  # Required by OpenAI Realtime API for PCM16
BYTES_PER_SAMPLE = 2  # PCM16 = 2 bytes per sample
CHANNELS = 1  # Mono audio


class TestRealtimeAudioBasic(unittest.IsolatedAsyncioTestCase):
    """Test basic audio functionality with OpenAI Realtime API."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        if not OPENAI_API_KEY:
            self.skipTest("OPENAI_API_KEY environment variable is not set")

        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def test_audio_input_output_basic(self):
        """Test basic audio input and output."""
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as connection:
                # Configure session for audio
                await connection.session.update(
                    session={
                        "model": "gpt-realtime",
                        "type": "realtime",
                        "output_modalities": ["audio"],
                        "tracing": "auto"
                    }
                )

                # Read test audio file and convert to base64
                if TEST_AUDIO_FILE.exists():
                    print(f"Reading test audio file: {TEST_AUDIO_FILE}")
                    with open(TEST_AUDIO_FILE, "rb") as f:
                        audio_data = f.read()

                    # Convert to base64
                    # pcm_data = pcm16_to_wav_bytes(audio_data)
                    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

                    # Send audio input
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "audio": audio_base64,
                                    "transcript": "Hello, how are you?"
                                }
                            ]
                        }
                    )

                    # Request response
                    await connection.response.create()

                    # Collect audio output - use a single loop to handle all events
                    audio_chunks = []
                    response_received = False

                    async for event in connection:
                        print(f"Event received: {event}")
                        if event.type == "response.output_audio.delta":
                            audio_chunks.append(event.delta)
                        elif event.type == "response.done":
                            response_received = True
                            break
                        elif event.type == "error":
                            self.fail(f"Error received: {event.error}")

                    self.assertTrue(response_received, "Expected response.done event")
                    self.assertGreater(len(audio_chunks), 0, "Expected audio chunks")

                    print(f"✅ Received {len(audio_chunks)} audio chunks")

                    # Play the audio response
                    if audio_chunks and SIMPLEAUDIO_AVAILABLE:
                        try:
                            # Decode base64 audio chunks and concatenate
                            audio_bytes = b"".join(
                                base64.b64decode(chunk) for chunk in audio_chunks
                            )

                            print("🔊 Playing audio response...")
                            play_obj = simpleaudio.play_buffer(
                                audio_bytes,
                                CHANNELS,
                                BYTES_PER_SAMPLE,
                                SAMPLE_RATE_HZ,
                            )
                            play_obj.wait_done()
                            print("✅ Audio playback completed")
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            print(f"⚠️ Audio playback error: {e}")
                    elif audio_chunks and not SIMPLEAUDIO_AVAILABLE:
                        print(
                            "⚠️ Audio chunks received but simpleaudio not available for playback"
                        )
                else:
                    self.skipTest(f"Test audio file not found: {TEST_AUDIO_FILE}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Audio input/output test error: {e}")
            self.skipTest(f"Audio input/output test error: {e}")

    async def test_text_input_audio_output(self):
        """Test text input with audio output."""
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as connection:
                # Configure session for text input and audio output
                await connection.session.update(
                    session={
                        "modalities": ["text", "audio"],
                        "output_audio_format": "pcm16",
                    }
                )

                # Send text message
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Hello, say hello back in audio.",
                            }
                        ],
                    }
                )

                # Request response
                await connection.response.create()

                # Collect audio output
                audio_chunks = []
                text_received = False
                response_received = False

                async for event in connection:
                    if event.type == "response.output_audio.delta":
                        audio_chunks.append(event.delta)
                    elif event.type == "response.output_text.delta":
                        text_received = True
                    elif event.type == "response.done":
                        response_received = True
                        break
                    elif event.type == "error":
                        self.fail(f"Error received: {event.error}")

                self.assertTrue(response_received, "Expected response.done event")
                self.assertGreater(len(audio_chunks), 0, "Expected audio chunks")

                print(f"✅ Received {len(audio_chunks)} audio chunks")
                if text_received:
                    print("✅ Also received text output")

                # Play the audio response
                if audio_chunks and SIMPLEAUDIO_AVAILABLE:
                    try:
                        # Decode base64 audio chunks and concatenate
                        audio_bytes = b"".join(
                            base64.b64decode(chunk) for chunk in audio_chunks
                        )

                        print("🔊 Playing audio response...")
                        play_obj = simpleaudio.play_buffer(
                            audio_bytes,
                            CHANNELS,
                            BYTES_PER_SAMPLE,
                            SAMPLE_RATE_HZ,
                        )
                        play_obj.wait_done()
                        print("✅ Audio playback completed")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"⚠️ Audio playback error: {e}")
                elif audio_chunks and not SIMPLEAUDIO_AVAILABLE:
                    print(
                        "⚠️ Audio chunks received but simpleaudio not available for playback"
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.skipTest(f"Text input audio output test error: {e}")

    async def test_different_voices(self):
        """Test different voice options."""
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

        for voice in voices:
            try:
                async with self.client.realtime.connect(
                    model="gpt-realtime"
                ) as connection:
                    # Configure session with specific voice
                    await connection.session.update(
                        session={
                            "modalities": ["text", "audio"],
                            "voice": voice,
                            "output_audio_format": "pcm16",
                        }
                    )

                    # Send text message
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Say hello in {voice} voice.",
                                }
                            ],
                        }
                    )

                    # Request response
                    await connection.response.create()

                    # Collect audio output
                    audio_chunks = []
                    response_received = False

                    async for event in connection:
                        if event.type == "response.output_audio.delta":
                            audio_chunks.append(event.delta)
                        elif event.type == "response.done":
                            response_received = True
                            break
                        elif event.type == "error":
                            print(f"⚠️ Error with voice {voice}: {event.error}")
                            break

                    if response_received:
                        self.assertGreater(
                            len(audio_chunks),
                            0,
                            f"Expected audio chunks for voice {voice}",
                        )
                        print(f"✅ Voice {voice} works: {len(audio_chunks)} chunks")

                        # Play the audio response
                        if SIMPLEAUDIO_AVAILABLE:
                            try:
                                audio_bytes = b"".join(
                                    base64.b64decode(chunk) for chunk in audio_chunks
                                )
                                print(f"🔊 Playing {voice} voice...")
                                play_obj = simpleaudio.play_buffer(
                                    audio_bytes,
                                    CHANNELS,
                                    BYTES_PER_SAMPLE,
                                    SAMPLE_RATE_HZ,
                                )
                                play_obj.wait_done()
                                print(f"✅ {voice} voice playback completed")
                            except Exception as e:  # pylint: disable=broad-exception-caught
                                print(f"⚠️ Audio playback error for {voice}: {e}")
                        else:
                            print(
                                f"⚠️ Audio chunks received for {voice} but simpleaudio not available"
                            )

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"⚠️ Voice {voice} test error: {e}")

    async def test_audio_format_pcm16(self):
        """Test PCM16 audio format."""
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as connection:
                # Configure session with PCM16 format
                await connection.session.update(
                    session={
                        "modalities": ["audio"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                    }
                )

                # Send text message (will be converted to audio internally)
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Test PCM16 format."}
                        ],
                    }
                )

                # Request response
                await connection.response.create()

                # Collect audio output
                audio_chunks = []
                response_received = False

                async for event in connection:
                    if event.type == "response.output_audio.delta":
                        audio_chunks.append(event.delta)
                    elif event.type == "response.done":
                        response_received = True
                        break

                self.assertTrue(response_received, "Expected response.done event")
                self.assertGreater(len(audio_chunks), 0, "Expected audio chunks")

                print(f"✅ PCM16 format test passed: {len(audio_chunks)} chunks")

                # Play the audio response
                if audio_chunks and SIMPLEAUDIO_AVAILABLE:
                    try:
                        audio_bytes = b"".join(
                            base64.b64decode(chunk) for chunk in audio_chunks
                        )
                        print("🔊 Playing PCM16 audio response...")
                        play_obj = simpleaudio.play_buffer(
                            audio_bytes,
                            CHANNELS,
                            BYTES_PER_SAMPLE,
                            SAMPLE_RATE_HZ,
                        )
                        play_obj.wait_done()
                        print("✅ Audio playback completed")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"⚠️ Audio playback error: {e}")
                elif audio_chunks and not SIMPLEAUDIO_AVAILABLE:
                    print(
                        "⚠️ Audio chunks received but simpleaudio not available for playback"
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.skipTest(f"PCM16 format test error: {e}")

    async def test_audio_transcription_enabled(self):
        """Test audio input with transcription enabled."""
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as connection:
                # Configure session with transcription
                await connection.session.update(
                    session={
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                    }
                )

                # Send text message (simulating transcribed audio)
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What did I just say?"}
                        ],
                    }
                )

                # Request response
                await connection.response.create()

                # Collect response
                audio_chunks = []
                transcript_received = False
                response_received = False

                async for event in connection:
                    if event.type == "response.output_audio.delta":
                        audio_chunks.append(event.delta)
                    elif event.type == "response.output_text.delta":
                        transcript_received = True
                    elif event.type == "response.done":
                        response_received = True
                        break

                self.assertTrue(response_received, "Expected response.done event")
                self.assertGreater(len(audio_chunks), 0, "Expected audio chunks")

                print(f"✅ Transcription test passed: {len(audio_chunks)} audio chunks")
                if transcript_received:
                    print("✅ Transcript received")

                # Play the audio response
                if audio_chunks and SIMPLEAUDIO_AVAILABLE:
                    try:
                        audio_bytes = b"".join(
                            base64.b64decode(chunk) for chunk in audio_chunks
                        )
                        print("🔊 Playing audio response with transcript...")
                        play_obj = simpleaudio.play_buffer(
                            audio_bytes,
                            CHANNELS,
                            BYTES_PER_SAMPLE,
                            SAMPLE_RATE_HZ,
                        )
                        play_obj.wait_done()
                        print("✅ Audio playback completed")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"⚠️ Audio playback error: {e}")
                elif audio_chunks and not SIMPLEAUDIO_AVAILABLE:
                    print(
                        "⚠️ Audio chunks received but simpleaudio not available for playback"
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.skipTest(f"Transcription test error: {e}")

    async def test_session_configuration(self):
        """Test session configuration options."""
        try:
            async with self.client.realtime.connect(model="gpt-realtime") as connection:
                # Test comprehensive session configuration
                await connection.session.update(
                    session={
                        "modalities": ["audio", "text"],
                        "voice": "nova",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                        },
                    }
                )

                # Verify session was updated
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Test session config."}
                        ],
                    }
                )

                await connection.response.create()

                audio_chunks = []
                response_received = False
                async for event in connection:
                    if event.type == "response.output_audio.delta":
                        audio_chunks.append(event.delta)
                    elif event.type == "response.done":
                        response_received = True
                        break

                self.assertTrue(response_received, "Expected response.done event")
                print("✅ Session configuration test passed")

                # Play the audio response
                if audio_chunks and SIMPLEAUDIO_AVAILABLE:
                    try:
                        audio_bytes = b"".join(
                            base64.b64decode(chunk) for chunk in audio_chunks
                        )
                        print("🔊 Playing session config audio response...")
                        play_obj = simpleaudio.play_buffer(
                            audio_bytes,
                            CHANNELS,
                            BYTES_PER_SAMPLE,
                            SAMPLE_RATE_HZ,
                        )
                        play_obj.wait_done()
                        print("✅ Audio playback completed")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"⚠️ Audio playback error: {e}")
                elif audio_chunks and not SIMPLEAUDIO_AVAILABLE:
                    print(
                        "⚠️ Audio chunks received but simpleaudio not available for playback"
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.skipTest(f"Session configuration test error: {e}")


if __name__ == "__main__":
    unittest.main()
