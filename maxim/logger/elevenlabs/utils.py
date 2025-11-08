from typing import Any, Optional
import io
import struct
import wave

from elevenlabs.core import RequestOptions
from maxim.scribe import scribe


class ElevenLabsUtils:
    @staticmethod
    def calculate_audio_duration(
        audio_data: bytes,
        mime_type: str = "audio/wav",
        output_format: Optional[str] = None,
    ) -> Optional[float]:
        """
        Calculate audio duration in seconds from audio bytes.

        Args:
                audio_data: Audio data bytes
                mime_type: MIME type of the audio (e.g., "audio/wav", "audio/mpeg")
                output_format: Optional ElevenLabs output format string (e.g., "mp3_44100_128")

        Returns:
                Duration in seconds, or None if calculation fails
        """
        if not audio_data:
            return None

        try:
            if mime_type.startswith("audio/wav") or mime_type == "audio/wav":
                # Parse WAV file header
                buffer = io.BytesIO(audio_data)
                with wave.open(buffer, "rb") as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    if sample_rate > 0:
                        duration = frames / float(sample_rate)
                        return duration
            elif (
                mime_type.startswith("audio/mpeg")
                or mime_type == "audio/mp3"
                or mime_type == "mp3"
            ):
                # Try to extract bitrate from output_format if provided (e.g., "mp3_44100_128" = 128 kbps)
                bitrate_kbps = None
                if output_format:
                    parts = output_format.split("_")
                    if len(parts) >= 3 and parts[0] == "mp3":
                        try:
                            bitrate_kbps = int(parts[2])
                        except (ValueError, IndexError):
                            pass

                # If we have bitrate from output_format, use it
                if bitrate_kbps and bitrate_kbps > 0:
                    duration = (len(audio_data) * 8) / (bitrate_kbps * 1000)
                    return duration

                # Otherwise, try to parse MP3 header
                if len(audio_data) >= 4:
                    # MP3 sync word is 0xFF 0xFB or 0xFF 0xFA
                    if audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0:
                        # Parse MP3 header
                        header = struct.unpack(">I", audio_data[:4])[0]

                        # Extract bitrate index (bits 12-15)
                        bitrate_index = (header >> 12) & 0x0F

                        # MPEG-1 Layer 3 bitrate table (kbps)
                        mpeg1_layer3_bitrates = [
                            0,
                            32,
                            40,
                            48,
                            56,
                            64,
                            80,
                            96,
                            112,
                            128,
                            160,
                            192,
                            224,
                            256,
                            320,
                            0,
                        ]

                        if bitrate_index < len(mpeg1_layer3_bitrates):
                            bitrate_kbps = mpeg1_layer3_bitrates[bitrate_index]
                            if bitrate_kbps > 0:
                                # Calculate duration: (file_size * 8) / (bitrate * 1000)
                                duration = (len(audio_data) * 8) / (bitrate_kbps * 1000)
                                return duration

                # Fallback: Estimate based on common bitrate (128 kbps for ElevenLabs)
                estimated_bitrate_kbps = 128
                duration = (len(audio_data) * 8) / (estimated_bitrate_kbps * 1000)
                return duration
        except Exception as e:
            scribe().debug(f"[MaximSDK] Failed to calculate audio duration: {e}")

        return None

    @staticmethod
    def get_maxim_trace_id(kwargs: dict[str, Any]) -> Optional[str]:
        request_options = kwargs.get("request_options")
        if not request_options:
            return None
        request_options = RequestOptions(request_options)
        trace_id = None
        if request_options and "additional_headers" in request_options:
            additional_headers = request_options["additional_headers"]
            if isinstance(additional_headers, dict):
                trace_id = additional_headers.get("x-maxim-trace-id")

        return trace_id

    @staticmethod
    def detect_audio_mime_type(audio_data: bytes) -> str:
        """
        Detect the MIME type of audio data from its header bytes.
        
        Args:
            audio_data: Audio data bytes
            
        Returns:
            Detected MIME type (defaults to "audio/wav" if detection fails)
        """
        if not audio_data or len(audio_data) < 4:
            return "audio/wav"  # Default fallback
        
        # Check for WAV file (starts with "RIFF")
        if audio_data[:4] == b"RIFF":
            return "audio/wav"
        
        # Check for MP3 file (starts with 0xFF and has sync bits)
        if audio_data[0] == 0xFF and (audio_data[1] & 0xE0) == 0xE0:
            return "audio/mpeg"
        
        # Default to WAV for unknown formats
        return "audio/wav"

    @staticmethod
    def get_audio_mime_type(kwargs: dict[str, Any]) -> str:
        output_format = kwargs.get("output_format")
        if isinstance(output_format, str):
            codec = output_format.split("_", 1)[0]
            if codec == "mp3":
                return "audio/mpeg"
            if codec in ("wav", "pcm"):
                return "audio/wav"
            if codec == "ogg":
                return "audio/ogg"
        return "audio/mpeg"
