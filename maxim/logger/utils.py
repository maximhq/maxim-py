import collections.abc
import inspect
import io
import math
import re
import struct
import types
import wave
from datetime import datetime
from decimal import Decimal
from typing import Any
from base64 import b64decode

from ..scribe import scribe

# Audio processing constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 10
DEFAULT_SILENCE_THRESHOLD = 300
DEFAULT_LOOKBACK_MS = 500
DEFAULT_MIN_SILENCE_RATIO = 0.95


def is_silence(pcm_bytes, threshold=DEFAULT_SILENCE_THRESHOLD, min_silence_ratio=DEFAULT_MIN_SILENCE_RATIO):
    """
    Detects if the given PCM16 byte buffer is mostly silence.

    Args:
        pcm_bytes (bytes): PCM16LE audio data.
        threshold (int): Max absolute value to consider as silence.
        min_silence_ratio (float): Minimum ratio of silent samples to consider the buffer as silence.

    Returns:
        bool: True if buffer is mostly silence, False otherwise.
    """
    num_samples = len(pcm_bytes) // 2
    if num_samples == 0:
        return True  # Empty buffer is considered silence

    silent_count = 0

    for i in range(num_samples):
        # '<h' is little-endian 16-bit signed integer
        sample = struct.unpack_from("<h", pcm_bytes, i * 2)[0]
        if abs(sample) < threshold:
            silent_count += 1

    silence_ratio = silent_count / num_samples
    return silence_ratio >= min_silence_ratio


def pcm16_to_wav_bytes(
    pcm_bytes: bytes, num_channels: int = 1, sample_rate: int = 24000
) -> bytes:
    """
    Convert PCM-16 audio data to WAV format bytes.

    Args:
        pcm_bytes (bytes): Raw PCM-16 audio data
        num_channels (int): Number of audio channels (default: 1)
        sample_rate (int): Sample rate in Hz (default: 24000)

    Returns:
        bytes: WAV format audio data
    """
    try:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()
    except Exception as e:
        scribe().error(
            f"[MaximSDK] Error converting PCM-16 audio data to WAV format: {e}"
        )
        return pcm_bytes

def string_pcm_to_wav_bytes(
    pcm_str: str | None, num_channels: int = 1, sample_rate: int = 24000
) -> bytes:
    """
    Convert a string of PCM-16 audio data to WAV format bytes.
    """
    if pcm_str is None:
        return b""
    try:
        pcm_bytes = b64decode(pcm_str, validate=True)
    except Exception as e:
        scribe().error(
            f"[MaximSDK] Error decoding string PCM-16 audio data: {e}"
        )
        return b""
    try:
        return pcm16_to_wav_bytes(pcm_bytes, num_channels, sample_rate)
    except Exception as e:
        scribe().error(
            f"[MaximSDK] Error converting string PCM-16 audio data to WAV format: {e}"
        )
        return b""

def trim_silence_edges(
    pcm_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    threshold: int = DEFAULT_SILENCE_THRESHOLD,
    lookback_ms: int = DEFAULT_LOOKBACK_MS,
    last_non_silent_removal_frames: int = 1,
) -> bytes:
    """
    Remove leading and trailing silence from PCM16LE mono audio.

    - Splits audio into fixed-size frames (frame_ms) and marks a frame silent
      if its max absolute sample amplitude is below `threshold`.
    - Trims all silent frames at the start and end, leaving the middle intact.
    - If the first non-silent frame is found at or after `lookback_ms`, the
      returned audio starts `lookback_ms` earlier (clamped to 0) to avoid abrupt
      cut-ins when the speech starts suddenly.

    Args:
        pcm_bytes: Raw PCM16LE mono audio bytes.
        sample_rate: Samples per second.
        frame_ms: Frame size in milliseconds used for silence detection.
        threshold: Amplitude threshold below which a frame is considered silent.
        lookback_ms: Lookback duration in milliseconds that extends silence trimming
                    by up to that duration to avoid abrupt audio cut-ins.

    Returns:
        Bytes for the trimmed PCM16LE audio. Returns empty bytes if entirely silent.
    """
    try:
        if not pcm_bytes:
            return pcm_bytes

        bytes_per_sample = 2
        total_samples = len(pcm_bytes) // bytes_per_sample
        if total_samples == 0:
            return b""

        frame_size_samples = max(1, int(sample_rate * frame_ms / 1000))
        frame_size_bytes = frame_size_samples * bytes_per_sample
        if frame_size_bytes == 0:
            frame_size_bytes = bytes_per_sample

        num_frames = (len(pcm_bytes) + frame_size_bytes - 1) // frame_size_bytes

        # Determine which frames are silent
        silent_flags: list[bool] = []
        view = memoryview(pcm_bytes)
        for i in range(num_frames):
            start = i * frame_size_bytes
            end = min(start + frame_size_bytes, len(pcm_bytes))
            frame = view[start:end]
            max_abs = 0
            # Iterate samples in frame
            # Use range step 2 to unpack int16 little-endian
            for off in range(0, len(frame) - (len(frame) % 2), 2):
                sample = struct.unpack_from("<h", frame, off)[0]
                abs_val = sample if sample >= 0 else -sample
                if abs_val > max_abs:
                    max_abs = abs_val
                    if max_abs >= threshold:
                        break
            silent_flags.append(max_abs < threshold)

        # Find first and last non-silent frames
        first_non_silent = 0
        while first_non_silent < num_frames and silent_flags[first_non_silent]:
            first_non_silent += 1

        if first_non_silent >= num_frames:
            # Entire audio is silent
            return b""

        last_non_silent = num_frames - 1
        while last_non_silent >= 0 and silent_flags[last_non_silent]:
            last_non_silent -= 1

        # Doing this so that we don't cut in the audio when the speech starts suddenly
        # Apply lookback if the detected start is sufficiently far into the audio
        if lookback_ms > 0:
            # Guard against division by zero and use ceiling to ensure we always subtract
            # at least the intended number of frames
            frame_ms = max(1, frame_ms)
            lookback_frames = math.ceil(lookback_ms / float(frame_ms))
            if first_non_silent * frame_ms >= lookback_ms:
                first_non_silent = max(0, first_non_silent - lookback_frames)

        start_byte = first_non_silent * frame_size_bytes
        end_byte = min((last_non_silent + last_non_silent_removal_frames) * frame_size_bytes, len(pcm_bytes))

        if start_byte <= 0 and end_byte >= len(pcm_bytes):
            # Nothing to trim
            return pcm_bytes
        return pcm_bytes[start_byte:end_byte]
    except Exception as e:
        scribe().warning(
            f"[MaximSDK] trim_silence_edges failed; error={e}"
        )
        # On failure, return original audio to avoid data loss
        return pcm_bytes


def make_object_serializable(obj: Any) -> Any:
    """
    Convert any Python object into a JSON-serializable format while preserving
    as much information as possible about the original object.

    Args:
        obj: Any Python object

    Returns:
        A JSON-serializable representation of the object
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic types that are already serializable
    if isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle Decimal
    if isinstance(obj, Decimal):
        return str(obj)

    # Handle complex numbers
    if isinstance(obj, complex):
        return {"type": "complex", "real": obj.real, "imag": obj.imag}

    # Handle bytes and bytearray
    if isinstance(obj, (bytes, bytearray)):
        return {"type": "bytes", "data": obj.hex(), "encoding": "hex"}

    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Handle regular expressions
    if isinstance(obj, re.Pattern):
        return {"type": "regex", "pattern": obj.pattern, "flags": obj.flags}

    # Handle functions
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        return {
            "type": "function",
            "name": obj.__name__,
            "module": obj.__module__,
            "source": inspect.getsource(obj) if inspect.isroutine(obj) else None,
            "signature": str(inspect.signature(obj))
            if inspect.isroutine(obj)
            else None,
        }

    # Handle exceptions
    if isinstance(obj, Exception):
        return {
            "type": "error",
            "error_type": obj.__class__.__name__,
            "message": str(obj),
            "args": make_object_serializable(obj.args),
            "traceback": str(obj.__traceback__) if obj.__traceback__ else None,
        }

    # Handle sets
    if isinstance(obj, (set, frozenset)):
        return {
            "type": "set",
            "is_frozen": isinstance(obj, frozenset),
            "values": [make_object_serializable(item) for item in obj],
        }

    # Handle dictionaries and mapping types
    if isinstance(obj, collections.abc.Mapping):
        return {str(key): make_object_serializable(value) for key, value in obj.items()}

    # Handle lists, tuples, and other iterables
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, collections.abc.Iterable)
        and not isinstance(obj, (str, bytes, bytearray))
    ):
        return [make_object_serializable(item) for item in obj]

    # Handle custom objects
    try:
        # Try to convert object's dict representation
        obj_dict = obj.__dict__
        return {
            "type": "custom_object",
            "class": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "attributes": make_object_serializable(obj_dict),
        }
    except AttributeError:
        # If object doesn't have __dict__, try to get string representation
        return {
            "type": "unknown",
            "class": obj.__class__.__name__,
            "string_repr": str(obj),
        }
