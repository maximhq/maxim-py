import collections.abc
import inspect
import io
import re
import struct
import types
import wave
from datetime import datetime
from decimal import Decimal
from typing import Any

from ..scribe import scribe


def is_silence(pcm_bytes, threshold=500, min_silence_ratio=0.95):
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
