"""Instrument the ElevenLabs STT and TTS methods"""

import functools
import time
from typing import Dict
from uuid import uuid4

from maxim.logger import GenerationConfigDict
from maxim.logger.components.attachment import FileDataAttachment
from maxim.logger.components.generation import (
    AudioContent,
    GenerationResult,
    GenerationResultChoice,
)
from maxim.logger.components.trace import TraceConfigDict
from maxim.logger.elevenlabs.utils import ElevenLabsUtils
from maxim.logger.logger import Logger
from maxim.scribe import scribe

try:
    from elevenlabs.speech_to_text.client import SpeechToTextClient
    from elevenlabs.text_to_speech.client import TextToSpeechClient
except ImportError:
    SpeechToTextClient = None
    TextToSpeechClient = None

_instrumented = False
_global_logger: Logger | None = None
# Map trace_id to generation_id for pipeline operations
_trace_to_generation: Dict[str, str] = {}
# Map trace_id to audio durations for pipeline operations
_trace_to_durations: Dict[str, Dict[str, float]] = {}


def wrap_speech_to_text_convert(func, logger: Logger):
    """Wrap STT convert method to add tracing with audio attachment input and transcript output."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        global _global_logger
        _global_logger = logger

        # Extract audio data from kwargs (audio_file or audio parameter)
        audio_data = None
        audio_file = kwargs.get("file") or kwargs.get("audio")

        # Pre-read audio data safely
        if audio_file:
            # If it's a file path, read it (we open our own handle, so it's safe)
            if isinstance(audio_file, str):
                try:
                    with open(audio_file, "rb") as f:
                        audio_data = f.read()
                except (IOError, OSError) as e:
                    scribe().warning(f"[MaximSDK] Failed to read audio file: {e}")
            # If it's already bytes, use it directly
            elif isinstance(audio_file, bytes):
                audio_data = audio_file
            # For file-like objects, read it and seek back so original function can use it
            elif hasattr(audio_file, "read"):
                try:
                    # Save current position
                    saved_file_position = audio_file.tell()
                    # Read the data
                    audio_file.seek(0)
                    audio_data = audio_file.read()
                    # Restore position for original function
                    audio_file.seek(saved_file_position)
                except (AttributeError, IOError, OSError) as e:
                    # If we can't seek, we'll skip attachment
                    scribe().debug(
                        f"[MaximSDK] Could not read audio file handle for attachment: {e}"
                    )
                    audio_data = None

        # Check for trace_id in request_options.additional_headers
        trace_id = ElevenLabsUtils.get_maxim_trace_id(kwargs)

        # Determine if we're managing the trace lifecycle
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())

        if is_local_trace:
            # Create new trace for STT (standalone operation)
            trace = logger.trace(
                TraceConfigDict(
                    id=final_trace_id,
                    name="ElevenLabs Speech-to-Text",
                    tags={"provider": "elevenlabs", "operation": "stt"},
                )
            )
        else:
            # Use existing trace (get reference without overwriting name/tags)
            trace = logger.trace(TraceConfigDict(id=final_trace_id))

        try:
            # Call the original function (audio_file_handle position was already restored if needed)
            result = func(self, *args, **kwargs)

            # Extract transcript text from result
            transcript = ""
            if isinstance(result, str):
                transcript = result
            elif hasattr(result, "text"):
                transcript = result.text
            elif isinstance(result, dict) and "text" in result:
                transcript = result["text"]

            if is_local_trace:
                # For standalone STT, add audio attachment to trace and set output as transcript
                if audio_data:
                    trace.add_attachment(
                        FileDataAttachment(
                            data=audio_data,
                            tags={"attach-to": "input"},
                            name="User Speech Audio",
                            timestamp=int(time.time()),
                            mime_type=ElevenLabsUtils.get_audio_mime_type(kwargs),
                        )
                    )
                trace.set_output(transcript)
            else:
                # For pipeline STT, create generation with user transcript as input
                # and attach audio input to generation
                generation_id = _trace_to_generation.get(final_trace_id)
                if generation_id is None:
                    generation_id = str(uuid4())
                    _trace_to_generation[final_trace_id] = generation_id

                    # Create generation with user transcript as input
                    generation = trace.generation(
                        GenerationConfigDict(
                            id=generation_id,
                            provider="elevenlabs",
                            model=kwargs.get("model_id", "unknown"),
                            name="STT-TTS Pipeline Generation",
                            messages=[{"role": "user", "content": transcript}],
                        )
                    )

                    # Attach audio input to generation and calculate duration
                    if audio_data:
                        audio_data_mime_type = ElevenLabsUtils.get_audio_mime_type(
                            kwargs
                        )
                        input_duration = ElevenLabsUtils.calculate_audio_duration(
                            audio_data, audio_data_mime_type
                        )
                        if input_duration is not None:
                            if final_trace_id not in _trace_to_durations:
                                _trace_to_durations[final_trace_id] = {}
                            _trace_to_durations[final_trace_id]["input"] = (
                                input_duration
                            )

                        generation.add_attachment(
                            FileDataAttachment(
                                data=audio_data,
                                tags={"attach-to": "input"},
                                name="User Speech Audio",
                                timestamp=int(time.time()),
                                mime_type=audio_data_mime_type,
                            )
                        )

            scribe().debug(
                f"[MaximSDK] STT conversion completed: {len(transcript)} chars"
            )
            return result

        except Exception as e:
            scribe().error(f"[MaximSDK] STT conversion error: {e}")
            # Only end trace if we're managing its lifecycle
            if is_local_trace:
                trace.end()
            raise

    return wrapper


def wrap_text_to_speech_convert(func, logger: Logger):
    """Wrap TTS convert method to add tracing with text input and audio attachment output."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        global _global_logger
        _global_logger = logger

        # Extract text from kwargs
        text = kwargs.get("text", "")
        if not text and args:
            text = str(args[0]) if args else ""

        trace_id = ElevenLabsUtils.get_maxim_trace_id(kwargs)

        # Determine if we're managing the trace lifecycle
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())

        if is_local_trace:
            # Create new trace for TTS (standalone operation)
            trace = logger.trace(
                TraceConfigDict(
                    id=final_trace_id,
                    name="ElevenLabs Text-to-Speech",
                    tags={"provider": "elevenlabs", "operation": "tts"},
                )
            )
            # Set input as text for standalone TTS
            trace.set_input(text)
        else:
            # Use existing trace (get reference without overwriting name/tags)
            trace = logger.trace(TraceConfigDict(id=final_trace_id))

        try:
            # Call the original function
            result = func(self, *args, **kwargs)

            # Extract audio data from result
            audio_data = None
            return_value = result

            if isinstance(result, bytes):
                audio_data = result
            elif hasattr(result, "__iter__") and not isinstance(result, (bytes, str)):
                # Handle iterator of bytes chunks
                # Check if it's an iterator (but not bytes/str which are iterable)
                try:
                    # Collect all chunks from the iterator
                    collected_chunks = []
                    for chunk in result:
                        chunk_bytes = (
                            chunk if isinstance(chunk, bytes) else bytes(chunk)
                        )
                        collected_chunks.append(chunk_bytes)

                    # Combine all chunks for attachment
                    audio_data = (
                        b"".join(collected_chunks) if collected_chunks else None
                    )

                    # Create a new iterator from collected chunks for return
                    # This allows the caller to still iterate over the result
                    def chunk_iterator():
                        for chunk in collected_chunks:
                            yield chunk

                    return_value = chunk_iterator()
                except (TypeError, AttributeError):
                    # If iteration fails or it's not actually an iterator, try other methods
                    scribe().debug(
                        "[MaximSDK] Result has __iter__ but not iterable as expected, trying other methods"
                    )
                    if hasattr(result, "read"):
                        audio_data = result.read()
                    elif hasattr(result, "getvalue"):
                        audio_data = result.getvalue()
                    else:
                        audio_data = None
            elif hasattr(result, "read"):
                audio_data = result.read()
            elif hasattr(result, "getvalue"):
                audio_data = result.getvalue()

            if is_local_trace:
                # For standalone TTS, add audio attachment to trace and set output
                if audio_data:
                    trace.add_attachment(
                        FileDataAttachment(
                            data=audio_data,
                            tags={"attach-to": "output"},
                            name="Assistant Speech Audio",
                            mime_type=ElevenLabsUtils.get_audio_mime_type(kwargs),
                            timestamp=int(time.time()),
                        )
                    )
                trace.set_input(text)
                trace.end()
            else:
                # For pipeline TTS, attach audio output to generation and set generation output
                generation_id = _trace_to_generation.get(final_trace_id)
                if generation_id:
                    # Calculate output audio duration
                    output_mime_type = ElevenLabsUtils.get_audio_mime_type(kwargs)
                    output_duration = None
                    if audio_data:
                        output_duration = ElevenLabsUtils.calculate_audio_duration(
                            audio_data, output_mime_type, kwargs.get("output_format")
                        )
                        if output_duration is not None:
                            if final_trace_id not in _trace_to_durations:
                                _trace_to_durations[final_trace_id] = {}
                            _trace_to_durations[final_trace_id]["output"] = (
                                output_duration
                            )

                    # Attach audio output to generation
                    if audio_data:
                        logger.generation_add_attachment(
                            generation_id,
                            FileDataAttachment(
                                data=audio_data,
                                tags={"attach-to": "output"},
                                name="Assistant Speech Audio",
                                mime_type=output_mime_type,
                                timestamp=int(time.time()),
                            ),
                        )

                    # Get durations for usage
                    durations = _trace_to_durations.get(final_trace_id, {})
                    input_duration = durations.get("input", 0.0)
                    # Use the calculated output_duration if available, otherwise get from stored durations
                    if output_duration is None:
                        output_duration = durations.get("output", 0.0)

                    # Set generation output as assistant response transcript
                    # Create a generation result with the assistant response
                    generation_result = GenerationResult(
                        id=str(uuid4()),
                        object="tts.response",
                        created=int(time.time()),
                        model=kwargs.get("model_id", "unknown"),
                        choices=[
                            GenerationResultChoice(
                                index=0,
                                message={
                                    "role": "assistant",
                                    "content": [
                                        AudioContent(type="audio", transcript=text)
                                    ],
                                    "tool_calls": [],
                                },
                                finish_reason="stop",
                                logprobs=None,
                            )
                        ],
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "input_audio_duration": input_duration
                            if input_duration and input_duration > 0
                            else None,
                            "output_audio_duration": output_duration
                            if output_duration and output_duration > 0
                            else None,
                        },
                    )
                    logger.generation_result(generation_id, generation_result)
                else:
                    # Generation doesn't exist yet (shouldn't happen in normal flow, but handle gracefully)
                    scribe().warning(
                        f"[MaximSDK] Generation not found for trace {final_trace_id} in TTS convert"
                    )
                    # Fallback: attach to trace and set trace output
                    if audio_data:
                        trace.add_attachment(
                            FileDataAttachment(
                                data=audio_data,
                                tags={"attach-to": "output"},
                                name="Assistant Speech Audio",
                                mime_type=ElevenLabsUtils.get_audio_mime_type(kwargs),
                                timestamp=int(time.time()),
                            )
                        )
                    trace.set_output(text)

            scribe().debug(
                f"[MaximSDK] TTS conversion completed: {len(audio_data) if audio_data else 0} bytes"
            )
            return return_value

        except Exception as e:
            scribe().error(f"[MaximSDK] TTS conversion error: {e}")
            # Only end trace if we're managing its lifecycle
            if is_local_trace:
                trace.end()
            raise

    return wrapper


def wrap_text_to_speech_stream(func, logger: Logger):
    """Wrap TTS stream method to add tracing with text input and audio stream output."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        global _global_logger
        _global_logger = logger

        # Extract text from kwargs
        text = kwargs.get("text", "")
        if not text and args:
            text = str(args[0]) if args else ""

        # Check for trace_id in request_options.additional_headers
        trace_id = ElevenLabsUtils.get_maxim_trace_id(kwargs)

        # Determine if we're managing the trace lifecycle
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())

        if is_local_trace:
            # Create new trace for TTS streaming (standalone operation)
            trace = logger.trace(
                TraceConfigDict(
                    id=final_trace_id,
                    name="ElevenLabs Text-to-Speech Stream",
                    tags={"provider": "elevenlabs", "operation": "tts_stream"},
                )
            )
            # Set input as text for standalone TTS stream
            trace.set_input(text)
        else:
            # Use existing trace (get reference without overwriting name/tags)
            trace = logger.trace(TraceConfigDict(id=final_trace_id))

        try:
            # Call the original function
            result = func(self, *args, **kwargs)

            # For streaming, we collect chunks as they come
            # Note: This is a simplified implementation - full streaming would require
            # wrapping the generator/iterator
            audio_chunks = []
            if hasattr(result, "__iter__") and not isinstance(result, (bytes, str)):
                # Create a generator wrapper to collect chunks
                def generator_wrapper():
                    try:
                        for chunk in result:
                            audio_chunks.append(chunk)
                            yield chunk
                    finally:
                        # Combine chunks and add as attachment after streaming completes
                        if audio_chunks:
                            combined_audio = b"".join(
                                chunk if isinstance(chunk, bytes) else bytes(chunk)
                                for chunk in audio_chunks
                            )
                            if is_local_trace:
                                # For standalone TTS stream, attach to trace
                                trace.add_attachment(
                                    FileDataAttachment(
                                        data=combined_audio,
                                        tags={"attach-to": "output"},
                                        name="Assistant Speech Audio (Stream)",
                                        mime_type=ElevenLabsUtils.get_audio_mime_type(
                                            kwargs
                                        ),
                                        timestamp=int(time.time()),
                                    )
                                )
                                trace.end()
                            else:
                                # For pipeline TTS stream, attach to generation
                                generation_id = _trace_to_generation.get(final_trace_id)
                                if generation_id:
                                    # Calculate output audio duration
                                    output_mime_type = (
                                        ElevenLabsUtils.get_audio_mime_type(kwargs)
                                    )
                                    output_duration = (
                                        ElevenLabsUtils.calculate_audio_duration(
                                            combined_audio,
                                            output_mime_type,
                                            kwargs.get("output_format"),
                                        )
                                    )
                                    if output_duration is not None:
                                        if final_trace_id not in _trace_to_durations:
                                            _trace_to_durations[final_trace_id] = {}
                                        _trace_to_durations[final_trace_id][
                                            "output"
                                        ] = output_duration

                                    logger.generation_add_attachment(
                                        generation_id,
                                        FileDataAttachment(
                                            data=combined_audio,
                                            tags={"attach-to": "output"},
                                            name="Assistant Speech Audio (Stream)",
                                            mime_type=output_mime_type,
                                            timestamp=int(time.time()),
                                        ),
                                    )

                                    # Get durations for usage
                                    durations = _trace_to_durations.get(
                                        final_trace_id, {}
                                    )
                                    input_duration = durations.get("input", 0.0)
                                    # Use the calculated output_duration if available, otherwise get from stored durations
                                    if output_duration is None:
                                        output_duration = durations.get("output", 0.0)

                                    # Set generation output as assistant response transcript
                                    generation_result = GenerationResult(
                                        id=str(uuid4()),
                                        object="tts.response",
                                        created=int(time.time()),
                                        model=kwargs.get("model_id", "unknown"),
                                        choices=[
                                            GenerationResultChoice(
                                                index=0,
                                                message={
                                                    "role": "assistant",
                                                    "content": [
                                                        AudioContent(
                                                            type="audio",
                                                            transcript=text,
                                                        )
                                                    ],
                                                    "tool_calls": [],
                                                },
                                                finish_reason="stop",
                                                logprobs=None,
                                            )
                                        ],
                                        usage={
                                            "prompt_tokens": 0,
                                            "completion_tokens": 0,
                                            "total_tokens": 0,
                                            "input_audio_duration": input_duration
                                            if input_duration > 0
                                            else None,
                                            "output_audio_duration": output_duration
                                            if output_duration and output_duration > 0
                                            else None,
                                        },
                                    )
                                    logger.generation_result(
                                        generation_id, generation_result
                                    )
                                else:
                                    # Fallback: attach to trace
                                    trace.add_attachment(
                                        FileDataAttachment(
                                            data=combined_audio,
                                            tags={"attach-to": "output"},
                                            name="Assistant Speech Audio (Stream)",
                                            mime_type=ElevenLabsUtils.get_audio_mime_type(
                                                kwargs
                                            ),
                                            timestamp=int(time.time()),
                                        )
                                    )
                                    trace.set_output(text)
                        scribe().debug("[MaximSDK] TTS stream conversion completed")

                return generator_wrapper()
            else:
                # If it's not an iterator, treat it like convert
                audio_data = result
                if isinstance(audio_data, bytes):
                    if is_local_trace:
                        trace.add_attachment(
                            FileDataAttachment(
                                data=audio_data,
                                tags={"attach-to": "output"},
                                name="Assistant Speech Audio (Stream)",
                                mime_type=ElevenLabsUtils.get_audio_mime_type(kwargs),
                                timestamp=int(time.time()),
                            )
                        )
                        trace.end()
                    else:
                        # For pipeline TTS stream, attach to generation
                        generation_id = _trace_to_generation.get(final_trace_id)
                        if generation_id:
                            # Calculate output audio duration
                            output_mime_type = ElevenLabsUtils.get_audio_mime_type(
                                kwargs
                            )
                            output_duration = None
                            if isinstance(audio_data, bytes):
                                output_duration = (
                                    ElevenLabsUtils.calculate_audio_duration(
                                        audio_data,
                                        output_mime_type,
                                        kwargs.get("output_format"),
                                    )
                                )
                                if output_duration is not None:
                                    if final_trace_id not in _trace_to_durations:
                                        _trace_to_durations[final_trace_id] = {}
                                    _trace_to_durations[final_trace_id]["output"] = (
                                        output_duration
                                    )

                            logger.generation_add_attachment(
                                generation_id,
                                FileDataAttachment(
                                    data=audio_data,
                                    tags={"attach-to": "output"},
                                    name="Assistant Speech Audio (Stream)",
                                    mime_type=output_mime_type,
                                    timestamp=int(time.time()),
                                ),
                            )
                            # Get durations for usage
                            durations = _trace_to_durations.get(final_trace_id, {})
                            input_duration = durations.get("input", 0.0)
                            # Use the calculated output_duration if available, otherwise get from stored durations
                            if output_duration is None:
                                output_duration = durations.get("output", 0.0)

                            # Set generation output as assistant response transcript
                            generation_result = GenerationResult(
                                id=str(uuid4()),
                                object="tts.response",
                                created=int(time.time()),
                                model=kwargs.get("model_id", "unknown"),
                                choices=[
                                    GenerationResultChoice(
                                        index=0,
                                        message={
                                            "role": "assistant",
                                            "content": [
                                                AudioContent(
                                                    type="audio", transcript=text
                                                )
                                            ],
                                            "tool_calls": [],
                                        },
                                        finish_reason="stop",
                                        logprobs=None,
                                    )
                                ],
                                usage={
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                    "input_audio_duration": input_duration
                                    if input_duration > 0
                                    else None,
                                    "output_audio_duration": output_duration
                                    if output_duration > 0
                                    else None,
                                },
                            )
                            logger.generation_result(generation_id, generation_result)
                        else:
                            # Fallback: attach to trace
                            trace.add_attachment(
                                FileDataAttachment(
                                    data=audio_data,
                                    tags={"attach-to": "output"},
                                    name="Assistant Speech Audio (Stream)",
                                    mime_type=ElevenLabsUtils.get_audio_mime_type(
                                        kwargs
                                    ),
                                    timestamp=int(time.time()),
                                )
                            )
                            trace.set_output(text)
                scribe().debug("[MaximSDK] TTS stream conversion completed")
                return result

        except Exception as e:
            scribe().error(f"[MaximSDK] TTS stream conversion error: {e}")
            # Only end trace if we're managing its lifecycle
            if is_local_trace:
                trace.end()
            raise

    return wrapper


def instrument_elevenlabs_stt_tts(logger: Logger):
    """Instrument the ElevenLabs STT and TTS methods for tracing."""
    global _instrumented, _global_logger
    if _instrumented:
        scribe().info("[MaximSDK] ElevenLabs STT/TTS already instrumented")
        return

    _global_logger = logger

    # Instrument STT methods if available
    if SpeechToTextClient is not None:
        if hasattr(SpeechToTextClient, "convert"):
            setattr(
                SpeechToTextClient,
                "convert",
                wrap_speech_to_text_convert(SpeechToTextClient.convert, logger),
            )
            scribe().info("[MaximSDK] Instrumented ElevenLabs Speech-to-Text convert")

    # Instrument TTS methods if available
    if TextToSpeechClient is not None:
        if hasattr(TextToSpeechClient, "convert"):
            setattr(
                TextToSpeechClient,
                "convert",
                wrap_text_to_speech_convert(TextToSpeechClient.convert, logger),
            )
            scribe().info("[MaximSDK] Instrumented ElevenLabs Text-to-Speech convert")

        if hasattr(TextToSpeechClient, "stream"):
            setattr(
                TextToSpeechClient,
                "stream",
                wrap_text_to_speech_stream(TextToSpeechClient.stream, logger),
            )
            scribe().info("[MaximSDK] Instrumented ElevenLabs Text-to-Speech stream")

    _instrumented = True
