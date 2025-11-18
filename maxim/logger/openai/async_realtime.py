import base64
import json
import time
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from openai import AsyncOpenAI
from openai._types import omit
from openai.resources.realtime import AsyncRealtime
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
    AsyncRealtimeInputAudioBufferResource,
)
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    RealtimeServerEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)

from maxim.logger import Generation, Logger, Session, Trace
from maxim.logger.components import FileDataAttachment, GenerationRequestMessage
from maxim.logger.components.generation import GenerationConfigDict
from maxim.logger.openai.realtime_handlers.async_handlers import (
    handle_conversation_item_message,
    handle_function_call_response,
    handle_text_response,
)
from maxim.logger.utils import pcm16_to_wav_bytes, trim_silence_edges

from ...scribe import scribe


# pylint: disable=broad-exception-caught
class MaximOpenAIAsyncRealtimeConnection(AsyncRealtimeConnection):
    def __init__(
        self, connection, logger: Logger, extra_headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(connection)
        self._logger = logger
        self._extra_headers = extra_headers or {}

        # Extract session metadata from headers
        session_id = self._extra_headers.get("x-maxim-session-id", None)
        generation_name = self._extra_headers.get("x-maxim-generation-name", None)

        # Create or use existing session
        self._session_id = session_id or str(uuid4())
        self._generation_name = generation_name
        self._is_local_session = session_id is None

        # State tracking
        self._session: Optional[Session] = None
        self._transcription_model: Optional[str] = None
        self._transcription_language: Optional[str] = None
        self._output_audio: Optional[bytes] = None
        self._current_trace: Optional[Trace] = None
        self._current_generation: Optional[Generation] = None
        self._last_user_message: Optional[str] = None
        self._session_model: Optional[str] = None
        self._session_config: Optional[Dict[str, Any]] = None
        self._system_instructions: str | None = None
        self._function_calls: Dict[
            str, Dict[str, Any]
        ] = {}  # call_id -> function call data
        self._function_call_arguments: Dict[
            str, str
        ] = {}  # call_id -> accumulated arguments
        self._has_pending_tool_calls: bool = (
            False  # Track if current response has tool calls
        )
        self._is_continuing_trace: bool = (
            False  # Track if we're continuing a trace after tool calls
        )
        self._tool_calls: Dict[str, Any] = {}  # call_id -> ToolCall object
        self._tool_call_outputs: Dict[str, str] = {}  # call_id -> tool output
        self._user_audio_buffer: Dict[
            str, bytes
        ] = {}  # item_id -> accumulated user audio bytes
        self._pending_user_audio: bytes = bytes()  # Buffer audio before item is created
        self._current_item_id: Optional[str] = None  # Track current item_id for audio association

        # Wrap input_audio_buffer to capture user audio

        class MaximInputAudioBuffer(AsyncRealtimeInputAudioBufferResource):
            def __init__(self, parent_connection, maxim_connection):
                super().__init__(parent_connection)
                self._maxim_connection = maxim_connection

            async def append(self, *, audio: str, **kwargs) -> None:  # type: ignore
                """Override append to capture audio before sending."""
                # Decode base64 audio and add to pending buffer
                try:
                    audio_bytes = base64.b64decode(audio)
                    self._maxim_connection._pending_user_audio += audio_bytes
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error capturing user audio: {str(e)}"
                    )
                # Call parent to actually send the audio
                await super().append(audio=audio, **kwargs)

            async def commit(self, **kwargs) -> None:  # type: ignore
                """Override commit to handle audio buffer commit."""
                # Audio will be associated with the next conversation.item.created event
                await super().commit(**kwargs)

        # Replace the input_audio_buffer with our wrapper
        self.input_audio_buffer = MaximInputAudioBuffer(self, self)

    async def __aiter__(self) -> AsyncIterator[RealtimeServerEvent]:
        """
        Override to intercept events and log them to Maxim.
        """
        from websockets.exceptions import ConnectionClosedOK

        try:
            async for event in super().__aiter__():
                await self._handle_event(event)
                yield event
        except ConnectionClosedOK:
            # End current trace if exists
            if self._current_trace is not None:
                try:
                    self._current_trace.end()
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error ending trace: {str(e)}"
                    )
            # End session if it's a local session
            if self._is_local_session and self._session is not None:
                try:
                    self._logger.session_end(self._session_id)
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error ending session: {str(e)}"
                    )
            return

    async def _handle_event(self, event: RealtimeServerEvent) -> None:
        """Handle realtime events and log to Maxim."""

        try:
            event_type = event.type

            if event_type == "session.created":
                await self._handle_session_created(event)
            elif event_type == "session.updated":
                await self._handle_session_updated(event)
            elif event_type == "conversation.item.added":
                await self._handle_conversation_item_created(event)
            elif event_type == "response.created":
                await self._handle_response_created(event)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await (
                    self._handle_conversation_item_input_audio_transcription_completed(
                        event
                    )
                )
            elif event_type == "response.output_audio.delta":
                await self._handle_response_output_audio_delta(event)
            elif event_type == "response.function_call_arguments.delta":
                await self._handle_function_call_arguments_delta(event)
            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call_arguments_done(event)
            elif event_type == "response.done":
                await self._handle_response_done(event)
            elif event_type == "conversation.item.deleted":
                await self._handle_conversation_item_deleted(event)
            elif event_type == "realtime.error":
                await self._handle_error(event)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling event: {str(e)}"
            )

    async def _handle_session_created(self, event: SessionCreatedEvent) -> None:
        """Handle session.created event to extract model configuration and create Maxim session."""
        try:
            session = event.session
            self._system_instructions = session.instructions

            # Extract model from session
            self._session_model = session.model
            if self._session_model is None:
                # Try to get from dict-like structure
                if hasattr(session, "model_dump"):
                    session_dict = session.model_dump()
                    self._session_model = session_dict.get("model")
                elif isinstance(session, dict):
                    self._session_model = session.get("model")

            # Store session config for model parameters
            if hasattr(session, "model_dump"):
                self._session_config = session.model_dump()
            elif isinstance(session, dict):
                self._session_config = session.copy()

            if self._session is None:
                try:
                    self._session = self._logger.session(
                        {"id": self._session_id, "name": "OpenAI Realtime Session"}
                    )
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error creating session: {str(e)}"
                    )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling session.created: {str(e)}"
            )

    async def _handle_session_updated(self, event: SessionUpdatedEvent) -> None:
        """Handle session.updated event to extract model configuration and create Maxim session."""
        try:
            session = event.session
            if (
                session.audio is not None
                and session.audio.input is not None
                and session.audio.input.transcription is not None
            ):
                self._transcription_model = session.audio.input.transcription.model
                self._transcription_language = (
                    session.audio.input.transcription.language
                )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling session.updated: {str(e)}"
            )

    async def _handle_conversation_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        """Handle conversation.item.input_audio_transcription.completed event to handle the transcription of the audio."""
        try:
            if self._current_generation is not None and self._current_trace is not None:
                # Ok, so let me explain why the below code is necessary.
                # So, basically, we get the `response.created` event before the `conversation.item.input_audio_transcription.completed` event.
                # In the `response.created` event, we create a new generation for the user's message
                # So, if we just directly create a new generation for the transcription, what would happen is that the STT generation would be logged after the actual assistant generation.
                # So, instead, what I do is that I override the current generation for the transcription and create a new generation for the assistant.
                # Not setting it in any other events as `input_audio_buffer.committed` happens before the `response.created` event so there is no trace present there (for the first turn) or would go into the previous turn's trace.

                model_parameters = self._current_generation.model_parameters
                self._current_generation.set_model(self._transcription_model)
                self._current_generation.add_message(
                    GenerationRequestMessage(
                        role="user",
                        content=event.transcript,
                    )
                )
                self._current_generation.set_model_parameters({
                    "language": self._transcription_language,
                })
                self._current_generation.set_name("User Speech Transcription")
                self._current_generation.result(
                    {
                        "id": event.event_id,
                        "object": "stt.response",
                        "created": int(time.time()),
                        "model": self._transcription_model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": event.usage.input_tokens or 0,
                            "completion_tokens": event.usage.output_tokens or 0,
                            "total_tokens": event.usage.total_tokens or 0,
                        },
                    }
                )
                # Attach user audio if available (for STT generation)
                # Note: Audio may already be attached from _handle_response_created, but we ensure it's here too
                item_id = getattr(event, "item_id", None)
                user_audio = None
                if item_id and item_id in self._user_audio_buffer:
                    user_audio = self._user_audio_buffer[item_id]
                    if user_audio:
                        user_audio = trim_silence_edges(user_audio,sample_rate=24000, last_non_silent_removal_frames=2)
                        try:
                            self._current_generation.add_attachment(
                                FileDataAttachment(
                                    data=pcm16_to_wav_bytes(user_audio),
                                    tags={"attach-to": "input"},
                                    name="User Audio Input",
                                    timestamp=int(time.time()),
                                )
                            )
                        except Exception as e:
                            scribe().warning(
                                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error adding user audio attachment to STT generation: {str(e)}"
                            )

                self._current_generation.end()
                self._current_generation = None

                # Create new generation for the assistant response with the correct transcript
                generation_id = str(uuid4())
                self._current_generation = self._current_trace.generation(
                    GenerationConfigDict(
                        id=generation_id,
                        provider="openai",
                        model=self._session_model or "unknown",
                        name=self._generation_name,
                        model_parameters=model_parameters,
                        messages=[
                            GenerationRequestMessage(
                                role="user",
                                content=event.transcript,
                            )
                        ],
                    )
                )
                
                # Attach user audio to the assistant generation as well
                if user_audio:
                    try:
                        self._current_generation.add_attachment(
                            FileDataAttachment(
                                data=pcm16_to_wav_bytes(user_audio),
                                tags={"attach-to": "input"},
                                name="User Audio Input",
                                timestamp=int(time.time()),
                            )
                        )
                    except Exception as e:
                        scribe().warning(
                            f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error adding user audio attachment to assistant generation: {str(e)}"
                        )
                    # Clean up after attaching to both generations
                    if item_id and item_id in self._user_audio_buffer:
                        del self._user_audio_buffer[item_id]
                    # Clear current item_id reference
                    if self._current_item_id == item_id:
                        self._current_item_id = None
            else:
                # No current generation exists - this shouldn't happen in normal flow,
                # but set _last_user_message for safety
                self._last_user_message = event.transcript
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling conversation.item.input_audio_transcription.completed: {str(e)}"
            )

    async def _handle_conversation_item_created(
        self, event: ConversationItemCreatedEvent
    ) -> None:
        """
        Handle conversation.item.added event to track user messages and function call outputs.
        This handles the extraction of the user message from the event because we do not receive the user message in the response.created event.
        """
        try:
            item = event.item

            if item.type == "message":
                role = item.role
                if role != "user":
                    return
                if len(item.content) > 0 and item.content[0].type == "input_audio":
                    # Associate pending audio with this item_id
                    item_id = item.id
                    self._current_item_id = item_id  # Track current item_id for audio association
                    if self._pending_user_audio:
                        self._user_audio_buffer[item_id] = self._pending_user_audio
                        self._pending_user_audio = bytes()
                    return
                self._last_user_message = handle_conversation_item_message(item)
            elif item.type == "function_call_output":
                call_id = getattr(item, "call_id", None)
                output = getattr(item, "output", None)

                if call_id and output is not None:
                    if not isinstance(output, str):
                        output = str(output)

                    self._tool_call_outputs[call_id] = output

                    if call_id in self._tool_calls:
                        try:
                            tool_call = self._tool_calls[call_id]
                            tool_call.result(output)
                        except Exception as e:
                            scribe().warning(
                                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error setting tool call result: {str(e)}"
                            )
                    else:
                        try:
                            self._logger.tool_call_result(call_id, output)
                        except Exception as e:
                            scribe().warning(
                                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error setting tool call result via logger: {str(e)}"
                            )

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling conversation.item.added: {str(e)}"
            )

    async def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        """Handle response.created event to start a new trace and generation."""
        try:
            response = event.response

            # Ensure session exists
            if self._session is None:
                try:
                    self._session = self._logger.session(
                        {"id": self._session_id, "name": "OpenAI Realtime Session"}
                    )
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error creating session: {str(e)}"
                    )
                    return

            # If we have a pending trace with tool calls, we will continue using it instead of creating a new one
            if self._current_trace is not None and self._has_pending_tool_calls:
                self._has_pending_tool_calls = False
                self._is_continuing_trace = True
            else:
                # End previous trace if exists (normal case - no tool calls)
                if self._current_trace is not None:
                    try:
                        self._current_trace.end()
                    except Exception as e:
                        scribe().warning(
                            f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error ending previous trace: {str(e)}"
                        )

                # Create new trace for this interaction
                trace_id = str(uuid4())
                try:
                    self._current_trace = self._session.trace(
                        {"id": trace_id, "name": "Realtime Interaction"}
                    )
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error creating trace: {str(e)}"
                    )
                    return
                self._is_continuing_trace = False

                # Clear pending audio buffer when starting a new trace
                # (audio should have been associated with previous item by now)
                self._pending_user_audio = bytes()

            # Create generation
            generation_id = response.id or str(uuid4())

            # Extract model parameters from session config
            model_parameters: Dict[str, Any] = {}
            if self._session_config:
                model_parameters = {
                    k: v
                    for k, v in self._session_config.items()
                    if k not in ["model", "instructions", "modalities"]
                }

            # Create generation config
            gen_config: GenerationConfigDict = {
                "id": generation_id,
                "model": self._session_model or "unknown",
                "provider": "openai",
                "name": self._generation_name,
                "model_parameters": model_parameters,
                "messages": [],
            }

            # Add system message if instructions exist
            if self._session_config:
                instructions = self._session_config.get("instructions")
                if instructions:
                    gen_config["messages"].append(
                        {"role": "system", "content": instructions}
                    )

            if self._last_user_message:
                gen_config["messages"].append(
                    {"role": "user", "content": self._last_user_message}
                )

                # Only set trace input if this is a new trace (not continuing after tool calls)
                if self._current_trace is not None and not self._is_continuing_trace:
                    try:
                        self._current_trace.set_input(self._last_user_message)
                    except Exception as e:
                        scribe().warning(
                            f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error setting trace input: {str(e)}"
                        )
                self._last_user_message = None

            # This is done to add the tool call outputs to the next generation's messages
            if self._is_continuing_trace and self._tool_call_outputs:
                for call_id in sorted(self._tool_call_outputs.keys()):
                    output = self._tool_call_outputs[call_id]
                    gen_config["messages"].append(
                        GenerationRequestMessage(
                            role="tool",
                            content=output,
                        )
                    )
                # Clear tool call outputs after adding them to messages to avoid duplicates
                self._tool_call_outputs.clear()

            try:
                self._current_generation = self._current_trace.generation(gen_config)
                self._is_continuing_trace = False
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error creating generation: {str(e)}"
                )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling response.created: {str(e)}"
            )

    async def _handle_function_call_arguments_delta(self, event: Any) -> None:
        """Handle response.function_call_arguments.delta event to accumulate function call arguments."""
        try:
            call_id = getattr(event, "call_id", None)
            delta = getattr(event, "delta", None)

            if call_id and delta:
                if call_id not in self._function_call_arguments:
                    self._function_call_arguments[call_id] = ""
                self._function_call_arguments[call_id] += delta
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling function call arguments delta: {str(e)}"
            )

    async def _handle_function_call_arguments_done(self, event: Any) -> None:
        """Handle response.function_call_arguments.done event to create tool call."""
        try:
            call_id = getattr(event, "call_id", None)
            arguments = getattr(event, "arguments", None)
            item_id = getattr(event, "item_id", None)
            function_name = getattr(event, "name", None)

            if not call_id or not arguments:
                return

            # Get accumulated arguments or use the final arguments
            final_arguments = arguments or self._function_call_arguments.get(
                call_id, ""
            )

            # If still unknown, try parsing from arguments JSON
            if not function_name:
                try:
                    args_dict = json.loads(final_arguments)
                    # Some APIs include function name in the arguments
                    if "name" in args_dict:
                        function_name = args_dict["name"]
                except Exception:
                    pass

            # If we still don't have a name, use item_id or call_id as fallback
            if not function_name:
                if item_id:
                    function_name = f"function_{item_id[:8]}"
                else:
                    function_name = f"function_{call_id[:8]}"

            # Create tool call in the current trace
            if self._current_trace is not None:
                try:
                    tool_call_config = {
                        "id": call_id,
                        "name": function_name,
                        "args": final_arguments,
                    }
                    tool_call = self._current_trace.tool_call(tool_call_config)
                    # Store the tool call object so we can set its result later
                    self._tool_calls[call_id] = tool_call
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error creating tool call: {str(e)}"
                    )

            # Clean up
            if call_id in self._function_call_arguments:
                del self._function_call_arguments[call_id]
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling function call arguments done: {str(e)}"
            )

    async def _handle_response_output_audio_delta(
        self, event: ResponseAudioDeltaEvent
    ) -> None:
        """Handle response.output_audio.delta event to accumulate audio chunks."""
        try:
            if self._output_audio is None:
                self._output_audio = bytes()
            self._output_audio += base64.b64decode(event.delta)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling response.output_audio.delta: {str(e)}"
            )

    async def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        """Handle response.done event to log final result."""
        try:
            response = event.response
            response_id = response.id or str(uuid4())

            # Process all output items to extract text and tool calls
            response_text = None
            tool_calls = []

            output_items = response.output
            if output_items:
                for output_item in output_items:
                    if hasattr(output_item, "type"):
                        if output_item.type == "message":
                            # Text response
                            if response_text is None:
                                response_text = handle_text_response(output_item)
                        elif output_item.type == "function_call":
                            # Function call - extract call_id, name, and arguments
                            function_call_data = handle_function_call_response(
                                output_item
                            )
                            call_id = (
                                getattr(output_item, "call_id", None)
                                or getattr(output_item, "id", None)
                                or str(uuid4())
                            )

                            # Ensure arguments is a JSON string
                            arguments = function_call_data["arguments"]
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)

                            tool_call = {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": function_call_data["name"],
                                    "arguments": arguments,
                                },
                            }
                            tool_calls.append(tool_call)

            # Extract usage
            usage = response.usage
            usage_dict = None
            if usage:
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif isinstance(usage, dict):
                    usage_dict = usage

                # Realtime uses input_tokens/output_tokens, but we expect prompt_tokens/completion_tokens
                if usage_dict:
                    normalized_usage: Dict[str, int] = {
                        "prompt_tokens": int(usage_dict.get("input_tokens", 0) or 0),
                        "completion_tokens": int(
                            usage_dict.get("output_tokens", 0) or 0
                        ),
                        "total_tokens": int(usage_dict.get("total_tokens", 0) or 0),
                    }
                    usage_dict = normalized_usage

            # Build result
            result: Dict[str, Any] = {
                "id": response_id,
                "object": "realtime.response",
                "created": int(time.time()),
                "model": self._session_model or "unknown",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text if not tool_calls else None,
                            "tool_calls": tool_calls if tool_calls else None,
                        },
                        "finish_reason": "stop" if not tool_calls else "tool_calls",
                    }
                ],
            }

            if usage_dict:
                result["usage"] = usage_dict

            # Log result
            if self._current_generation is not None:
                try:
                    self._current_generation.result(result)

                    if self._output_audio is not None:
                        self._current_generation.add_attachment(
                            FileDataAttachment(
                                data=pcm16_to_wav_bytes(self._output_audio),
                                tags={"attach-to": "output"},
                                name="Assistant Audio Response",
                                timestamp=int(time.time()),
                            )
                        )
                        self._output_audio = None
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error logging generation result: {str(e)}"
                    )

            # Check if this response has tool calls
            has_tool_calls = len(tool_calls) > 0

            # If there are tool calls, don't end the trace yet - wait for the next response
            # that will contain the final assistant output
            if has_tool_calls:
                self._has_pending_tool_calls = True
                # Don't end trace or clean up - keep it open for the next response
            else:
                # No tool calls - this is the final response, end the trace
                if self._current_trace is not None:
                    try:
                        self._current_trace.end()
                    except Exception as e:
                        scribe().warning(
                            f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error ending trace: {str(e)}"
                        )

                # Clean up
                self._current_generation = None
                self._current_trace = None
                self._has_pending_tool_calls = False
                self._is_continuing_trace = False
                self._tool_calls.clear()
                self._tool_call_outputs.clear()
                # Clear last user message to prevent stale data in next turn
                self._last_user_message = None

            # Always clear function calls tracking
            self._function_calls.clear()
            self._function_call_arguments.clear()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling response.done: {str(e)}"
            )

    async def _handle_conversation_item_deleted(self, _event: Any) -> None:
        """Handle conversation.item.deleted event to clear stale user message references."""
        try:
            # When items are deleted (e.g., during summarization), clear _last_user_message
            # to prevent stale data from being used in the next generation.
            # The next response.created event will get the user message from the
            # conversation.item.created or conversation.item.input_audio_transcription.completed event.
            self._last_user_message = None
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling conversation.item.deleted: {str(e)}"
            )

    async def _handle_error(self, event: Any) -> None:
        """Handle realtime.error event."""
        try:
            error_obj = getattr(event, "error", None)
            if error_obj is None:
                return

            error_message = str(error_obj)
            if hasattr(error_obj, "message"):
                error_message = str(error_obj.message)
            elif isinstance(error_obj, dict):
                error_message = error_obj.get("message", str(error_obj))

            if self._current_generation is not None:
                try:
                    self._current_generation.error(
                        {
                            "message": error_message,
                            "type": getattr(type(error_obj), "__name__", None),
                        }
                    )
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error logging generation error: {str(e)}"
                    )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling error event: {str(e)}"
            )


class MaximOpenAIAsyncRealtimeManager(AsyncRealtimeConnectionManager):
    def __init__(self, client: AsyncOpenAI, logger: Logger, **kwargs):
        super().__init__(client=client, **kwargs)
        self._logger = logger
        self._extra_headers = kwargs.get("extra_headers", {})

    async def __aenter__(self) -> MaximOpenAIAsyncRealtimeConnection:
        """
        Override to return MaximOpenAIAsyncRealtimeConnection instead of base connection.
        """
        base_connection = await super().__aenter__()
        # Wrap the base connection with our instrumented connection
        # The parent's __init__ will create the resources (session, response, etc.) correctly
        wrapped_connection = MaximOpenAIAsyncRealtimeConnection(
            base_connection._connection,
            self._logger,
            extra_headers=self._extra_headers
            if hasattr(self, "_extra_headers")
            else None,
        )
        return wrapped_connection


class MaximOpenAIAsyncRealtime(AsyncRealtime):
    def __init__(self, client: AsyncOpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    def connect(
        self,
        *,
        call_id=omit,
        model=omit,
        extra_query=None,
        extra_headers=None,
        websocket_connection_options=None,
    ):
        """
        Override to return MaximOpenAIAsyncRealtimeManager instead of base manager.
        """
        return MaximOpenAIAsyncRealtimeManager(
            client=self._client,
            logger=self._logger,
            call_id=call_id,
            model=model,
            extra_query=extra_query if extra_query is not None else {},
            extra_headers=extra_headers if extra_headers is not None else {},
            websocket_connection_options=websocket_connection_options
            if websocket_connection_options is not None
            else {},
        )
