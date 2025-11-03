from typing import AsyncIterator, Optional, Dict, Any
from uuid import uuid4
import time
import json

from openai import AsyncOpenAI
from openai.resources.realtime import AsyncRealtime
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    SessionCreatedEvent,
    RealtimeServerEvent,
)
from openai._types import omit

from maxim.logger import Logger, Trace, Generation, Session
from maxim.logger.components.generation import GenerationConfigDict

from maxim.logger.openai.realtime_handlers.async_handlers import (
    handle_conversation_item_message,
    handle_function_call_response,
    handle_text_response,
)
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
        self._current_trace: Optional[Trace] = None
        self._current_generation: Optional[Generation] = None
        self._current_generation_id: Optional[str] = None
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
        ignored_events = ["response.output_text.delta", "response.function_call_arguments.delta"]
        if event.type not in ignored_events:
            print(f"{vars(event)}\n\n")

        try:
            event_type = event.type

            if event_type == "session.created":
                await self._handle_session_created(event)
            elif event_type == "conversation.item.added":
                await self._handle_conversation_item_created(event)
            elif event_type == "response.created":
                await self._handle_response_created(event)
            elif event_type == "response.function_call_arguments.delta":
                await self._handle_function_call_arguments_delta(event)
            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call_arguments_done(event)
            elif event_type == "response.done":
                await self._handle_response_done(event)
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
                self._session_config = session.model_copy()

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

    async def _handle_conversation_item_created(
        self, event: ConversationItemCreatedEvent
    ) -> None:
        """
        Handle conversation.item.added event to track user messages.
        This handles the extraction of the user message from the event because we do not receive the user message in the response.created event.
        """
        try:
            item = event.item
            
            if item.type == "message":
                role = item.role
                if role != "user":
                    return
                self._last_user_message = handle_conversation_item_message(item)

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

            # End previous trace if exists
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

            # Create generation
            generation_id = response.id or str(uuid4())
            self._current_generation_id = generation_id

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

                if self._current_trace is not None:
                    try:
                        self._current_trace.set_input(self._last_user_message)
                    except Exception as e:
                        scribe().warning(
                            f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error setting trace input: {str(e)}"
                        )
                self._last_user_message = None

            try:
                self._current_generation = self._current_trace.generation(gen_config)
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

            if not call_id or not arguments:
                return

            # Get accumulated arguments or use the final arguments
            final_arguments = arguments or self._function_call_arguments.get(
                call_id, ""
            )

            # Extract function name - try to get from stored function call data or parse from arguments
            function_name = "unknown"

            # Check if we have stored function call data
            if call_id in self._function_calls:
                stored_call = self._function_calls[call_id]
                function_name = stored_call.get("name", "unknown")

            # If still unknown, try parsing from arguments JSON
            if function_name == "unknown":
                try:
                    args_dict = json.loads(final_arguments)
                    # Some APIs include function name in the arguments
                    if "name" in args_dict:
                        function_name = args_dict["name"]
                except Exception:
                    pass

            # If we still don't have a name, use item_id or call_id as fallback
            if function_name == "unknown":
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
                    self._current_trace.tool_call(tool_call_config)
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

    async def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        """Handle response.done event to log final result."""
        try:
            response = event.response
            response_id = response.id or str(uuid4())

            # Process all output items to extract text and tool calls
            response_text = None
            tool_calls = []

            output_items = getattr(response, "output", [])
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
            usage = getattr(response, "usage", None)
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
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error logging generation result: {str(e)}"
                    )

            # End trace for this interaction
            if self._current_trace is not None:
                try:
                    self._current_trace.end()
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error ending trace: {str(e)}"
                    )

            # Clean up
            self._current_generation = None
            self._current_generation_id = None
            self._current_trace = None
            self._function_calls.clear()
            self._function_call_arguments.clear()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximOpenAIAsyncRealtimeConnection] Error handling response.done: {str(e)}"
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
