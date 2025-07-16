import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from queue import Queue
from websockets.asyncio.client import ClientConnection
from ....logger import Logger
from .....scribe import scribe


class WebSocketEvent:
    """Represents a websocket event with metadata."""

    def __init__(
        self,
        event_type: str,
        data: Any,
        direction: str,
        timestamp: Optional[float] = None,
    ):
        self.event_type = event_type
        self.data = data
        self.direction = direction  # 'inbound' or 'outbound'
        self.timestamp = timestamp or time.time()
        self.event_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "direction": self.direction,
            "timestamp": self.timestamp,
        }


def connect_with_maxim_wrapper():
    pass


class OpenAIRealtimeWebsocketWrapper(ClientConnection):
    """
    A WebSocket ClientConnection that logs OpenAI Realtime API events while maintaining
    full compatibility with the websockets.ClientConnection interface.

    This wrapper extends websockets.ClientConnection and can be used as a drop-in replacement
    while adding OpenAI-specific logging and event processing capabilities.
    """

    def __init__(
        self, logger: Logger, session_id: Optional[str] = None, *args, **kwargs
    ):
        # OpenAI-specific setup
        self.maxim_logger = logger
        self.session_id = session_id or str(uuid.uuid4())

        # Event processing
        self.event_callbacks: List[Callable[[WebSocketEvent], None]] = []
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ws-processor"
        )

        # Stats
        self.events_processed = 0
        self.connection_start_time = time.time()

        # Call parent constructor
        super().__init__(*args, **kwargs)

        scribe().info(
            f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Initialized with session_id: {self.session_id}"
        )

    async def send(self, message, **kwargs):
        """Override send to log outbound messages."""
        try:
            # Process the message for logging
            await self._process_outbound_message(message)

            # Call the parent send method
            result = await super().send(message, **kwargs)

            scribe().debug(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Sent message successfully"
            )

            return result

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to send message: {e}"
            )
            raise

    async def recv(self, **kwargs):
        """Override recv to log inbound messages."""
        try:
            # Call the parent recv method
            message = await super().recv(**kwargs)

            # Process the message for logging
            await self._process_inbound_message(message)

            scribe().debug(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Received message successfully"
            )

            return message

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to receive message: {e}"
            )
            raise

    async def recv_streaming(self, **kwargs):
        """Override recv_streaming to log inbound streaming messages."""
        try:
            # Call the parent recv_streaming method
            stream = await super().recv_streaming(**kwargs)

            # TODO: Consider how to handle streaming message logging
            scribe().debug(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Received streaming message"
            )

            return stream

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to receive streaming message: {e}"
            )
            raise

    async def close(self, code=1000, reason="", **kwargs):
        """Override close to log connection closure and cleanup."""
        try:
            scribe().info(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Closing connection with code: {code}, reason: {reason}"
            )

            # Call the parent close method
            result = await super().close(code, reason, **kwargs)

            # Cleanup resources
            self._cleanup()

            return result

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Error during close: {e}"
            )
            raise

    async def ping(self, data=b"", **kwargs):
        """Override ping to log ping frames."""
        try:
            scribe().debug(f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Sending ping")

            # Call the parent ping method
            result = await super().ping(data, **kwargs)

            return result

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to send ping: {e}"
            )
            raise

    async def pong(self, data=b"", **kwargs):
        """Override pong to log pong frames."""
        try:
            scribe().debug(f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Sending pong")

            # Call the parent pong method
            result = await super().pong(data, **kwargs)

            return result

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to send pong: {e}"
            )
            raise

    def add_event_callback(self, callback: Callable[[WebSocketEvent], None]):
        """Add a callback function to be called for each processed event."""
        self.event_callbacks.append(callback)
        scribe().debug(
            f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Added event callback. Total callbacks: {len(self.event_callbacks)}"
        )

    def remove_event_callback(self, callback: Callable[[WebSocketEvent], None]):
        """Remove an event callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
            scribe().debug(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Removed event callback. Total callbacks: {len(self.event_callbacks)}"
            )

    async def _process_inbound_message(self, message: Any):
        """Process and log an inbound message."""
        try:
            # Parse the message
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    event_type = (
                        data.get("type", "unknown") if isinstance(data, dict) else "raw"
                    )
                except json.JSONDecodeError:
                    data = message
                    event_type = "raw"
            else:
                data = message
                event_type = "binary" if isinstance(message, bytes) else "raw"

            # Create event for processing
            event = WebSocketEvent(
                event_type=event_type, data=data, direction="inbound"
            )

            # Process event asynchronously
            self._process_event_async(event)
            self.events_processed += 1

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to process inbound message: {e}"
            )

    async def _process_outbound_message(self, message: Any):
        """Process and log an outbound message."""
        try:
            # Parse the message
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    event_type = (
                        data.get("type", "unknown") if isinstance(data, dict) else "raw"
                    )
                except json.JSONDecodeError:
                    data = message
                    event_type = "raw"
            else:
                data = message
                event_type = "binary" if isinstance(message, bytes) else "raw"

            # Create event for processing
            event = WebSocketEvent(
                event_type=event_type, data=data, direction="outbound"
            )

            # Process event asynchronously
            self._process_event_async(event)
            self.events_processed += 1

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to process outbound message: {e}"
            )

    def _process_event_async(self, event: WebSocketEvent):
        """Process a websocket event asynchronously."""
        try:
            # Submit to thread pool for parallel processing
            self.executor.submit(self._process_event_sync, event)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to submit event for processing: {e}"
            )

    def _process_event_sync(self, event: WebSocketEvent):
        """Synchronously process and log the event."""
        try:
            # Call custom callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Event callback failed: {e}"
                    )

            # Log to Maxim if logger is available
            if self.maxim_logger:
                try:
                    # TODO: Implement Maxim-specific logging here
                    pass
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Maxim logging failed: {e}"
                    )

            scribe().debug(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Processed {event.direction} event: {event.event_type}"
            )

        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Failed to process event {event.event_id}: {e}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get connection and processing statistics."""
        duration = time.time() - self.connection_start_time
        return {
            "session_id": self.session_id,
            "connection_duration": duration,
            "events_processed": self.events_processed,
            "active_callbacks": len(self.event_callbacks),
            "state": str(self.state) if hasattr(self, "state") else "unknown",
            "local_address": (
                str(self.local_address) if hasattr(self, "local_address") else None
            ),
            "remote_address": (
                str(self.remote_address) if hasattr(self, "remote_address") else None
            ),
        }

    def _cleanup(self):
        """Cleanup resources during connection closure."""
        try:
            # Shutdown event processor
            self.executor.shutdown(wait=False)

            stats = self.get_stats()
            scribe().info(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Connection closed. Final stats: {stats}"
            )
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Error during cleanup: {e}"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            # Note: In async context, close() should be awaited, but this is sync context
            self._cleanup()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][OpenAIRealtimeWebsocketWrapper] Error in context manager exit: {e}"
            )
