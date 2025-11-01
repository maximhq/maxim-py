from openai import AsyncOpenAI
from openai.resources.realtime import AsyncRealtime
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
from openai._types import omit
from typing import AsyncIterator

from maxim.logger import Logger


class MaximOpenAIAsyncRealtimeConnection(AsyncRealtimeConnection):
    def __init__(self, connection, logger: Logger):
        super().__init__(connection)
        self._logger = logger

    async def __aiter__(self) -> AsyncIterator[RealtimeServerEvent]:
        """
        Override to intercept events and print their details.
        """
        from websockets.exceptions import ConnectionClosedOK

        try:
            async for event in super().__aiter__():
                # Print event details for observability
                event_type = getattr(event, "type", "unknown")
                try:
                    print(
                        f"[MaximSDK] Realtime Event - Type: {event_type}, Details: {vars(event)}"
                    )
                except Exception:
                    print(
                        f"[MaximSDK] Realtime Event - Type: {event_type}, Details: {event}"
                    )
                yield event
        except ConnectionClosedOK:
            return


class MaximOpenAIAsyncRealtimeManager(AsyncRealtimeConnectionManager):
    def __init__(self, client: AsyncOpenAI, logger: Logger, **kwargs):
        super().__init__(client=client, **kwargs)
        self._logger = logger

    async def __aenter__(self) -> MaximOpenAIAsyncRealtimeConnection:
        """
        Override to return MaximOpenAIAsyncRealtimeConnection instead of base connection.
        """
        base_connection = await super().__aenter__()
        # Wrap the base connection with our instrumented connection
        # The parent's __init__ will create the resources (session, response, etc.) correctly
        wrapped_connection = MaximOpenAIAsyncRealtimeConnection(
            base_connection._connection, self._logger
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
            websocket_connection_options=websocket_connection_options if websocket_connection_options is not None else {},
        )
