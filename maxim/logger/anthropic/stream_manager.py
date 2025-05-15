from types import TracebackType
from typing import Callable, Union

from anthropic import MessageStreamEvent, MessageStreamManager
from anthropic.lib.streaming import MessageStream

from ...scribe import scribe


class TextStreamWrapper:
    def __init__(self, wrapper, callback):
        self._wrapper = wrapper
        self._callback = callback

    def __iter__(self):
        return self

    def __next__(self):
        stream = self._wrapper.__enter__()
        event = stream.__next__()
        self._callback(event)
        if event.type == "text":
            if event.text is not None:
                return event.text
        return ""


class StreamWrapper:
    def __init__(
        self,
        mgr: MessageStreamManager,
        callback: Callable[[MessageStreamEvent], None],
    ) -> None:
        self.__mgr = mgr
        self.__callback = callback

    def __enter__(self) -> MessageStream:
        stream = self.__mgr.__enter__()

        # Create a wrapper that processes chunks while still yielding them
        class StreamWithCallback(MessageStream):
            def __init__(self, stream, callback):
                self._stream = stream
                self._callback = callback

            def __iter__(self):
                return self

            def __next__(self):
                chunk = next(self._stream)
                try:
                    self._callback(chunk)
                except Exception as e:
                    scribe().error(f"Error in callback: {e}")
                return chunk

            @property
            def text_stream(self) -> TextStreamWrapper:  # type: ignore
                # Return a wrapper around the original text_stream that calls the callback
                return TextStreamWrapper(self, self._callback)

        return StreamWithCallback(stream, self.__callback)

    def __exit__(
        self,
        exc_type: Union[type[BaseException], None],
        exc: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ) -> None:
        self.__mgr.__exit__(exc_type, exc, exc_tb)
