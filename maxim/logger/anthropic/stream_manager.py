from types import TracebackType
from typing import Callable, Union

from anthropic import MessageStreamManager, MessageStream, MessageStreamEvent


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


class StreamWrapper(MessageStreamManager):
    def __init__(
        self,
        mgr: MessageStreamManager,
        callback: Callable[[MessageStreamEvent], None],
    ) -> None:
        # Do not call super().__init__() since we're wrapping another manager
        self._mgr = mgr
        self._callback = callback
        self._stream = None

    def __enter__(self) -> MessageStream:
        stream = self._mgr.__enter__()

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
                return TextStreamWrapper(self, self._callback)

        self._stream = StreamWithCallback(stream, self._callback)
        return self._stream

    def __exit__(
        self,
        exc_type: Union[type[BaseException], None],
        exc: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ) -> None:
        self._mgr.__exit__(exc_type, exc, exc_tb)

    def __iter__(self):
        if self._stream is None:
            self.__enter__()
        return self

    def __next__(self):
        if self._stream is None:
            self.__enter__()
        return next(self._stream)

    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped manager
        return getattr(self._mgr, name)
