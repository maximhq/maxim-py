from types import TracebackType
from typing import Callable, Union

from anthropic import MessageStreamManager, MessageStream, MessageStreamEvent


from ...scribe import scribe


class TextStreamWrapper:
    """Wrapper for text stream events with callback support.

    This class wraps a text stream to provide callback functionality
    for processing text events. It implements the iterator protocol
    to allow for seamless iteration over text content.

    Attributes:
        _wrapper: The underlying stream wrapper.
        _callback: Callback function to process stream events.
    """

    def __init__(self, wrapper, callback):
        """Initialize the text stream wrapper.

        Args:
            wrapper: The underlying stream wrapper to delegate to.
            callback: Callback function to be called for each event.
        """
        self._wrapper = wrapper
        self._callback = callback

    def __iter__(self):
        """Return the iterator object.

        Returns:
            TextStreamWrapper: Self as the iterator.
        """
        return self

    def __next__(self):
        """Get the next text content from the stream.

        This method processes the next event from the stream, calls the
        callback function, and returns the text content if available.

        Returns:
            str: The text content from the event, or empty string if no text.

        Raises:
            StopIteration: When the stream is exhausted.
        """
        stream = self._wrapper.__enter__()
        event = stream.__next__()
        self._callback(event)
        if event.type == "text":
            if event.text is not None:
                return event.text
        return ""


class StreamWrapper(MessageStreamManager):
    """Wrapper for Anthropic MessageStreamManager with callback support.

    This class wraps the Anthropic MessageStreamManager to provide callback
    functionality for processing stream events. It maintains compatibility
    with the original API while adding the ability to execute custom callbacks
    for each stream event.

    The wrapper implements the context manager protocol and iterator protocol
    to provide seamless integration with existing code.

    Attributes:
        _mgr (MessageStreamManager): The underlying stream manager.
        _callback (Callable): Callback function for processing events.
        _stream (MessageStream): The active stream instance.
    """

    def __init__(
        self,
        mgr: MessageStreamManager,
        callback: Callable[[MessageStreamEvent], None],
    ) -> None:
        """Initialize the stream wrapper.

        Args:
            mgr (MessageStreamManager): The underlying stream manager to wrap.
            callback (Callable[[MessageStreamEvent], None]): Callback function
                to be called for each stream event.

        Note:
            Does not call super().__init__() since we're wrapping another manager.
        """
        # Do not call super().__init__() since we're wrapping another manager
        self._mgr = mgr
        self._callback = callback
        self._stream = None

    def __enter__(self) -> MessageStream:
        """Enter the stream context and return a callback-enabled stream.

        This method creates a custom MessageStream that wraps the original
        stream to add callback functionality for each event.

        Returns:
            MessageStream: A stream instance with callback support.
        """
        stream = self._mgr.__enter__()

        class StreamWithCallback(MessageStream):
            """Internal MessageStream subclass with callback support.

            This class wraps a MessageStream to add callback functionality
            while maintaining the original stream interface.
            """

            def __init__(self, stream, callback):
                """Initialize the stream with callback.

                Args:
                    stream: The original MessageStream instance.
                    callback: Callback function for processing events.
                """
                self._stream = stream
                self._callback = callback

            def __iter__(self):
                """Return the iterator object.

                Returns:
                    StreamWithCallback: Self as the iterator.
                """
                return self

            def __next__(self):
                """Get the next event from the stream and process it.

                This method gets the next chunk from the stream, executes
                the callback function, and returns the chunk.

                Returns:
                    MessageStreamEvent: The next event from the stream.

                Raises:
                    StopIteration: When the stream is exhausted.
                """
                chunk = next(self._stream)
                try:
                    self._callback(chunk)
                except Exception as e:
                    scribe().error(f"Error in callback: {e}")
                return chunk

            @property
            def text_stream(self) -> TextStreamWrapper:  # type: ignore
                """Get a text-only stream wrapper.

                Returns:
                    TextStreamWrapper: A wrapper that extracts only text content
                        from the stream events.
                """
                return TextStreamWrapper(self, self._callback)

        self._stream = StreamWithCallback(stream, self._callback)
        return self._stream

    def __exit__(
        self,
        exc_type: Union[type[BaseException], None],
        exc: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ) -> None:
        """Exit the stream context.

        This method delegates the context exit to the underlying stream manager
        to ensure proper cleanup and resource management.

        Args:
            exc_type: The exception type if an exception occurred.
            exc: The exception instance if an exception occurred.
            exc_tb: The traceback if an exception occurred.
        """
        self._mgr.__exit__(exc_type, exc, exc_tb)

    def __iter__(self):
        """Return the iterator object.

        If no stream is active, this method will automatically enter
        the context to create one.

        Returns:
            StreamWrapper: Self as the iterator.
        """
        if self._stream is None:
            self.__enter__()
        return self

    def __next__(self):
        """Get the next event from the stream.

        If no stream is active, this method will automatically enter
        the context to create one.

        Returns:
            MessageStreamEvent: The next event from the stream.

        Raises:
            StopIteration: When the stream is exhausted.
        """
        if self._stream is None:
            self.__enter__()
        return next(self._stream)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped manager.

        This method ensures that any attributes or methods not explicitly
        defined in this wrapper are delegated to the underlying stream manager.

        Args:
            name (str): The attribute name to access.

        Returns:
            Any: The attribute value from the underlying manager.
        """
        # Delegate attribute access to the wrapped manager
        return getattr(self._mgr, name)
