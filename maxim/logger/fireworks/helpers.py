"""Helper functions for Fireworks AI SDK streaming operations.

This module provides helper functions for handling streaming responses from the
Fireworks AI SDK while maintaining proper integration with Maxim's logging system.
It handles both synchronous and asynchronous streaming operations, accumulating
content and usage data to provide complete logging information.
"""

from collections.abc import Iterator, AsyncGenerator, Generator
from fireworks.llm.LLM import ChatCompletionChunk
from ..logger import Generation, Trace
from ...scribe import scribe
from .utils import FireworksUtils

class FireworksHelpers:
    """Helper class for Fireworks AI SDK streaming operations.

    This class provides static helper methods for handling streaming responses
    from the Fireworks AI SDK. It ensures proper integration with Maxim's logging
    system by accumulating streaming content and usage data, then logging the
    complete response once the stream is finished.

    The class supports both synchronous and asynchronous streaming operations,
    maintaining the original streaming behavior while adding comprehensive logging
    capabilities.

    All methods are static and can be called directly on the class without
    instantiation.
    """

    @staticmethod
    def sync_stream_helper(
        stream: Iterator[ChatCompletionChunk],
        generation: Generation,
        trace: Trace,
        is_local_trace: bool,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Helper function for synchronous streaming with Maxim logging.

        This method wraps a synchronous streaming response from Fireworks AI to
        provide proper logging integration. It accumulates content and usage data
        from streaming chunks, then logs the complete response once the stream
        is finished.

        The method maintains the original streaming behavior by yielding each chunk
        as it arrives, while simultaneously collecting data for logging purposes.

        Args:
            stream (Iterator[ChatCompletionChunk]): The original streaming response
                from Fireworks AI containing ChatCompletionChunk objects.
            generation (Generation): The Maxim generation object to log the final
                result to. Can be None if logging is not available.
            trace (Trace): The Maxim trace object associated with this generation.
                Used for setting output and ending the trace.
            is_local_trace (bool): Whether this is a local trace that should be
                ended automatically when the stream completes.

        Yields:
            ChatCompletionChunk: Each chunk from the original stream is yielded
                unchanged to maintain the original streaming behavior.

        Returns:
            Generator[ChatCompletionChunk, None, None]: A generator that yields
                the same chunks as the input stream while providing logging.

        Raises:
            Exception: Any exceptions during streaming are caught and logged as
                warnings, but the stream continues to function normally.
        """
        accumulated_content = ""
        final_usage = None
        
        try:
            for chunk in stream:
                # Accumulate content from chunks
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        accumulated_content += delta.content
                
                # Collect usage data from chunks
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = chunk.usage
                
                yield chunk

            # After stream is complete, log the accumulated result
            if generation is not None:
                # Create a complete response object from accumulated data
                response_to_parse = FireworksUtils.parse_chunks_to_response(accumulated_content, final_usage)
                generation.result(FireworksUtils.parse_completion(response_to_parse))
            
            if is_local_trace and trace is not None:
                trace.set_output(accumulated_content)
                trace.end()

        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
            )
    
    @staticmethod
    async def async_stream_helper(
        stream: AsyncGenerator[ChatCompletionChunk, None],
        generation: Generation,
        trace: Trace,
        is_local_trace: bool,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Helper function for asynchronous streaming with Maxim logging.

        This method wraps an asynchronous streaming response from Fireworks AI to
        provide proper logging integration. It accumulates content and usage data
        from streaming chunks, then logs the complete response once the stream
        is finished.

        The method maintains the original async streaming behavior by yielding each
        chunk as it arrives, while simultaneously collecting data for logging purposes.

        Args:
            stream (AsyncGenerator[ChatCompletionChunk, None]): The original async
                streaming response from Fireworks AI containing ChatCompletionChunk objects.
            generation (Generation): The Maxim generation object to log the final
                result to. Can be None if logging is not available.
            trace (Trace): The Maxim trace object associated with this generation.
                Used for setting output and ending the trace.
            is_local_trace (bool): Whether this is a local trace that should be
                ended automatically when the stream completes.

        Yields:
            ChatCompletionChunk: Each chunk from the original async stream is yielded
                unchanged to maintain the original streaming behavior.

        Returns:
            AsyncGenerator[ChatCompletionChunk, None]: An async generator that yields
                the same chunks as the input stream while providing logging.

        Raises:
            Exception: Any exceptions during streaming are caught and logged as
                warnings, but the async stream continues to function normally.
        """
        accumulated_content = ""
        final_usage = None
        
        try:
            async for chunk in stream:
                # Accumulate content from chunks
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        accumulated_content += delta.content

                # Collect usage data from chunks
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = chunk.usage

                yield chunk

            # After async stream is complete, log the accumulated result
            if generation is not None:
                # Create a complete response object from accumulated data
                response_to_parse = FireworksUtils.parse_chunks_to_response(accumulated_content, final_usage)
                generation.result(FireworksUtils.parse_completion(response_to_parse))

            if is_local_trace and trace is not None:
                trace.set_output(accumulated_content)
                trace.end()

        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
            )
