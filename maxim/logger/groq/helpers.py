"""Helper functions for Groq SDK streaming operations.

This module provides helper functions for handling streaming responses from the
Groq SDK while maintaining proper integration with Maxim's logging system.
It handles both synchronous and asynchronous streaming operations, accumulating
content and usage data to provide complete logging information.
"""

from typing import Optional
from collections.abc import Generator, AsyncGenerator

from groq._streaming import Stream, AsyncStream
from groq.types.chat import ChatCompletionChunk

from .utils import GroqUtils
from ..logger import Generation, Trace
from ...scribe import scribe
from uuid import uuid4


class GroqHelpers:
    """Helper class for Groq SDK streaming operations.

    This class provides static helper methods for handling streaming responses
    from the Groq SDK. It ensures proper integration with Maxim's logging
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
        stream: Stream[ChatCompletionChunk],
        generation: Generation,
        trace: Optional[Trace],
        is_local_trace: bool,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Helper function for synchronous streaming with Maxim logging.

        This method wraps a synchronous streaming response from Groq to
        provide proper logging integration. It accumulates content and usage data
        from streaming chunks, then logs the complete response once the stream
        is finished.

        The method maintains the original streaming behavior by yielding each chunk
        as it arrives, while simultaneously collecting data for logging purposes.

        Args:
            stream (Stream[ChatCompletionChunk]): The original streaming response
                from Groq containing ChatCompletionChunk objects.
            generation (Generation): The Maxim generation object to log the final
                result to. Can be None if logging is not available.
            trace (Optional[Trace]): The Maxim trace object associated with this
                generation. Used for setting output and ending the trace.
            is_local_trace (bool): Whether this is a local trace that should be
                ended automatically when the stream completes.

        Yields:
            ChatCompletionChunk: Each chunk from the original stream is yielded
                unchanged to maintain the original streaming behavior.

        Returns:
            Generator[ChatCompletionChunk, None, None]: A generator that yields
                the same chunks as the input stream while providing logging.
        """

        accumulated_content = ""
        final_usage = None
        collected_chunks: list[ChatCompletionChunk] = []
        tool_calls = []

        try:
            for chunk in stream:
                # Accumulate content from chunks
                collected_chunks.append(chunk)
                yield chunk

            for chunk in collected_chunks:
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content

                # Collect usage data from chunks
                if (
                    hasattr(chunk, "x_groq")
                    and chunk.x_groq
                    and hasattr(chunk.x_groq, "usage")
                    and chunk.x_groq.usage
                ):
                    final_usage = chunk.x_groq.usage
                elif hasattr(chunk, "usage") and chunk.usage:
                    final_usage = chunk.usage

                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.tool_calls:
                        tool_calls.append(chunk.choices[0].delta.tool_calls)

            # After stream is complete, log the accumulated result
            response_to_parse = GroqUtils.parse_chunks_to_response(
                accumulated_content, final_usage, tool_calls
            )
            parsed = GroqUtils.parse_completion(response_to_parse)
            generation.result(parsed)

            # Create ToolCall entities on the trace when tools are used.
            try:
                if trace is not None:
                    tool_calls_flat = GroqUtils.extract_tool_calls_from_completion(
                        response_to_parse
                    )
                    for tc in tool_calls_flat:
                        fn = (
                            getattr(tc, "function", None)
                            if hasattr(tc, "function")
                            else (tc.get("function") if isinstance(tc, dict) else None)
                        )
                        tool_call_id = (
                            getattr(tc, "id", None)
                            if hasattr(tc, "id")
                            else (tc.get("id") if isinstance(tc, dict) else None)
                        )
                        if not isinstance(tool_call_id, str) or not tool_call_id:
                            tool_call_id = str(uuid4())
                        tool_name = (
                            getattr(fn, "name", "unknown")
                            if fn is not None and hasattr(fn, "name")
                            else (
                                fn.get("name", "unknown")
                                if isinstance(fn, dict)
                                else "unknown"
                            )
                        )
                        tool_args = (
                            getattr(fn, "arguments", "")
                            if fn is not None and hasattr(fn, "arguments")
                            else (
                                fn.get("arguments", "") if isinstance(fn, dict) else ""
                            )
                        )

                        _ = trace.tool_call(
                            {
                                "id": tool_call_id,
                                "name": tool_name,
                                "description": "Groq tool call",
                                "args": tool_args,
                            }
                        )
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error creating tool_call entities from stream: {e}"
                )

            if is_local_trace and trace is not None:
                trace.set_output(accumulated_content)
                trace.end()

        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][GroqInstrumentation] Error in streaming generation: {e}",
            )

    @staticmethod
    async def async_stream_helper(
        stream: AsyncStream[ChatCompletionChunk],
        generation: Generation,
        trace: Optional[Trace],
        is_local_trace: bool,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Helper function for asynchronous streaming with Maxim logging.

        This method wraps an asynchronous streaming response from Groq to
        provide proper logging integration. It accumulates content and usage data
        from streaming chunks, then logs the complete response once the stream
        is finished.

        The method maintains the original async streaming behavior by yielding each
        chunk as it arrives, while simultaneously collecting data for logging purposes.

        Args:
            stream (AsyncStream[ChatCompletionChunk]): The original async
                streaming response from Groq containing ChatCompletionChunk objects.
            generation (Generation): The Maxim generation object to log the final
                result to. Can be None if logging is not available.
            trace (Optional[Trace]): The Maxim trace object associated with this
                generation. Used for setting output and ending the trace.
            is_local_trace (bool): Whether this is a local trace that should be
                ended automatically when the stream completes.

        Yields:
            ChatCompletionChunk: Each chunk from the original async stream is yielded
                unchanged to maintain the original streaming behavior.

        Returns:
            AsyncGenerator[ChatCompletionChunk, None]: An async generator that yields
                the same chunks as the input stream while providing logging.
        """

        accumulated_content = ""
        final_usage = None
        collected_chunks: list[ChatCompletionChunk] = []
        tool_calls = []

        try:
            async for chunk in stream:
                # Accumulate content from chunks
                collected_chunks.append(chunk)
                yield chunk

            for chunk in collected_chunks:
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content

                # Collect usage data from chunks
                if (
                    hasattr(chunk, "x_groq")
                    and chunk.x_groq
                    and hasattr(chunk.x_groq, "usage")
                    and chunk.x_groq.usage
                ):
                    final_usage = chunk.x_groq.usage
                elif hasattr(chunk, "usage") and chunk.usage:
                    final_usage = chunk.usage

                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    if chunk.choices[0].delta.tool_calls:
                        tool_calls.append(chunk.choices[0].delta.tool_calls)

            # Create a proper response object with real attributes
            response_to_parse = GroqUtils.parse_chunks_to_response(
                accumulated_content,
                final_usage,
                tool_calls,
            )
            parsed = GroqUtils.parse_completion(response_to_parse)
            generation.result(parsed)

            # Create ToolCall entities on the trace when tools are used.
            try:
                if trace is not None:
                    tool_calls_flat = GroqUtils.extract_tool_calls_from_completion(
                        response_to_parse
                    )
                    for tc in tool_calls_flat:
                        fn = (
                            getattr(tc, "function", None)
                            if hasattr(tc, "function")
                            else (tc.get("function") if isinstance(tc, dict) else None)
                        )
                        tool_call_id = (
                            getattr(tc, "id", None)
                            if hasattr(tc, "id")
                            else (tc.get("id") if isinstance(tc, dict) else None)
                        )
                        if not isinstance(tool_call_id, str) or not tool_call_id:
                            tool_call_id = str(uuid4())
                        tool_name = (
                            getattr(fn, "name", "unknown")
                            if fn is not None and hasattr(fn, "name")
                            else (
                                fn.get("name", "unknown")
                                if isinstance(fn, dict)
                                else "unknown"
                            )
                        )
                        tool_args = (
                            getattr(fn, "arguments", "")
                            if fn is not None and hasattr(fn, "arguments")
                            else (
                                fn.get("arguments", "") if isinstance(fn, dict) else ""
                            )
                        )

                        _ = trace.tool_call(
                            {
                                "id": tool_call_id,
                                "name": tool_name,
                                "description": "Groq tool call",
                                "args": tool_args,
                            }
                        )
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error creating tool_call entities from async stream: {e}"
                )

            if is_local_trace and trace is not None:
                trace.set_output(accumulated_content)
                trace.end()

        except Exception as e:
            generation.error({"message": str(e)})
            scribe().warning(
                f"[MaximSDK][GroqInstrumentation] Error in async streaming generation: {e}",
            )
