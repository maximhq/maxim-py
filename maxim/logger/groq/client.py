"""Groq API instrumentation for Maxim logging.

This module provides instrumentation for the Groq SDK to integrate with Maxim's
logging and monitoring capabilities. It patches the Groq client methods to
automatically track API calls, model parameters, and responses.

The instrumentation supports both synchronous and asynchronous chat completions,
streaming responses, and various model parameters specific to Groq.
"""

import functools
from typing import Any, Optional
from uuid import uuid4
from groq.resources.chat import Completions, AsyncCompletions
from .utils import GroqUtils
from .helpers import GroqHelpers
from ..logger import Generation, Logger, Trace, GenerationConfigDict
from ...scribe import scribe

_INSTRUMENTED = False

def instrument_groq(logger: Logger) -> None:
    """Patch Groq's chat completion methods for Maxim logging.
    
    This function instruments the Groq SDK by patching the chat completion
    methods to automatically log API calls, model parameters, and responses to
    Maxim. It supports both synchronous and asynchronous operations, streaming
    responses, and various Groq specific features.
    
    The instrumentation is designed to be non-intrusive and maintains the original
    API behavior while adding comprehensive logging capabilities.
    
    Args:
        logger (Logger): The Maxim logger instance to use for tracking and
            logging API interactions. This logger will be used to create
            traces and generations for each API call.
    """

    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().debug("[MaximSDK] Groq already instrumented")
        return

    def wrap_sync_create(create_func):
        """Wrapper for synchronous chat completion create method.
        
        This wrapper function intercepts synchronous chat completion requests
        to Groq and adds comprehensive logging capabilities while
        preserving the original API behavior.
        
        Args:
            create_func: The original Groq chat completion create method
                to be wrapped with logging capabilities.
        
        Returns:
            Wrapped function that provides the same interface as the original
            but with added Maxim logging and monitoring.
        """

        @functools.wraps(create_func)
        def wrapper(self: Completions, *args: Any, **kwargs: Any):
            # Extract Maxim-specific headers for trace and generation configuration
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)

            # Initialize trace and generation for logging
            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=model or "",
                    provider="groq",
                    name=generation_name,
                    model_parameters=GroqUtils.get_model_params(**kwargs),
                    messages=GroqUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                GroqUtils.add_image_attachments_from_messages(generation, messages or [])

            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in generating content: {e}",
                )

            try:
                # Call the original Groq API method
                response = create_func(self, *args, **kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                # We will raise the error back to the caller and not handle it here
                raise
            
            # Process response and log results
            try:
                if generation is not None:
                    if is_streaming:
                        response = GroqHelpers.sync_stream_helper(response, generation, trace, is_local_trace)
                    else:
                        generation.result(GroqUtils.parse_completion(response))
                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(response.choices[0].message.content or "")
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in logging generation: {e}"
                )

            # Apply tags if provided
            if generation_tags is not None and generation is not None:
                for key, value in generation_tags.items():
                    generation.add_tag(key, value)
            if trace_tags is not None and trace is not None:
                for key, value in trace_tags.items():
                    trace.add_tag(key, value)

            return response

        return wrapper

    def wrap_async_create(create_func):
        """Wrapper for asynchronous chat completion create method.
        
        This wrapper function intercepts asynchronous chat completion requests
        to Groq and adds comprehensive logging capabilities while
        preserving the original API behavior and async semantics.
        
        Args:
            create_func: The original Groq async chat completion create method
                to be wrapped with logging capabilities.
        
        Returns:
            Wrapped async function that provides the same interface as the original
            but with added Maxim logging and monitoring.
        """

        @functools.wraps(create_func)
        async def wrapper(self: AsyncCompletions, *args: Any, **kwargs: Any):
            # Extract Maxim-specific headers for trace and generation configuration
            extra_headers = kwargs.get("extra_headers", None)
            trace_id = None
            generation_name = None
            generation_tags = None
            trace_tags = None

            if extra_headers is not None:
                trace_id = extra_headers.get("x-maxim-trace-id", None)
                generation_name = extra_headers.get("x-maxim-generation-name", None)
                generation_tags = extra_headers.get("x-maxim-generation-tags", None)
                trace_tags = extra_headers.get("x-maxim-trace-tags", None)

            # Determine if we need to create a new trace or use existing one
            is_local_trace = trace_id is None
            model = kwargs.get("model", None)
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)

            # Initialize trace and generation for logging
            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = GenerationConfigDict(
                    id=str(uuid4()),
                    model=model or "",
                    provider="groq",
                    name=generation_name,
                    model_parameters=GroqUtils.get_model_params(**kwargs),
                    messages=GroqUtils.parse_message_param(messages or []),
                )
                generation = trace.generation(gen_config)

                # Check for image URLs in messages and add as attachments
                GroqUtils.add_image_attachments_from_messages(generation, messages or [])

            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in generating content: {e}",
                )

            try:
                # Call the original Groq API method
                response = await create_func(self, *args, **kwargs)
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                # We will raise the error back to the caller and not handle it here
                raise

            # Process response and log results
            try:
                if generation is not None: 
                    if is_streaming:
                        response = GroqHelpers.async_stream_helper(response, generation, trace, is_local_trace)
                    else:
                        generation.result(GroqUtils.parse_completion(response))
                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(response.choices[0].message.content or "")
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                if generation is not None:
                    generation.error({"message": str(e)})
                scribe().warning(
                    f"[MaximSDK][GroqInstrumentation] Error in logging generation: {e}",
                )

            # Apply tags if provided
            if generation_tags is not None and generation is not None:
                for key, value in generation_tags.items():
                    generation.add_tag(key, value)
            if trace_tags is not None and trace is not None:
                for key, value in trace_tags.items():
                    trace.add_tag(key, value)

            return response

        return wrapper

    # Apply the patches to both sync and async chat completion methods
    setattr(Completions, 'create', wrap_sync_create(Completions.create))
    setattr(AsyncCompletions, 'create', wrap_async_create(AsyncCompletions.create))
    _INSTRUMENTED = True
