import functools
from ..logger import Logger, Generation, Trace
from ...scribe import scribe
from typing import Any, Optional
from fireworks.llm.LLM import ChatCompletion
from uuid import uuid4
from .utils import FireworksUtils
from .helpers import FireworksHelpers

_INSTRUMENTED = False

def instrument_fireworks(logger: Logger) -> None:
    """Patch Fireworks's chat completion methods for Maxim logging.
    
    This function instruments the Fireworks SDK by patching the chat completion
    methods to automatically log API calls, model parameters, and responses to
    Maxim. It supports both synchronous and asynchronous operations, streaming
    responses, and various Fireworks specific features.
    """

    global _INSTRUMENTED
    if _INSTRUMENTED:
        scribe().info("[MaximSDK] Fireworks already instrumented")
        return
    
    def wrap_create(create_func):
        """Wrapper for synchronous chat completion create method.
        
        This wrapper function intercepts synchronous chat completion requests
        to Fireworks and adds comprehensive logging capabilities while
        preserving the original API behavior.
        """

        @functools.wraps(create_func)
        def wrapper(self: ChatCompletion, *args: Any, **kwargs: Any):
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
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)
            
            # Try to get model from self (ChatCompletion instance)
            model = self._create_setup()
            if model is not None:
                model = model.split("/")[-1]

            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "fireworks",
                    "name": generation_name,
                    "model_parameters": FireworksUtils.get_model_params(**kwargs),
                    "messages": FireworksUtils.parse_message_param(messages),
                }
                generation = trace.generation(gen_config)
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )

            # Call the original Fireworks API method
            response = create_func(self, *args, **kwargs)
                
            try:
                if generation is not None: 
                    if is_streaming:
                        response = FireworksHelpers.sync_stream_helper(response, generation, trace, is_local_trace)
                    else:
                        generation.result(FireworksUtils.parse_completion(response))
                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(response.choices[0].message.content or "")
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
                )

            # Apply tags if provided
            if generation_tags is not None:
                generation.set_tags(generation_tags)
            if trace_tags is not None:
                trace.set_tags(trace_tags)

            return response
        
        return wrapper

    def wrap_acreate(acreate_func):
        """Wrapper for asynchronous chat completion create method.
        
        This wrapper function intercepts asynchronous chat completion requests
        to Fireworks and adds comprehensive logging capabilities while
        preserving the original API behavior.
        """
        @functools.wraps(acreate_func)
        async def wrapper(self: ChatCompletion, *args: Any, **kwargs: Any):
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
            final_trace_id = trace_id or str(uuid4())
            generation: Optional[Generation] = None
            trace: Optional[Trace] = None
            messages = kwargs.get("messages", None)
            is_streaming = kwargs.get("stream", False)

            # Try to get model from self (ChatCompletion instance)
            model = self._create_setup()
            if model is not None:
                model = model.split("/")[-1]

            # Initialize trace and generation for logging
            try:
                trace = logger.trace({"id": final_trace_id})
                gen_config = {
                    "id": str(uuid4()),
                    "model": model,
                    "provider": "fireworks",
                    "name": generation_name,
                    "model_parameters": FireworksUtils.get_model_params(**kwargs),
                    "messages": FireworksUtils.parse_message_param(messages),
                }
                generation = trace.generation(gen_config)
                
                # Check for image URLs in messages and add as attachments
                FireworksUtils.add_image_attachments_from_messages(generation, messages)
                        
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in generating content: {e}",
                )

            response = await acreate_func(self, *args, **kwargs)

            # Process response and log results
            try:
                if generation is not None: 
                    if is_streaming:
                        response = FireworksHelpers.async_stream_helper(response, generation, trace, is_local_trace)
                    else:
                        generation.result(FireworksUtils.parse_completion(response))
                        if is_local_trace and trace is not None:
                            if response.choices and len(response.choices) > 0:
                                trace.set_output(response.choices[0].message.content or "")
                            else:
                                trace.set_output("")
                            trace.end()
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK][FireworksInstrumentation] Error in logging generation: {e}",
                )

            # Apply tags if provided
            if generation_tags is not None:
                generation.set_tags(generation_tags)
            if trace_tags is not None:
                trace.set_tags(trace_tags)

            return response

        return wrapper
        
        
    ChatCompletion.create = wrap_create(ChatCompletion.create)
    ChatCompletion.acreate = wrap_acreate(ChatCompletion.acreate)
