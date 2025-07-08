from typing import Any, Optional, Union, Iterator
from uuid import uuid4

from together import Together
from together.resources.chat.completions import ChatCompletions
from together.types import CompletionResponse, CompletionChunk 

from ...scribe import scribe
from ..components.generation import Generation
from ..components.trace import Trace
from ..logger import Logger
from .utils import TogetherUtils
from ..logger import (
    Generation,
    GenerationConfigDict,
    Logger,
    Trace,
    TraceConfigDict,
)


class MaximTogetherChat:
    """Maxim-enhanced Together Chat client.

    This class wraps the Together Chat Completions resource to integrate with Maxim's
    logging and monitoring capabilities. It automatically tracks chat completions
    and logs them through the Maxim platform.

    The class handles trace management, generation logging, and error handling
    while maintaining compatibility with the original Together Chat API.

    Attributes:
        _completions (ChatCompletions): The wrapped chat completions instance.
        _logger (Logger): The Maxim logger instance for tracking interactions.
    """

    def __init__(self, completions: ChatCompletions, logger: Logger):
        """Initialize the Maxim Together Chat client.

        Args:
            completions (ChatCompletions): The Together chat completions instance.
            logger (Logger): The Maxim logger instance for tracking and
                logging chat interactions.
        """
        self._completions = completions
        self._logger = logger

    def create_non_stream(self, *args, **kwargs) -> Any:
        """Create a non-streaming chat completion with Maxim logging.
        """
        extra_headers = kwargs.get("extra_headers", None)
        trace_id = None
        generation_name = None
        if extra_headers is not None:
            trace_id = extra_headers.get("x-maxim-trace-id", None)
            generation_name = extra_headers.get("x-maxim-generation-name", None)
        is_local_trace = trace_id is None
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        messages = kwargs.get("messages", None)
        print(f"[NON-STREAM] Messages: {messages}")
        model = kwargs.get("model", None)
        try:
            trace = self._logger.trace(TraceConfigDict(id=final_trace_id))
            gen_config = GenerationConfigDict(
                id=str(uuid4()),
                model=model,
                provider="together",
                name=generation_name,
                model_parameters=TogetherUtils.get_model_params(**kwargs),
                messages=messages,
            )
            print(f"[NON-STREAM] Gen config: {gen_config}")
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][TogetherClient] Error in generating content: {str(e)}"
            )
            
        response = self._completions.create(*args, **kwargs)

        try:
            if generation is not None:
                generation.result(response)
            if is_local_trace and trace is not None:
                if response is not None:
                    trace.set_output(str(response.choices[0].text)) # type: ignore
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][TogetherClient] Error in logging generation: {str(e)}"
            )

        return response

    def create_stream(self, *args, **kwargs) -> Any:
        """Create a streaming chat completion with Maxim logging.
        """
        return self._completions.create(*args, **kwargs)

    def create(
            self,
            *args,
            model: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            logprobs: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None,
            presence_penalty: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            logit_bias: Optional[dict] = None,
            extra_headers: Optional[dict] = None,
            **kwargs) -> Union[CompletionResponse, Iterator[CompletionChunk]]:
        """Create a chat completion with Maxim logging.

        This method handles chat completion creation while automatically
        logging the interaction through Maxim. It manages trace creation,
        generation tracking, and error handling.

        Args:
            *args: Variable length argument list passed to the parent create method.
            **kwargs: Arbitrary keyword arguments passed to the parent create method.
                Special headers:
                - x-maxim-trace-id: Optional trace ID for associating with existing trace.
                - x-maxim-generation-name: Optional name for the generation.

        Returns:
            Any: The response from the Together API create method.

        Note:
            If logging fails, the method will still return the API response
            but will log a warning message.
        """
        
        is_streaming = kwargs.get("stream", False)
        # Add all parameters back to kwargs
        kwargs["max_tokens"] = max_tokens
        kwargs["model"] = model

        if stop_sequences is not None:
            kwargs["stop_sequences"] = stop_sequences
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k
        if top_p is not None:
            kwargs["top_p"] = top_p
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            kwargs["logit_bias"] = logit_bias
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers
            
        if is_streaming == True:
            return self.create_stream(*args, **kwargs)
        else:
            return self.create_non_stream(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped completions instance."""
        return getattr(self._completions, name)
        