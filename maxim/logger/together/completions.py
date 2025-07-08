from typing import Optional, Union, Iterator
from uuid import uuid4

from together import Together
from together.resources.completions import Completions
from together.types import CompletionResponse, CompletionChunk 

from ...scribe import scribe
from ..components.generation import Generation
from ..components.trace import Trace
from .utils import TogetherUtils
from ..logger import (
    Generation,
    GenerationConfigDict,
    Logger,
    Trace,
    TraceConfigDict,
)


class MaximTogetherCompletions(Completions):
    """Maxim-enhanced Together Completions client.

    This class extends the Together Completions resource to integrate with Maxim's
    logging and monitoring capabilities. It automatically tracks completions
    and logs them through the Maxim platform.

    The class handles trace management, generation logging, and error handling
    while maintaining compatibility with the original Together Completions API.

    Attributes:
        _logger (Logger): The Maxim logger instance for tracking interactions.
    """

    def __init__(self, client: Together, logger: Logger):
        """Initialize the Maxim Together Completions client.

        Args:
            client (Together): The Together client instance.
            logger (Logger): The Maxim logger instance for tracking and
                logging completion interactions.
        """
        super().__init__(client.client)
        self._logger = logger
        
    def create_non_stream(self, *args, **kwargs) -> CompletionResponse:
        """Create a non-streaming completion with Maxim logging.
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
        prompt = kwargs.get("prompt", None)
        model = kwargs.get("model", None)
        try:
            trace = self._logger.trace(TraceConfigDict(id=final_trace_id))
            gen_config = GenerationConfigDict(
                id=str(uuid4()),
                model=model,
                provider="together",
                name=generation_name,
                model_parameters=TogetherUtils.get_model_params(**kwargs),
                messages=[{"role": "user", "content": prompt}],
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][TogetherClient] Error in generating content: {str(e)}"
            )
            
        response = super().create(*args, **kwargs)

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

    def create_stream(self, *args, **kwargs) -> Iterator[CompletionChunk]:
        """Create a streaming completion with Maxim logging.
        """
        pass
        
    # client.completions.create
    # It has prompt and model as required parameters
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
        """Create a completion with Maxim logging.

        This method handles completion creation while automatically
        logging the interaction through Maxim. It manages trace creation,
        generation tracking, and error handling.

        Args:
            *args: Variable length argument list passed to the parent create method.
            **kwargs: Arbitrary keyword arguments passed to the parent create method.
                Special headers:
                - x-maxim-trace-id: Optional trace ID for associating with existing trace.
                - x-maxim-generation-name: Optional name for the generation.

        Returns:
            CompletionResponse | Iterator[CompletionChunk]: The response from the Together API create method.
            # CompletionResponse when stream is False
            # Iterator[CompletionChunk] when stream is True

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

        # extra_headers = kwargs.get("extra_headers", None)
        # trace_id = None
        # generation_name = None
        # if extra_headers is not None:
        #     trace_id = extra_headers.get("x-maxim-trace-id", None)
        #     generation_name = extra_headers.get("x-maxim-generation-name", None)
        # is_local_trace = trace_id is None
        # final_trace_id = trace_id or str(uuid4())
        # generation: Optional[Generation] = None
        # trace: Optional[Trace] = None
        # prompt = kwargs.get("prompt", None)
        # model = kwargs.get("model", None)
        #
        # try:
        #     trace = self._logger.trace({"id": final_trace_id})
        #     gen_config = {
        #         "id": str(uuid4()),
        #         "model": model,
        #         "provider": "together",
        #         "name": generation_name,
        #         "model_parameters": TogetherUtils.get_model_params(**kwargs),
        #         "prompt": prompt,
        #     }
        #     generation = trace.generation(gen_config)
        # except Exception as e:
        #     scribe().warning(
        #         f"[MaximSDK][TogetherClient] Error in generating content: {str(e)}"
        #     )
        #
        # response = super().create(*args, **kwargs)
        #
        # try:
        #     if generation is not None:
        #         generation.result(TogetherUtils.parse_completion(response))
        #     if is_local_trace and trace is not None:
        #         trace.set_output(response.choices[0].text or "")
        #         trace.end()
        # except Exception as e:
        #     scribe().warning(
        #         f"[MaximSDK][TogetherClient] Error in logging generation: {str(e)}"
        #     )
        #
        # return response
        #