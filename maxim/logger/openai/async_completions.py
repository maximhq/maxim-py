from typing import Optional
from uuid import uuid4

from openai import AsyncOpenAI
from openai.resources.chat import AsyncCompletions
from typing_extensions import override

from ...scribe import scribe
from ..logger import Generation, GenerationConfig, Logger, Trace, TraceConfig
from .utils import OpenAIUtils


class MaximAsyncOpenAIChatCompletions(AsyncCompletions):
    def __init__(self, client: AsyncOpenAI, logger: Logger):
        super().__init__(client=client)
        self._logger = logger

    @override
    async def create(self, *args, **kwargs):
        metadata = kwargs.get("metadata", None)
        trace_id = None
        generation_name = None
        maxim_metadata = None
        if metadata is not None:
            maxim_metadata = metadata.get("maxim", None)
            if maxim_metadata is not None:
                trace_id = maxim_metadata.get("trace_id", None)
                generation_name = maxim_metadata.get("generation_name", None)
        is_local_trace = trace_id is None
        model = kwargs.get("model", None)
        final_trace_id = trace_id or str(uuid4())
        generation: Optional[Generation] = None
        trace: Optional[Trace] = None
        messages = kwargs.get("messages", None)
        try:
            trace = self._logger.trace(TraceConfig(id=final_trace_id))
            gen_config = GenerationConfig(
                id=str(uuid4()),
                model=model,
                provider="openai",
                name=generation_name,
                model_parameters=OpenAIUtils.get_model_params(**kwargs),
                messages=OpenAIUtils.parse_message_param(messages),
            )
            generation = trace.generation(gen_config)
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in generating content: {str(e)}"
            )

        response = await super().create(*args, **kwargs)

        try:
            if generation is not None:
                generation.result(OpenAIUtils.parse_completion(response))
            if is_local_trace and trace is not None:
                trace.set_output(response.choices[0].message.content or "")
                trace.end()
        except Exception as e:
            scribe().warning(
                f"[MaximSDK][MaximAsyncOpenAIChatCompletions] Error in logging generation: {str(e)}"
            )

        return response
