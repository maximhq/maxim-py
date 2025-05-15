import os

from litellm.integrations.custom_logger import CustomLogger

from ...maxim import Config, Maxim
from ..litellm import MaximLiteLLMTracer
from ..logger import LoggerConfig


class MaximLiteLLMProxyTracer(CustomLogger):
    """
    Custom logger for Litellm Proxy.
    """

    def __init__(self):
        super().__init__()
        maxim_api_key = os.getenv("MAXIM_API_KEY")
        if maxim_api_key is None:
            raise ValueError("MAXIM_API_KEY is not set")
        maxim_log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")
        if maxim_log_repo_id is None:
            raise ValueError("MAXIM_LOG_REPO_ID is not set")
        self.logger = Maxim(Config(api_key=maxim_api_key)).logger(
            LoggerConfig(id=maxim_log_repo_id)
        )
        self.litellm_tracer = MaximLiteLLMTracer(self.logger)

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Runs when a LLM call starts.
        """
        self.litellm_tracer.log_pre_api_call(model, messages, kwargs)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        self.litellm_tracer.log_success_event(
            kwargs, response_obj, start_time, end_time
        )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call fails.
        """
        self.litellm_tracer.log_failure_event(
            kwargs, response_obj, start_time, end_time
        )

    async def async_log_pre_api_call(self, model, messages, kwargs):
        """
        Runs when a LLM call starts.
        """
        await self.litellm_tracer.async_log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call succeeds.
        """
        await self.litellm_tracer.async_log_success_event(
            kwargs, response_obj, start_time, end_time
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Runs when a LLM call fails.
        """
        await self.litellm_tracer.async_log_failure_event(
            kwargs, response_obj, start_time, end_time
        )