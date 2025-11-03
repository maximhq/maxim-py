from openai import OpenAI
from openai.resources.realtime import Realtime
from openai.resources.realtime.realtime import RealtimeConnectionManager

from maxim.logger import Logger


class MaximOpenAIRealtimeConnectionManager(RealtimeConnectionManager):
    def __init__(self, client: OpenAI, logger: Logger):
        self._client = client
        self._logger = logger


class MaximOpenAIRealtime(Realtime):
    def __init__(self: Realtime, client: OpenAI, logger: Logger):
        self._client = client
        self._logger = logger
