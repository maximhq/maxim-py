import unittest

import portkey_ai

from maxim import Config, Maxim
from maxim.logger import LoggerConfig
from maxim.logger.portkey import instrument_portkey
from maxim.logger.openai.client import MaximOpenAIClient, MaximOpenAIAsyncClient


logger = Maxim(Config(api_key="a", base_url="https://example.com")).logger(
    LoggerConfig(id="repo")
)


class TestPortkeyIntegration(unittest.TestCase):
    def test_instrument_portkey_sync(self):
        client = portkey_ai.Portkey(api_key="dummy")
        instrument_portkey(client, logger)
        self.assertIsInstance(client.openai_client, MaximOpenAIClient)

    def test_instrument_portkey_async(self):
        client = portkey_ai.AsyncPortkey(api_key="dummy")
        instrument_portkey(client, logger)
        self.assertIsInstance(client.openai_client, MaximOpenAIAsyncClient)


if __name__ == "__main__":
    unittest.main()
