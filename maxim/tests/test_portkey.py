import unittest

import portkey_ai

from maxim import Maxim
from maxim.logger.openai.client import MaximOpenAIAsyncClient, MaximOpenAIClient
from maxim.logger.portkey import instrument_portkey

logger = Maxim().logger()


class TestPortkeyIntegration(unittest.TestCase):
    def test_instrument_portkey_sync(self):
        client = portkey_ai.Portkey(api_key="/FGpTnLIT91fFT8XmIljONdo17aY",virtual_key="open-ai-virtual-57f8a1")
        instrument_portkey(client, logger)
        self.assertIsInstance(client.openai_client, MaximOpenAIClient)
        response = client.chat.completions.create(messages=[{"role": "user", "content": "Tell me what is big bang theory in 100 words?"}], model="gpt-4o", max_tokens=1000)
        print(response)

    def test_instrument_portkey_async(self):
        client = portkey_ai.AsyncPortkey(api_key="/FGpTnLIT91fFT8XmIljONdo17aY",virtual_key="open-ai-virtual-57f8a1")
        instrument_portkey(client, logger)
        self.assertIsInstance(client.openai_client, MaximOpenAIAsyncClient)

