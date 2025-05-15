import json
import os
import unittest

from maxim.maxim import Config, Maxim
from maxim.models import QueryBuilder, VariableType

# reading testConfig.json and setting the values

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

# local config
apiKey = data["dev"]["apiKey"]
promptId = data["dev"]["promptId"]
baseUrl = data["dev"]["baseUrl"]
folderID = data["dev"]["folderId"]
promptChainId = data["dev"]["promptChainId"]


class TestPromptChains(unittest.TestCase):
    def setUp(self):
        self.maxim = Maxim(
            Config(api_key=apiKey, base_url=baseUrl, debug=True, prompt_management=True)
        )
        self.payload = {
            "input": {
                "type": VariableType.TEXT,
                "payload": "Hello",
            },
        }

    def test_getPromptChain_with_deployment_variables(self):
        promptChain = self.maxim.get_prompt_chain(
            promptChainId,
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if promptChain is None:
            raise Exception("Prompt chain not found")
        self.assertEqual(promptChain.prompt_chain_id, promptChainId)

    def test_prompt_chain_run(self) -> None:
        promptChain = self.maxim.get_prompt_chain(
            promptChainId,
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if promptChain is None:
            raise Exception("Prompt chain not found")
        result = promptChain.run("test")
        print(result)
