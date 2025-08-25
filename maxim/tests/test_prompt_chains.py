import os
import unittest

from maxim.maxim import Config, Maxim
from maxim.models import QueryBuilder, VariableType
from dotenv import load_dotenv

load_dotenv()

# local config using prod environment
apiKey = os.getenv("MAXIM_API_KEY")
promptId = os.getenv("MAXIM_PROMPT_ID")
baseUrl = os.getenv("MAXIM_BASE_URL")
folderID = os.getenv("MAXIM_FOLDER_ID")
promptChainId = os.getenv("MAXIM_PROMPT_CHAIN_ID")


class TestPromptChains(unittest.TestCase):
    def setUp(self):
        # Clear singleton instance if it exists
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            Config(api_key=apiKey, base_url=baseUrl, debug=True, prompt_management=True)
        )
        self.payload = {
            "input": {
                "type": VariableType.TEXT,
                "payload": "Hello",
            },
        }

    def tearDown(self):
        # Clean up the Maxim instance
        if hasattr(self, "maxim"):
            self.maxim.cleanup()

        # Clear singleton instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

    def test_getPromptChain_with_deployment_variables(self):
        promptChain = self.maxim.get_prompt_chain(
            promptChainId,
            QueryBuilder().and_().deployment_var("Environment", "Prod").build(),
        )
        if promptChain is None:
            raise Exception("Prompt chain not found")
        self.assertEqual(promptChain.prompt_chain_id, promptChainId)

    def test_prompt_chain_run(self) -> None:
        promptChain = self.maxim.get_prompt_chain(
            promptChainId,
            QueryBuilder().and_().deployment_var("Environment", "Prod").build(),
        )
        if promptChain is None:
            raise Exception("Prompt chain not found")
        result = promptChain.run("test")
        print(result)
