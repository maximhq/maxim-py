import logging
from typing import List, Optional

from ..apis.maxim_apis import MaximAPI
from ..models import AgentResponse, Node, PromptChain


class RunnablePromptChain:
    maxim_api: MaximAPI
    prompt_chain_id: str
    version: int
    version_id: str
    nodes: List[Node]

    def __init__(self, prompt_chain: PromptChain, maxim_api: MaximAPI):
        self.prompt_chain_id = prompt_chain.prompt_chain_id
        self.version = prompt_chain.version
        self.version_id = prompt_chain.version_id
        self.nodes = prompt_chain.nodes
        self.maxim_api = maxim_api

    def run(
        self, input: str, variables: Optional[dict[str, str]] = None
    ) -> Optional[AgentResponse]:
        if self.maxim_api is None:
            logging.error("[MaximSDK] Invalid prompt chain. APIs are not initialized.")
            return None
        return self.maxim_api.run_prompt_chain_version(
            self.version_id, input, variables
        )
