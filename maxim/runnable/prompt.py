import logging
from typing import Dict, List, Optional, Union

from ..apis.maxim_apis import MaximAPI
from ..models import Message, Prompt, PromptResponse, ImageURL


class RunnablePrompt:
    maxim_api: MaximAPI
    prompt_id: str
    version_id: str
    messages: List[Message]
    model_parameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: Optional[str] = None
    provider: Optional[str] = None
    deployment_id: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    def __init__(self, prompt: Prompt, maxim_api: MaximAPI):
        self.prompt_id = prompt.prompt_id
        self.version_id = prompt.version_id
        self.messages = prompt.messages
        self.model_parameters = prompt.model_parameters
        self.model = prompt.model
        self.provider = prompt.provider
        self.deployment_id = prompt.deployment_id
        self.tags = prompt.tags
        self.maxim_api = maxim_api

    def run(
        self,
        input: str,
        image_urls: Optional[List[ImageURL]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptResponse]:
        if self.maxim_api is None:
            logging.error("[MaximSDK] Invalid prompt. APIs are not initialized.")
            return None
        return self.maxim_api.run_prompt_version(
            self.version_id, input, image_urls, variables
        )
