import logging
from typing import Dict, List, Optional, Union

from ..apis.maxim_apis import MaximAPI
from ..models import Message, Prompt, PromptResponse, ImageURL
from ..logger.components import Trace, Span, GenerationConfigDict
import uuid
import time
from dataclasses import asdict


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
    _parent: Optional[Union[Trace, Span]] = None
    _generation_config: Optional[GenerationConfigDict] = None

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

    def with_logger(
        self,
        parent: Union[Trace, Span],
        generation_config: Optional[GenerationConfigDict] = None,
    ) -> "RunnablePrompt":
        self._parent = parent
        self._generation_config = generation_config
        return self

    def _execute_prompt_with_logging(
        self,
        input: str,
        image_urls: Optional[List[ImageURL]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptResponse]:
        if not self._parent:
            return self.maxim_api.run_prompt_version(
                self.version_id, input, image_urls, variables
            )

        generation_id = (
            self._generation_config.get("id")
            if self._generation_config
            else str(uuid.uuid4())
        )

        # Initial config for generation
        gen_config = self._generation_config.copy() if self._generation_config else {}
        gen_config["id"] = generation_id
        if "model" not in gen_config:
            gen_config["model"] = self.model or ""
        if "provider" not in gen_config:
            gen_config["provider"] = self.provider or ""

        if "messages" not in gen_config:
            gen_config["messages"] = []

        if "model_parameters" not in gen_config:
            gen_config["model_parameters"] = self.model_parameters or {}

        generation = self._parent.generation(gen_config)

        try:
            result = self.maxim_api.run_prompt_version(
                self.version_id, input, image_urls, variables
            )

            if result:
                resolved_messages = result.resolved_messages or []
                for msg in resolved_messages:
                    # Handle legacy "payload" wrapper
                    if isinstance(msg, dict) and "payload" in msg:
                        msg = msg["payload"]

                    normalized_msg = msg.copy()
                    
                    # Ensure role and content are present
                    if "role" not in normalized_msg:
                        normalized_msg["role"] = "assistant" # Default if missing, though ideally shouldn't happen
                    if "content" not in normalized_msg:
                        normalized_msg["content"] = ""

                    content = normalized_msg["content"]
                    if isinstance(content, list):
                        new_content = []
                        for item in content:
                            if isinstance(item, dict):
                                new_item = item.copy()
                                if new_item.get("type") == "image_url":
                                    new_item["type"] = "image"
                                    if isinstance(new_item.get("image_url"), dict):
                                        new_item["image_url"] = new_item["image_url"].get("url", "")
                                new_content.append(new_item)
                            else:
                                new_content.append(item)
                        normalized_msg["content"] = new_content

                    generation.add_message(normalized_msg)

                if variables:
                    for k, v in variables.items():
                        str_v = str(v)
                        if str_v:
                            generation.add_tag(k, str_v)

                generation_result = {
                    "id": result.id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": result.model,
                    "choices": [asdict(c) for c in result.choices],
                    "usage": asdict(result.usage),
                }
                generation.result(generation_result)

            return result

        except Exception as e:
            generation.error(
                {
                    "message": str(e),
                    "code": "500",
                }
            )
            raise e

    def run(
        self,
        input: str,
        image_urls: Optional[List[ImageURL]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptResponse]:
        if self.maxim_api is None:
            logging.error("[MaximSDK] Invalid prompt. APIs are not initialized.")
            return None
        return self._execute_prompt_with_logging(input, image_urls, variables)
