import logging
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Callable, Dict, List, Optional, Union

from ..apis.maxim_apis import MaximAPI
from ..logger import GenerationConfig, Span, Trace
from ..models import ImageURL, Message, Prompt, PromptResponse


def _execute_prompt_with_logging(
    parent: Optional[Union[Trace, Span]],
    generation_config: Optional[GenerationConfig],
    prompt: Prompt,
    options: Optional[Dict],
    executor: Callable[[], PromptResponse],
) -> PromptResponse:
    if not parent:
        return executor()

    if generation_config and is_dataclass(generation_config):
        generation_config = asdict(generation_config)

    generation_id = (
        generation_config["id"]
        if generation_config and "id" in generation_config
        else str(uuid.uuid4())
    )
    resolved_messages = []

    gen_kwargs = {
        "id": generation_id,
        "model": prompt.model or "",
        "provider": prompt.provider or "",
        "messages": resolved_messages,
        "model_parameters": prompt.model_parameters or {},
        "maxim_prompt_id": prompt.prompt_id or "",
        "maxim_prompt_version_id": prompt.version_id or "",
    }
    if generation_config:
        gen_kwargs.update(generation_config)

    generation = parent.generation(gen_kwargs)

    try:
        result = executor()

        if hasattr(result, "resolved_messages") and result.resolved_messages:
            resolved_messages = result.resolved_messages
            # Handle legacy payload format where messages may be wrapped
            if (
                resolved_messages
                and len(resolved_messages) > 0
                and isinstance(resolved_messages[0], dict)
                and "payload" in resolved_messages[0]
            ):
                resolved_messages = [m.get("payload", m) for m in resolved_messages]

        for message in resolved_messages:
            generation.add_message(message)

        if options and "variables" in options and options["variables"]:
            generation.add_metadata(options["variables"])

        result_to_log = result
        if is_dataclass(result):
            result_to_log = asdict(result)
        elif hasattr(result, "to_dict"):
            result_to_log = result.to_dict()

        if isinstance(result_to_log, dict):
            if "created" not in result_to_log:
                result_to_log["created"] = int(time.time())
            if "object" not in result_to_log:
                result_to_log["object"] = "chat.completion"

        generation.result(result_to_log)

        return result
    except Exception as e:
        generation.error(
            {
                "message": str(e),
                "code": getattr(e, "code", "UNKNOWN_ERROR"),
            }
        )
        raise


class RunnablePrompt:
    maxim_api: MaximAPI
    prompt_id: str
    version_id: str
    version: int
    messages: List[Message]
    model_parameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: Optional[str] = None
    provider: Optional[str] = None
    deployment_id: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None
    parent: Optional[Union[Trace, Span]] = None
    generation_config: Optional[GenerationConfig] = None

    def __init__(self, prompt: Prompt, maxim_api: MaximAPI):
        self.prompt_id = prompt.prompt_id
        self.version_id = prompt.version_id
        self.version = prompt.version
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
        generation_config: Optional[GenerationConfig] = None,
    ) -> "RunnablePrompt":
        self.parent = parent
        self.generation_config = generation_config
        return self

    def run(
        self,
        input: str,
        image_urls: Optional[List[ImageURL]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[PromptResponse]:
        if self.maxim_api is None:
            logging.error("[MaximSDK] Invalid prompt. APIs are not initialized.")
            return None

        def executor():
            return self.maxim_api.run_prompt_version(
                self.version_id, input, image_urls, variables
            )

        current_prompt_state = Prompt(
            prompt_id=self.prompt_id,
            version_id=self.version_id,
            version=self.version,
            messages=self.messages,
            model_parameters=self.model_parameters,
            model=self.model,
            provider=self.provider,
            deployment_id=self.deployment_id,
            tags=self.tags,
        )

        options = {"imageUrls": image_urls, "variables": variables}

        return _execute_prompt_with_logging(
            self.parent,
            self.generation_config,
            current_prompt_state,
            options,
            executor,
        )
