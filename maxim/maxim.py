import atexit
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union

from typing_extensions import deprecated

from .models.prompt_chain import PromptChainRuleGroupType

from .apis import MaximAPI
from .cache import MaximCache, MaximInMemoryCache
from .dataset import add_entries as _add_dataset_entries
from .filter_objects import (
    IncomingQuery,
    QueryObject,
    find_all_matches,
    find_best_match,
    parse_incoming_query,
)
from .logger import Logger, LoggerConfig, LoggerConfigDict
from .models import (
    DatasetEntry,
    ChatCompletionMessage,
    Folder,
    FolderEncoder,
    Node,
    Prompt,
    PromptChain,
    PromptChainVersion,
    PromptChainVersionsAndRules,
    PromptNode,
    PromptResponse,
    PromptVersion,
    QueryRule,
    RuleGroupType,
    RuleType,
    Tool,
    VersionAndRulesWithPromptChainIdEncoder,
    VersionAndRulesWithPromptId,
    VersionAndRulesWithPromptIdEncoder,
    VersionsAndRules,
)
from .runnable import RunnablePrompt, RunnablePromptChain
from .scribe import scribe
from .test_runs import TestRunBuilder
from .version import current_version
from .expiring_key_value_store import ExpiringKeyValueStore


class ConfigDict(TypedDict, total=False):
    """
    A class representing the configuration for the Maxim SDK.

    Attributes:
        api_key (Optional[str], optional): The API key for the Maxim instance. Defaults to None.
        base_url (Optional[str], optional): The base URL for the Maxim instance. Defaults to None.
        cache (Optional[MaximCache], optional): The cache to use for the Maxim instance. Defaults to None.
        debug (Optional[bool], optional): Whether to enable debug logging. Defaults to False.
        raise_exceptions (Optional[bool], optional): Whether to raise exceptions during logging operations. Defaults to False.
        prompt_management (Optional[bool], optional): Whether to enable prompt management. Defaults to False.
    """

    api_key: Optional[str]
    base_url: Optional[str]
    cache: Optional[MaximCache]
    debug: Optional[bool]
    raise_exceptions: Optional[bool]
    prompt_management: Optional[bool]


@deprecated(
    "This class will be removed in a future version. Use {} which is TypedDict."
)
@dataclass
class Config:
    """
    A class representing the configuration for the Maxim SDK.

    Attributes:
        api_key (str): The API key for the Maxim instance.
        base_url (Optional[str], optional): The base URL for the Maxim instance. Defaults to "https://app.getmaxim.ai".
        cache (Optional[MaximCache], optional): The cache to use for the Maxim instance. Defaults to None.
        debug (Optional[bool], optional): Whether to enable debug logging. Defaults to False.
        raise_exceptions (Optional[bool], optional): Whether to raise exceptions during logging operations. Defaults to False.
        prompt_management (Optional[bool], optional): Whether to enable prompt management. Defaults to False.
    """

    api_key: Optional[str] = os.environ.get("MAXIM_API_KEY")
    base_url: Optional[str] = None
    cache: Optional[MaximCache] = None
    debug: Optional[bool] = False
    raise_exceptions: Optional[bool] = False
    prompt_management: Optional[bool] = False


EntityType = {"PROMPT": "PROMPT", "FOLDER": "FOLDER", "PROMPT_CHAIN": "PROMPT_CHAIN"}


def get_config_dict(config: Union[Config, ConfigDict]) -> dict[str, Any]:
    """
    Converts a Config or ConfigDict to a dictionary with default values.

    Args:
        config (Union[Config, ConfigDict]): The configuration object to convert.

    Returns:
        dict[str, Any]: A dictionary containing the configuration parameters with defaults applied.
    """
    return (
        {
            "api_key": config.api_key,
            "base_url": config.base_url,
            "cache": config.cache or MaximInMemoryCache(),
            "debug": config.debug,
            "raise_exceptions": config.raise_exceptions,
            "prompt_management": config.prompt_management,
        }
        if isinstance(config, Config)
        else dict(config)
    )


class Maxim:
    _lock = threading.Lock()
    """
    A class representing the Maxim SDK.

    This class provides methods for interacting with the Maxim API.
    """

    def __init__(self, config: Union[Config, ConfigDict, None] = None):
        """
        Initializes a new instance of the Maxim class.

        Args:
            config (Config): The configuration for the Maxim instance.
        """
        # Thread-safe singleton pattern implementation
        with Maxim._lock:
            if hasattr(Maxim, "_instance"):
                raise RuntimeError(
                    "Maxim is already initialized. You can initialize it only once."
                )
            Maxim._instance = self
        self.has_cleaned_up = False
        atexit.register(self.cleanup)
        self.ascii_logo = (
            f"\033[32m[MaximSDK] Initializing Maxim AI(v{current_version})\033[0m"
        )
        # Print the ASCII logo when initializing
        print(self.ascii_logo)
        if config is None:
            config = ConfigDict()
        final_config = get_config_dict(config)
        if final_config.get("api_key", None) is None:
            # Checking in the env variable
            api_key = os.environ.get("MAXIM_API_KEY", None)
            if api_key is None:
                raise ValueError(
                    "API key is required. Either set it in the environment variable MAXIM_API_KEY or pass it as a parameter to the Config object."
                )
            final_config["api_key"] = api_key

        self.base_url = final_config.get("base_url", None) or "https://app.getmaxim.ai"
        self.api_key = final_config.get("api_key", None)
        self.is_running = True
        self.raise_exceptions = final_config.get("raise_exceptions", False)
        self.maxim_api = MaximAPI(self.base_url, self.api_key)
        self.__is_debug = final_config.get("debug", False)
        self.__loggers: Dict[str, Logger] = {}
        self.prompt_management = final_config.get("prompt_management", False)
        self.__cache = final_config.get("cache", MaximInMemoryCache())
        # Local TTL cache for promptVersionNumber single-condition fetches
        self.__prompt_version_by_number_cache: ExpiringKeyValueStore[Prompt] = ExpiringKeyValueStore()
        if self.prompt_management:
            self.__sync_thread = threading.Thread(target=self.__sync_timer)
            self.__sync_thread.daemon = True
            self.__sync_thread.start()
        else:
            self.__sync_thread = None

    def enable_prompt_management(self, cache: Optional[MaximCache] = None) -> "Maxim":
        """
        Enables prompt management functionality with optional cache configuration.

        Args:
            cache (Optional[MaximCache], optional): Custom cache implementation to use.
                Defaults to None (uses existing cache).

        Returns:
            Maxim: The current Maxim instance for method chaining.
        """
        self.prompt_management = True
        if cache is not None:
            self.__cache = cache
        if not self.__sync_thread or not self.__sync_thread.is_alive():
            self.__sync_thread = threading.Thread(target=self.__sync_timer)
            self.__sync_thread.daemon = True
            self.__sync_thread.start()
        return self

    def enable_exceptions(self, val: bool) -> "Maxim":
        """
        Enables or disables exception raising during logging operations.

        Args:
            val (bool): True to enable exception raising, False to disable.

        Returns:
            Maxim: The current Maxim instance for method chaining.
        """
        self.raise_exceptions = val
        return self

    def __sync_timer(self):
        """
        Runs the synchronization timer in a separate thread.

        Periodically syncs prompts and folders every 60 seconds while the instance is running
        and prompt management is enabled.
        """
        if not self.prompt_management:
            return
        while self.is_running:
            self.__sync_entities()
            time.sleep(60)

    def __sync_entities(self):
        """
        Synchronizes all entities (prompts, prompt chains, and folders) from the Maxim API.

        This method coordinates the synchronization of all cached entities by calling
        individual sync methods for prompts, prompt chains, and folders.
        """
        scribe().debug("[MaximSDK] Syncing prompts and folders")
        if not self.prompt_management:
            return
        if not self.is_running:
            return
        self.__sync_prompts()
        self.__sync_prompt_chains()
        self.__sync_folders()
        scribe().debug("[MaximSDK] Syncing completed")

    def __sync_prompts(self):
        """
        Synchronizes prompts from the Maxim API and stores them in the cache.

        Fetches all prompts from the API and caches them using the appropriate cache key.
        Handles exceptions based on the raise_exceptions configuration.
        """
        scribe().debug("[MaximSDK] Syncing prompts")
        try:
            prompts = self.maxim_api.get_prompts()
            scribe().debug(f"[MaximSDK] Found {len(prompts)} prompts")
            for prompt in prompts:
                try:
                    self.__cache.set(
                        self.__get_cache_key(EntityType["PROMPT"], prompt.promptId),
                        json.dumps(prompt, cls=VersionAndRulesWithPromptIdEncoder),
                    )
                except Exception as err:
                    scribe().error(f"[MaximSDK] Error while syncing prompts: {err}")
                    if self.raise_exceptions:
                        raise err
        except Exception as err:
            scribe().error(f"[MaximSDK] Error while syncing prompts: {err}")
            if self.raise_exceptions:
                raise err

    def __sync_prompt_chains(self):
        """
        Synchronizes prompt chains from the Maxim API and stores them in the cache.

        Fetches all prompt chains from the API and caches them using the appropriate cache key.
        Handles exceptions based on the raise_exceptions configuration.
        """
        scribe().debug("[MaximSDK] Syncing prompt Chains")
        try:
            prompt_chains = self.maxim_api.get_prompt_chains()
            scribe().debug(f"[MaximSDK] Found {len(prompt_chains)} prompt chains")
            for prompt_chain in prompt_chains:
                try:
                    self.__cache.set(
                        self.__get_cache_key(
                            EntityType["PROMPT_CHAIN"], prompt_chain.promptChainId
                        ),
                        json.dumps(
                            prompt_chain, cls=VersionAndRulesWithPromptChainIdEncoder
                        ),
                    )
                except Exception as err:
                    scribe().error(
                        f"[MaximSDK] Error while syncing prompt chains: {err}"
                    )
                    if self.raise_exceptions:
                        raise err
        except Exception as err:
            scribe().error(f"[MaximSDK] Error while syncing prompts chains: {err}")
            if self.raise_exceptions:
                raise err

    def __sync_folders(self):
        """
        Synchronizes folders from the Maxim API and stores them in the cache.

        Fetches all folders from the API and caches them using the appropriate cache key.
        Handles exceptions based on the raise_exceptions configuration.
        """
        scribe().debug("[MaximSDK] Syncing folders")
        try:
            folders = self.maxim_api.get_folders()
            scribe().debug(f"[MaximSDK] Found {len(folders)} folders")
            for folder in folders:
                try:
                    self.__cache.set(
                        self.__get_cache_key(EntityType["FOLDER"], folder.id),
                        json.dumps(folder, cls=FolderEncoder),
                    )
                except Exception as err:
                    scribe().error(f"[MaximSDK] Error while syncing folders: {err}")
                    if self.raise_exceptions:
                        raise err
        except Exception as err:
            scribe().error(f"[MaximSDK] Error while syncing folders: {err}")
            if self.raise_exceptions:
                raise err

    def __get_cache_key(self, entity: str, id: str) -> str:
        """
        Generates a cache key for the given entity type and ID.

        Args:
            entity (str): The entity type (e.g., "PROMPT", "FOLDER", "PROMPT_CHAIN").
            id (str): The entity ID.

        Returns:
            str: The formatted cache key.
        """
        if entity == "PROMPT":
            return f"prompt:{id}"
        if entity == "PROMPT_CHAIN":
            return f"prompt_chain:{id}"
        if entity == "FOLDER":
            return f"folder:{id}"
        raise ValueError(f"Invalid entity type: {entity}")

    def __get_prompt_from_cache(self, key: str) -> Optional[VersionsAndRules]:
        """
        Retrieves a prompt from the cache using the specified key.

        Args:
            key (str): The cache key for the prompt.

        Returns:
            Optional[VersionsAndRules]: The cached prompt data if found, None otherwise.
        """
        data = self.__cache.get(key)
        if not data:
            return None
        json_data = json.loads(data)
        return VersionAndRulesWithPromptId.from_dict(json_data)

    def __get_all_prompts_from_cache(self) -> Optional[List[VersionsAndRules]]:
        """
        Retrieves all prompts from the cache.

        Returns:
            Optional[List[VersionsAndRules]]: A list of all cached prompts if found, None otherwise.
        """
        keys = self.__cache.get_all_keys()
        if not keys:
            return None
        data = [self.__cache.get(key) for key in keys if key.startswith("prompt:")]
        prompt_list = []
        for d in data:
            if d is not None:
                json_data = json.loads(d)
                if "promptId" in json_data:
                    json_data.pop("promptId")
                prompt_list.append(VersionsAndRules.from_dict(json_data))
        return prompt_list

    def __get_prompt_chain_from_cache(
        self, key: str
    ) -> Optional[PromptChainVersionsAndRules]:
        """
        Retrieves a prompt chain from the cache using the specified key.

        Args:
            key (str): The cache key for the prompt chain.

        Returns:
            Optional[PromptChainVersionsAndRules]: The cached prompt chain data if found, None otherwise.
        """
        data = self.__cache.get(key)
        if not data:
            return None
        parsed_data = json.loads(data)
        return PromptChainVersionsAndRules.from_dict(parsed_data)

    def __get_all_prompt_chains_from_cache(
        self,
    ) -> Optional[List[PromptChainVersionsAndRules]]:
        """
        Retrieves all prompt chains from the cache.

        Returns:
            Optional[List[PromptChainVersionsAndRules]]: A list of all cached prompt chains if found, None otherwise.
        """
        keys = self.__cache.get_all_keys()
        if not keys:
            return None
        data = [
            self.__cache.get(key) for key in keys if key.startswith("prompt_chain:")
        ]
        prompt_chain_list = []
        for d in data:
            if d is not None:
                json_data = json.loads(d)
                if "promptChainId" in json_data:
                    json_data.pop("promptChainId")
                prompt_chain_list.append(
                    PromptChainVersionsAndRules.from_dict(json_data)
                )
        return prompt_chain_list

    def __get_folder_from_cache(self, key: str) -> Optional[Folder]:
        """
        Retrieves a folder from the cache using the specified key.

        Args:
            key (str): The cache key for the folder.

        Returns:
            Optional[Folder]: The cached folder data if found, None otherwise.
        """
        data = self.__cache.get(key)
        if not data:
            return None
        json_data = json.loads(data)
        return Folder(**json_data)

    def __get_all_folders_from_cache(self) -> Optional[List[Folder]]:
        """
        Retrieves all folders from the cache.

        Returns:
            Optional[List[Folder]]: A list of all cached folders if found, None otherwise.
        """
        keys = self.__cache.get_all_keys()
        if not keys:
            return None
        data = [self.__cache.get(key) for key in keys if key.startswith("folder:")]
        return [Folder(**json.loads(d)) for d in data if d is not None]

    def __format_prompt(self, prompt_version: PromptVersion) -> Prompt:
        """
        Formats a PromptVersion object into a Prompt object.

        Args:
            prompt_version (PromptVersion): The prompt version to format.

        Returns:
            Prompt: The formatted prompt object.
        """
        return Prompt(
            prompt_id=prompt_version.promptId,
            version_id=prompt_version.id,
            version=prompt_version.version,
            messages=prompt_version.config.messages if prompt_version.config else [],
            tags=prompt_version.config.tags if prompt_version.config else {},
            model_parameters=(
                prompt_version.config.modelParameters if prompt_version.config else {}
            ),
            model=prompt_version.config.model if prompt_version.config else None,
            provider=prompt_version.config.provider if prompt_version.config else None,
            deployment_id=(
                prompt_version.config.deployment_id if prompt_version.config else None
            ),
        )

    def __format_prompt_chain(
        self, prompt_chain_version: PromptChainVersion
    ) -> PromptChain:
        """
        Formats a PromptChainVersion object into a PromptChain object.

        Args:
            prompt_chain_version (PromptChainVersion): The prompt chain version to format.

        Returns:
            PromptChain: The formatted prompt chain object.
        """
        prompt_nodes: List[Node] = []
        if prompt_chain_version.config is not None:
            for node in prompt_chain_version.config.nodes:
                if isinstance(node.content, PromptNode):
                    prompt_nodes.append(node)
        return PromptChain(
            prompt_chain_id=prompt_chain_version.promptChainId,
            version_id=prompt_chain_version.id,
            version=prompt_chain_version.version,
            nodes=prompt_nodes,
        )

    def __get_prompt_version_for_rule(
        self,
        prompt_version_and_rules: VersionsAndRules,
        rule: Optional[QueryRule] = None,
    ) -> Optional[Prompt]:
        """
        Determines the appropriate prompt version based on the provided rule.

        This method evaluates query rules against available prompt versions to find the best match.
        It handles both rule-based matching and fallback scenarios.

        Args:
            prompt_version_and_rules (VersionsAndRules): The prompt versions and their associated rules.
            rule (Optional[QueryRule], optional): The rule to match against. If None, uses default logic.

        Returns:
            Optional[Prompt]: The matching prompt if found, None otherwise.
        """
        if rule:
            incoming_query = IncomingQuery(
                query=rule.query, operator=rule.operator, exactMatch=rule.exact_match
            )
            objects = []
            for version_id, version_rules in prompt_version_and_rules.rules.items():
                for version_rule in version_rules:
                    if not version_rule.rules.query:
                        continue
                    if rule.scopes:
                        for key in rule.scopes.keys():
                            if key != "folder":
                                raise ValueError("Invalid scope added")
                    version = next(
                        (
                            v
                            for v in prompt_version_and_rules.versions
                            if v.id == version_id
                        ),
                        None,
                    )
                    if not version:
                        continue
                    query = version_rule.rules.query
                    if version.config and version.config.tags:
                        parsed_incoming_query = parse_incoming_query(
                            incoming_query.query
                        )
                        tags = version.config.tags
                        for key, value in tags.items():
                            if value is None:
                                continue
                            if parsed_incoming_query is None:
                                continue
                            incomingQueryFields = [
                                q.field for q in parsed_incoming_query
                            ]
                            if key in incomingQueryFields:
                                query.rules.append(
                                    RuleType(field=key, operator="=", value=value)
                                )
                    objects.append(QueryObject(query=query, id=version_id))

            deployed_version_object = find_best_match(objects, incoming_query)
            if deployed_version_object:
                deployed_version = next(
                    (
                        v
                        for v in prompt_version_and_rules.versions
                        if v.id == deployed_version_object.id
                    ),
                    None,
                )
                if deployed_version:
                    return self.__format_prompt(deployed_version)

        else:
            if prompt_version_and_rules.rules:
                for version_id, version_rules in prompt_version_and_rules.rules.items():
                    does_query_exist = any(
                        ruleElm.rules.query is not None for ruleElm in version_rules
                    )
                    if does_query_exist:
                        deployed_version = next(
                            (
                                v
                                for v in prompt_version_and_rules.versions
                                if v.id == version_id
                            ),
                            None,
                        )
                        if deployed_version:
                            return self.__format_prompt(deployed_version)
            else:
                return self.__format_prompt(prompt_version_and_rules.versions[0])

        if prompt_version_and_rules.fallbackVersion:
            return self.__format_prompt(prompt_version_and_rules.fallbackVersion)
        return None

    def __get_prompt_chain_version_for_rule(
        self,
        prompt_chain_version_and_rules: PromptChainVersionsAndRules,
        rule: Optional[QueryRule] = None,
    ) -> Optional[PromptChain]:
        """
        Determines the appropriate prompt chain version based on the provided rule.

        This method evaluates query rules against available prompt chain versions to find the best match.
        It handles both rule-based matching and fallback scenarios.

        Args:
            prompt_chain_version_and_rules (PromptChainVersionsAndRules): The prompt chain versions and their associated rules.
            rule (Optional[QueryRule], optional): The rule to match against. If None, uses default logic.

        Returns:
            Optional[PromptChain]: The matching prompt chain if found, None otherwise.
        """
        if rule:
            incoming_query = IncomingQuery(
                query=rule.query, operator=rule.operator, exactMatch=rule.exact_match
            )
            objects = []
            for (
                version_id,
                version_rules,
            ) in prompt_chain_version_and_rules.rules.items():
                for version_rule in version_rules:
                    if not version_rule.rules.query:
                        continue
                    if rule.scopes:
                        for key in rule.scopes.keys():
                            if key != "folder":
                                raise ValueError("Invalid scope added")
                    version: Optional[PromptChainVersion] = None
                    for v in prompt_chain_version_and_rules.versions:
                        if v.id != version_id:
                            continue
                        version = v
                        break
                    if not version:
                        continue
                    if version_rule.rules.query is not None:
                        query = version_rule.rules.query
                        if isinstance(query, (RuleGroupType, PromptChainRuleGroupType)):
                            objects.append(QueryObject(query=query, id=version_id))

            deployed_version_object = find_best_match(objects, incoming_query)
            if deployed_version_object:
                deployed_version = next(
                    (
                        v
                        for v in prompt_chain_version_and_rules.versions
                        if v.id == deployed_version_object.id
                    ),
                    None,
                )
                if deployed_version:
                    return self.__format_prompt_chain(deployed_version)

        else:
            if prompt_chain_version_and_rules.rules:
                for (
                    version_id,
                    version_rules,
                ) in prompt_chain_version_and_rules.rules.items():
                    does_query_exist = any(
                        ruleElm.rules.query is not None for ruleElm in version_rules
                    )
                    if does_query_exist:
                        deployed_version = next(
                            (
                                v
                                for v in prompt_chain_version_and_rules.versions
                                if v.id == version_id
                            ),
                            None,
                        )
                        if deployed_version:
                            return self.__format_prompt_chain(deployed_version)
            else:
                return self.__format_prompt_chain(
                    prompt_chain_version_and_rules.versions[0]
                )

        if prompt_chain_version_and_rules.fallbackVersion:
            return self.__format_prompt_chain(
                prompt_chain_version_and_rules.fallbackVersion
            )
        return None

    def __get_folders_for_rule(
        self, folders: List[Folder], rule: QueryRule
    ) -> List[Folder]:
        """
        Filters folders based on the provided rule.

        This method evaluates folders against the given query rule to find matching folders
        based on their tags and the rule's query parameters.

        Args:
            folders (List[Folder]): The list of folders to filter.
            rule (QueryRule): The rule to match folders against.

        Returns:
            List[Folder]: A list of folders that match the given rule.
        """
        incoming_query = IncomingQuery(
            query=rule.query, operator=rule.operator, exactMatch=rule.exact_match
        )
        objects = []
        for folder in folders:
            query = RuleGroupType(rules=[], combinator="AND")
            if not folder.tags:
                continue
            parsed_incoming_query = parse_incoming_query(incoming_query.query)
            tags = folder.tags
            for key, value in tags.items():
                if key in [q.field for q in parsed_incoming_query]:
                    # if not isinstance(value,None):
                    query.rules.append(RuleType(field=key, operator="=", value=value))
            if not query.rules:
                continue
            objects.append(QueryObject(query=query, id=folder.id))

        folder_objects = find_all_matches(objects, incoming_query)
        ids = [fo.id for fo in folder_objects]
        return [f for f in folders if f.id in ids]

    def get_prompt(self, id: str, rule: QueryRule) -> Optional[RunnablePrompt]:
        """
        Retrieves a prompt based on the provided id and rule.

        Args:
            id (str): The id of the prompt.
            rule (QueryRule): The rule to match the prompt against.

        Returns:
            Optional[Prompt]: The prompt object if found, otherwise None.
        """
        if self.prompt_management is False:
            raise Exception(
                "prompt_management is disabled. You can enable it by initializing Maxim with Config(...prompt_management=True)."
            )
        # First, check if this is a single-condition promptVersionNumber query
        try:
            parsed_rules = parse_incoming_query(rule.query)
        except Exception:
            parsed_rules = []
        if len(parsed_rules) == 1 and parsed_rules[0].field == "promptVersionNumber" and parsed_rules[0].operator == "=":
            version_number = None
            try:
                version_number = int(parsed_rules[0].value)
            except Exception:
                version_number = None
            if version_number is not None:
                cache_key = f"pvnum:{id}:{version_number}"
                cached_prompt = self.__prompt_version_by_number_cache.get(cache_key)
                if cached_prompt is not None:
                    return RunnablePrompt(cached_prompt, self.maxim_api)
                # Fetch prompt only for the specific version
                version_and_rules_with_prompt_id = self.maxim_api.get_prompt(id, version_number)
                if len(version_and_rules_with_prompt_id.versions) == 0:
                    return None
                specific = next((v for v in version_and_rules_with_prompt_id.versions if v.version == version_number), None)
                if specific is None:
                    return None
                formatted = self.__format_prompt(specific)
                # Cache for 60 seconds
                self.__prompt_version_by_number_cache.set(cache_key, formatted, 60)
                return RunnablePrompt(formatted, self.maxim_api)

        key = self.__get_cache_key("PROMPT", id)
        version_and_rules_with_prompt_id = self.__get_prompt_from_cache(key)
        if version_and_rules_with_prompt_id is None:
            version_and_rules_with_prompt_id = self.maxim_api.get_prompt(id)
            if len(version_and_rules_with_prompt_id.versions) == 0:
                return None
            self.__cache.set(
                id,
                json.dumps(
                    version_and_rules_with_prompt_id,
                    cls=VersionAndRulesWithPromptIdEncoder,
                ),
            )
        if not version_and_rules_with_prompt_id:
            return None
        prompt = self.__get_prompt_version_for_rule(
            version_and_rules_with_prompt_id, rule
        )
        if not prompt:
            return None
        # wrapping this prompt with Runnable
        return RunnablePrompt(prompt, self.maxim_api)

    def get_prompts(self, rule: QueryRule) -> List[RunnablePrompt]:
        """
        Retrieves all prompts that match the given rule.

        Args:
            rule (QueryRule): The rule to match the prompts against.

        Returns:
            List[Prompt]: A list of prompts that match the given rule.
        """
        if self.prompt_management is False:
            raise Exception(
                "prompt_management is disabled. Please enable it in config."
            )
        version_and_rules = self.__get_all_prompts_from_cache()
        prompts: list[RunnablePrompt] = []
        if version_and_rules is None or len(version_and_rules) == 0:
            self.__sync_entities()
            version_and_rules = self.__get_all_prompts_from_cache()
        if not version_and_rules:
            return []
        for v in version_and_rules:
            if rule.scopes:
                if "folder" not in rule.scopes.keys():
                    return []
                else:
                    if rule.scopes["folder"] != v.folderId:
                        continue
            prompt = self.__get_prompt_version_for_rule(v, rule)
            if prompt is not None:
                prompts.append(RunnablePrompt(prompt, self.maxim_api))
        if len(prompts) == 0:
            return []
        return prompts

    def get_prompt_chain(
        self, id: str, rule: QueryRule
    ) -> Optional[RunnablePromptChain]:
        """
        Retrieves a prompt chain based on the provided id and rule.

        Args:
            id (str): The id of the prompt chain.
            rule (QueryRule): The rule to match the prompt chain against.

        Returns:
            Optional[PromptChain]: The prompt chain object if found, otherwise None.
        """
        if self.prompt_management is False:
            raise Exception(
                "prompt_management is disabled. Please enable it in config by setting Config(...,prompt_management=True)."
            )
        key = self.__get_cache_key("PROMPT_CHAIN", id)
        version_and_rules = self.__get_prompt_chain_from_cache(key)
        if version_and_rules is None:
            version_and_rules = self.maxim_api.getPromptChain(id)
            if len(version_and_rules.versions) == 0:
                return None
            self.__cache.set(
                id,
                json.dumps(
                    version_and_rules, cls=VersionAndRulesWithPromptChainIdEncoder
                ),
            )
        if not version_and_rules:
            return None
        prompt_chain = self.__get_prompt_chain_version_for_rule(version_and_rules, rule)
        if not prompt_chain:
            return None
        # The reason we have created a separate class
        # called agents is for two reasons
        # 1. We are not able to import MaximAPI in models fir circular deps
        # 2. We want to set the foundations of agents for future
        return RunnablePromptChain(prompt_chain, maxim_api=self.maxim_api)

    def get_folder_by_id(self, id: str) -> Optional[Folder]:
        """
        Retrieves a folder based on the provided id.

        Args:
            id (str): The id of the folder.

        Returns:
            Optional[Folder]: The folder object if found, otherwise None.
        """
        if self.prompt_management is False:
            raise Exception(
                "prompt_management is disabled. Please enable it in config."
            )
        key = self.__get_cache_key("FOLDER", id)
        folder = self.__get_folder_from_cache(key)
        if folder is None:
            try:
                folder = self.maxim_api.get_folder(id)
                if not folder:
                    return None
                self.__cache.set(key, json.dumps(folder, cls=FolderEncoder))
            except Exception:
                return None
        return folder

    def get_folders(self, rule: QueryRule) -> List[Folder]:
        """
        Retrieves all folders that match the given rule.

        Args:
            rule (QueryRule): The rule to match the folders against.

        Returns:
            List[Folder]: A list of folders that match the given rule.
        """
        if self.prompt_management is False:
            raise Exception(
                "prompt_management is disabled. Please enable it in config."
            )
        folders = self.__get_all_folders_from_cache()
        if folders is None or len(folders) == 0:
            self.__sync_entities()
            folders = self.__get_all_folders_from_cache()
        if not folders:
            return []
        return self.__get_folders_for_rule(folders, rule)

    def logger(
        self, config: Optional[Union[LoggerConfig, LoggerConfigDict]] = None
    ) -> Logger:
        """
        Creates a logger based on the provided configuration.

        Args:
            config (LoggerConfig): The configuration for the logger.

        Returns:
            Logger: The logger object.
        """
        final_config: LoggerConfigDict = {}
        if config is not None:
            if isinstance(config, LoggerConfig):
                final_config = {
                    "id": config.id,
                    "auto_flush": config.auto_flush,
                    "flush_interval": config.flush_interval,
                }
            else:
                final_config = config

        if final_config.get("id", None) is None:
            repo_id = os.environ.get("MAXIM_LOG_REPO_ID")
            if repo_id is None:
                raise ValueError(
                    "Log repo id is required. Either set environment variable MAXIM_LOG_REPO_ID or pass it as a parameter in LoggerConfig."
                )
            final_config["id"] = repo_id

        repo_id = final_config.get("id", None)
        if repo_id is None:
            raise ValueError("Log repository id is required")
        if repo_id in self.__loggers:
            return self.__loggers[repo_id]
        logger = Logger(
            config=final_config,
            api_key=self.api_key,
            base_url=self.base_url,
            is_debug=self.__is_debug,
            raise_exceptions=self.raise_exceptions,
        )
        self.__loggers[repo_id] = logger

        return logger

    def _check_if_repo_exists(self, logger: Logger):
        """
        Checks if the log repository exists.
        """

        def check():
            try:
                exists = self.maxim_api.does_log_repository_exist(logger.id)
                if not exists:
                    scribe().warning(
                        f"[MaximSDK] Log repository not found: {logger.id}. We will be dropping all logs."
                    )
                    if self.raise_exceptions:
                        raise ValueError(f"Log repository not found: {logger.id}")
                    return
                scribe().debug(f"[MaximSDK] Log repository found: {logger.id}")
            except Exception as e:
                scribe().error(
                    f"[MaximSDK] Failed to check repository existence: {str(e)}"
                )
                if self.raise_exceptions:
                    raise

        thread = threading.Thread(target=check)
        thread.daemon = True
        thread.start()

    def create_test_run(self, name: str, in_workspace_id: str) -> TestRunBuilder:
        """
        Creates a test run builder based on the provided name and workspace id.

        Args:
            name (str): The name of the test run.
            in_workspace_id (str): The workspace id to create the test run in.

        Returns:
            TestRunBuilder: The test run builder object.
        """
        return TestRunBuilder(
            name=name,
            workspace_id=in_workspace_id,
            api_key=self.api_key,
            base_url=self.base_url,
            evaluators=[],
        )

    def chat_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> Optional[PromptResponse]:
        """
        Performs a chat completion request using the specified model and messages.

        Args:
            model (str): The model name to use for completion. The expected format is "provider/model_name". Example "openai/gpt-3.5-turbo".
            messages (List[ChatCompletionMessage]): List of chat messages in the conversation
            tools (Optional[List[Tool]], optional): List of tools available to the model. Defaults to None.
            **kwargs: Additional model parameters to pass to the completion request

        Returns:
            Optional[PromptResponse]: The completion response if successful, None otherwise
        """
        return self.maxim_api.run_prompt(model, messages, tools, **kwargs)

    def add_dataset_entries(
        self,
        dataset_id: str,
        dataset_entries: list[Union[DatasetEntry, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Add entries to a dataset.

        Args:
            dataset_id (str): The ID of the dataset to add entries to
            dataset_entries (list[DatasetEntry | dict[str, Any]]): Entries to add.
                Note: For file-type variables, pass typed attachments
                (FileAttachment, FileDataAttachment, UrlAttachment), not dicts.
        Returns:
            dict[str, Any]: Response data from the API

        Raises:
            TypeError: If entry type is not DatasetEntry or dict[str, Any]
            Exception: If API call fails
        """
        return _add_dataset_entries(self.maxim_api, dataset_id, dataset_entries)

    def cleanup(self):
        """
        Cleans up the Maxim sync thread.
        """
        if self.has_cleaned_up:
            return
        # We sleep to allow all pending writes in the queue to come in and then we can flush
        time.sleep(2)
        scribe().debug("[MaximSDK] Cleaning up Maxim sync thread")
        self.is_running = False
        for logger in self.__loggers.values():
            logger.cleanup(is_sync=True)
        scribe().debug("[MaximSDK] Cleanup done")
