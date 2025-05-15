import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

from .prompt import Prompt


class AgentCost(TypedDict):
    input: float
    output: float
    total: float


class AgentUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AgentResponseMeta(TypedDict):
    cost: AgentCost
    usage: AgentUsage
    bound_variable_responses: Optional[dict[str, Any]]
    retrieved_context: Optional[str]


@dataclass
class AgentResponse:
    response: str
    meta: AgentResponseMeta

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentResponse":
        return AgentResponse(
            response=data["response"],
            meta=AgentResponseMeta(
                cost=AgentCost(**data["meta"]["cost"]),
                usage=AgentUsage(**data["meta"]["usage"]),
                bound_variable_responses=data["meta"].get("bound_variable_responses"),
                retrieved_context=data["meta"].get("retrieved_context"),
            ),
        )


@dataclass
class PromptNode:
    prompt: Prompt


@dataclass
class CodeBlockNode:
    code: str


@dataclass
class ApiParams:
    id: str
    key: str
    value: str


@dataclass
class ApiNode:
    api: dict[str, Any]


@dataclass
class Node:
    order: int
    content: Union[PromptNode, CodeBlockNode, ApiNode]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        content_type = next(iter(set(data.keys()) - {"order"}))
        content_data = data[content_type]
        if content_type == "prompt":
            content = PromptNode(prompt=Prompt.from_dict(content_data))
        elif content_type == "code":
            content = CodeBlockNode(code=content_data)
        else:  # api
            content = ApiNode(api=content_data)
        return Node(order=data["order"], content=content)


# Note: Any changes here should be done in RunnablePromptChain as well
@dataclass
class PromptChain:
    prompt_chain_id: str
    version: int
    version_id: str
    nodes: List[Node]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChain":
        return PromptChain(
            prompt_chain_id=data["promptChainId"],
            version=data["version"],
            version_id=data["versionId"],
            nodes=[Node.from_dict(node) for node in data["nodes"]],
        )


@dataclass
class PromptChainVersionConfig:
    nodes: List[Node]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChainVersionConfig":
        return PromptChainVersionConfig(
            nodes=[Node.from_dict(node) for node in data["nodes"]]
        )


@dataclass
class PromptChainVersion:
    id: str
    version: int
    promptChainId: str
    description: Optional[str]
    config: Optional[PromptChainVersionConfig]
    createdAt: str
    updatedAt: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChainVersion":
        return PromptChainVersion(
            id=data["id"],
            version=data["version"],
            promptChainId=data["promptChainId"],
            description=data.get("description"),
            config=(
                PromptChainVersionConfig.from_dict(data["config"])
                if data.get("config")
                else None
            ),
            createdAt=data["createdAt"],
            updatedAt=data["updatedAt"],
        )


@dataclass
class PromptChainRuleType:
    field: str
    value: Union[str, int, List[str], bool, None]  # adding None here
    operator: str
    valueSource: Optional[str] = None
    exactMatch: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        return PromptChainRuleType(
            field=obj["field"],
            value=obj["value"],
            operator=obj["operator"],
            valueSource=obj.get("valueSource", None),
            exactMatch=obj.get("exactMatch", None),
        )


@dataclass
class PromptChainRuleGroupType:
    rules: List[Union["PromptChainRuleType", "PromptChainRuleGroupType"]]
    combinator: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        rules = []
        for rule in obj["rules"]:
            if "rules" in rule:
                rules.append(PromptChainRuleGroupType.from_dict(rule))
            else:
                rules.append(PromptChainRuleType.from_dict(rule))
        return PromptChainRuleGroupType(rules=rules, combinator=obj["combinator"])


@dataclass
class PromptChainDeploymentRules:
    version: int
    query: Optional[PromptChainRuleGroupType] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        query = obj.get("query", None)
        if query:
            query = PromptChainRuleGroupType.from_dict(query)
        return PromptChainDeploymentRules(version=obj["version"], query=query)


@dataclass
class VersionSpecificDeploymentConfig:
    id: str
    timestamp: datetime
    rules: PromptChainDeploymentRules
    isFallback: bool = False

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        rules = PromptChainDeploymentRules.from_dict(obj["rules"])
        return VersionSpecificDeploymentConfig(
            id=obj["id"],
            timestamp=obj["timestamp"],
            rules=rules,
            isFallback=obj.get("isFallback", False),
        )


@dataclass
class PromptChainVersionsAndRules:
    folderId: str
    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptChainVersion]
    fallbackVersion: Optional[PromptChainVersion]

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        rules = obj["rules"]
        # Decoding each rule
        for key in rules:
            rules[key] = [
                VersionSpecificDeploymentConfig.from_dict(rule) for rule in rules[key]
            ]
        versions = [
            PromptChainVersion.from_dict(version) for version in obj["versions"]
        ]
        fallbackVersion = obj.get("fallbackVersion", None)
        if fallbackVersion:
            fallbackVersion = PromptChainVersion.from_dict(fallbackVersion)
        return PromptChainVersionsAndRules(
            rules=rules,
            versions=versions,
            folderId=obj.get("folderId", None),
            fallbackVersion=fallbackVersion,
        )


@dataclass
class VersionAndRulesWithPromptChainId(PromptChainVersionsAndRules):
    promptChainId: str = ""

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        existing_rules = obj["rules"]
        rules: Dict[str, List[VersionSpecificDeploymentConfig]] = {}
        # Decoding each rule
        for key in existing_rules:
            configs: List[VersionSpecificDeploymentConfig] = []
            for rule in existing_rules[key]:
                configs.append(VersionSpecificDeploymentConfig.from_dict(rule))
            rules[key] = configs
        versions: List[PromptChainVersion] = []
        for version_dict in obj["versions"]:
            versions.append(PromptChainVersion.from_dict(version_dict))
        fallback_version: Optional[PromptChainVersion] = None

        if (fallback_version_dict := obj.get("fallbackVersion", None)) is not None:
            fallback_version = PromptChainVersion.from_dict(fallback_version_dict)
        return VersionAndRulesWithPromptChainId(
            rules=rules,
            versions=versions,
            promptChainId=obj["promptChainId"],
            folderId=obj.get("folderId", None),
            fallbackVersion=fallback_version,
        )


@dataclass
class MaximApiPromptChainResponse:
    data: PromptChainVersionsAndRules
    error: Optional[dict[str, Any]]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MaximApiPromptChainResponse":
        return MaximApiPromptChainResponse(
            data=PromptChainVersionsAndRules.from_dict(data["data"]),
            error=data.get("error"),
        )


@dataclass
class PromptChainWithId(PromptChainVersionsAndRules):
    promptChainId: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        rules = obj["rules"]
        # Decoding each rule
        for key in rules:
            rules[key] = [
                VersionSpecificDeploymentConfig.from_dict(rule) for rule in rules[key]
            ]
        versions = [
            PromptChainVersion.from_dict(version) for version in obj["versions"]
        ]
        fallbackVersion = obj.get("fallbackVersion", None)
        if fallbackVersion:
            fallbackVersion = PromptChainVersion.from_dict(fallbackVersion)
        return PromptChainWithId(
            promptChainId=obj["promptChainId"],
            rules=rules,
            versions=versions,
            folderId=obj.get("folderId", None),
            fallbackVersion=fallbackVersion,
        )


class VersionAndRulesWithPromptChainIdEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, VersionAndRulesWithPromptChainId):
            return asdict(o)
        return super().default(o)


@dataclass
class MaximApiPromptChainsResponse:
    data: List[PromptChainWithId]
    error: Optional[dict[str, Any]]

    @staticmethod
    def from_dict(incoming_data: Dict[str, Any]) -> "MaximApiPromptChainsResponse":
        return MaximApiPromptChainsResponse(
            data=[PromptChainWithId.from_dict(item) for item in incoming_data["data"]],
            error=incoming_data.get("error"),
        )
