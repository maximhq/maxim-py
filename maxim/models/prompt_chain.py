import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

from .prompt import Prompt


class AgentCost(TypedDict):
    """
    Type definition for agent execution costs.

    Attributes:
        input: Cost for input tokens
        output: Cost for output tokens
        total: Total execution cost
    """
    input: float
    output: float
    total: float


class AgentUsage(TypedDict):
    """
    Type definition for agent token usage statistics.

    Attributes:
        prompt_tokens: Number of tokens used in the prompt
        completion_tokens: Number of tokens used in the completion
        total_tokens: Total number of tokens used
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AgentResponseMeta(TypedDict):
    """
    Type definition for agent response metadata.

    Attributes:
        cost: Cost breakdown for the response
        usage: Token usage statistics
        bound_variable_responses: Optional dictionary of bound variable values
        retrieved_context: Optional retrieved context string
    """
    cost: AgentCost
    usage: AgentUsage
    bound_variable_responses: Optional[dict[str, Any]]
    retrieved_context: Optional[str]


@dataclass
class AgentResponse:
    """
    Represents a complete agent response with metadata.

    This class encapsulates both the actual response content and associated
    metadata including cost, usage statistics, and contextual information.

    Attributes:
        response: The actual response text from the agent
        meta: Metadata about the response including costs and usage
    """
    response: str
    meta: AgentResponseMeta

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AgentResponse":
        """
        Create an AgentResponse instance from a dictionary.

        Args:
            data: Dictionary containing response and meta fields

        Returns:
            AgentResponse: New instance created from the dictionary data
        """
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
    """
    Node containing a prompt in a prompt chain.

    This node type wraps a Prompt object and represents a step in the chain
    that executes a prompt.

    Attributes:
        prompt: The Prompt object to be executed
    """
    prompt: Prompt


@dataclass
class CodeBlockNode:
    """
    Node containing executable code in a prompt chain.

    This node type contains code that can be executed as part of the
    prompt chain workflow.

    Attributes:
        code: The code string to be executed
    """
    code: str


@dataclass
class ApiParams:
    """
    Parameters for API node configuration.

    Attributes:
        id: Unique identifier for the parameter
        key: Parameter key name
        value: Parameter value
    """
    id: str
    key: str
    value: str


@dataclass
class ApiNode:
    """
    Node containing API configuration in a prompt chain.

    This node type wraps API configuration and represents a step in the chain
    that makes an API call.

    Attributes:
        api: Dictionary containing API configuration
    """
    api: dict[str, Any]


@dataclass
class Node:
    """
    Generic node in a prompt chain.

    A Node represents a single step in a prompt chain and can contain
    different types of content (prompt, code, or API configuration).

    Attributes:
        order: Execution order of this node in the chain
        content: The actual content (PromptNode, CodeBlockNode, or ApiNode)
    """
    order: int
    content: Union[PromptNode, CodeBlockNode, ApiNode]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        """
        Create a Node instance from a dictionary.

        Args:
            data: Dictionary containing order and content type fields

        Returns:
            Node: New instance created from the dictionary data
        """
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
    """
    Complete prompt chain with versioning information.

    A PromptChain represents a complete workflow consisting of multiple
    nodes that are executed in sequence. Each chain has versioning
    information for tracking changes over time.

    Attributes:
        prompt_chain_id: Unique identifier for the prompt chain
        version: Version number of this chain
        version_id: Unique identifier for this specific version
        nodes: List of nodes that make up the chain
    """
    prompt_chain_id: str
    version: int
    version_id: str
    nodes: List[Node]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChain":
        """
        Create a PromptChain instance from a dictionary.

        Args:
            data: Dictionary containing prompt chain data

        Returns:
            PromptChain: New instance created from the dictionary data
        """
        return PromptChain(
            prompt_chain_id=data["promptChainId"],
            version=data["version"],
            version_id=data["versionId"],
            nodes=[Node.from_dict(node) for node in data["nodes"]],
        )


@dataclass
class PromptChainVersionConfig:
    """
    Configuration for a specific prompt chain version.

    Contains the actual configuration data (nodes) for a particular
    version of a prompt chain.

    Attributes:
        nodes: List of nodes in this version configuration
    """
    nodes: List[Node]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChainVersionConfig":
        """
        Create a PromptChainVersionConfig instance from a dictionary.

        Args:
            data: Dictionary containing version configuration data

        Returns:
            PromptChainVersionConfig: New instance created from the dictionary data
        """
        return PromptChainVersionConfig(
            nodes=[Node.from_dict(node) for node in data["nodes"]]
        )


@dataclass
class PromptChainVersion:
    """
    Specific version of a prompt chain with metadata.

    Represents a particular version of a prompt chain including
    its configuration, description, and timestamps.

    Attributes:
        id: Unique identifier for this version
        version: Version number
        promptChainId: ID of the parent prompt chain
        description: Optional description of this version
        config: Optional configuration for this version
        createdAt: Timestamp when this version was created
        updatedAt: Timestamp when this version was last updated
    """
    id: str
    version: int
    promptChainId: str
    description: Optional[str]
    config: Optional[PromptChainVersionConfig]
    createdAt: str
    updatedAt: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptChainVersion":
        """
        Create a PromptChainVersion instance from a dictionary.

        Args:
            data: Dictionary containing version data

        Returns:
            PromptChainVersion: New instance created from the dictionary data
        """
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
    """
    Individual rule for prompt chain deployment logic.

    Defines a single rule that can be used to determine which version
    of a prompt chain should be deployed based on various conditions.

    Attributes:
        field: The field to evaluate in the rule
        value: The value to compare against (can be various types including None)
        operator: The comparison operator to use
        valueSource: Optional source of the value
        exactMatch: Optional flag for exact matching
    """
    field: str
    value: Union[str, int, List[str], bool, None]  # adding None here
    operator: str
    valueSource: Optional[str] = None
    exactMatch: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a PromptChainRuleType instance from a dictionary.

        Args:
            obj: Dictionary containing rule data

        Returns:
            PromptChainRuleType: New instance created from the dictionary data
        """
        return PromptChainRuleType(
            field=obj["field"],
            value=obj["value"],
            operator=obj["operator"],
            valueSource=obj.get("valueSource", None),
            exactMatch=obj.get("exactMatch", None),
        )


@dataclass
class PromptChainRuleGroupType:
    """
    Group of rules with a combinator for prompt chain deployment logic.

    Allows grouping multiple rules together with logical operators
    (AND/OR) to create complex deployment conditions.

    Attributes:
        rules: List of rules or nested rule groups
        combinator: Logical operator to combine rules ("and" or "or")
    """
    rules: List[Union["PromptChainRuleType", "PromptChainRuleGroupType"]]
    combinator: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a PromptChainRuleGroupType instance from a dictionary.

        Args:
            obj: Dictionary containing rule group data

        Returns:
            PromptChainRuleGroupType: New instance created from the dictionary data
        """
        rules = []
        for rule in obj["rules"]:
            if "rules" in rule:
                rules.append(PromptChainRuleGroupType.from_dict(rule))
            else:
                rules.append(PromptChainRuleType.from_dict(rule))
        return PromptChainRuleGroupType(rules=rules, combinator=obj["combinator"])


@dataclass
class PromptChainDeploymentRules:
    """
    Deployment rules for a specific prompt chain version.

    Defines the conditions under which a particular version of a
    prompt chain should be deployed or used.

    Attributes:
        version: The version number these rules apply to
        query: Optional rule group that defines the deployment conditions
    """
    version: int
    query: Optional[PromptChainRuleGroupType] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a PromptChainDeploymentRules instance from a dictionary.

        Args:
            obj: Dictionary containing deployment rules data

        Returns:
            PromptChainDeploymentRules: New instance created from the dictionary data
        """
        query = obj.get("query", None)
        if query:
            query = PromptChainRuleGroupType.from_dict(query)
        return PromptChainDeploymentRules(version=obj["version"], query=query)


@dataclass
class VersionSpecificDeploymentConfig:
    """
    Configuration for deploying a specific version.

    Contains deployment configuration including rules, timestamps,
    and fallback information for a particular version.

    Attributes:
        id: Unique identifier for this deployment configuration
        timestamp: When this configuration was created
        rules: The deployment rules for this configuration
        isFallback: Whether this is a fallback configuration
    """
    id: str
    timestamp: datetime
    rules: PromptChainDeploymentRules
    isFallback: bool = False

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a VersionSpecificDeploymentConfig instance from a dictionary.

        Args:
            obj: Dictionary containing deployment configuration data

        Returns:
            VersionSpecificDeploymentConfig: New instance created from the dictionary data
        """
        rules = PromptChainDeploymentRules.from_dict(obj["rules"])
        return VersionSpecificDeploymentConfig(
            id=obj["id"],
            timestamp=obj["timestamp"],
            rules=rules,
            isFallback=obj.get("isFallback", False),
        )


@dataclass
class PromptChainVersionsAndRules:
    """
    Container for prompt chain versions and their deployment rules.

    Aggregates all versions of a prompt chain along with their associated
    deployment rules and folder organization.

    Attributes:
        folderId: ID of the folder containing this prompt chain
        rules: Dictionary mapping rule IDs to lists of deployment configurations
        versions: List of all available versions
        fallbackVersion: Optional fallback version to use when rules don't match
    """
    folderId: str
    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptChainVersion]
    fallbackVersion: Optional[PromptChainVersion]

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a PromptChainVersionsAndRules instance from a dictionary.

        Args:
            obj: Dictionary containing versions and rules data

        Returns:
            PromptChainVersionsAndRules: New instance created from the dictionary data
        """
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
    """
    Extension of PromptChainVersionsAndRules that includes the prompt chain ID.

    Provides the same functionality as PromptChainVersionsAndRules but also
    includes the prompt chain identifier for complete context.

    Attributes:
        promptChainId: Unique identifier of the prompt chain
        Inherits all attributes from PromptChainVersionsAndRules
    """
    promptChainId: str = ""

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a VersionAndRulesWithPromptChainId instance from a dictionary.

        Args:
            obj: Dictionary containing versions, rules, and prompt chain ID data

        Returns:
            VersionAndRulesWithPromptChainId: New instance created from the dictionary data
        """
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
    """
    Response wrapper for single prompt chain API calls.

    Encapsulates the response from API calls that return information
    about a single prompt chain, including error handling.

    Attributes:
        data: The prompt chain versions and rules data
        error: Optional error information if the API call failed
    """
    data: PromptChainVersionsAndRules
    error: Optional[dict[str, Any]]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MaximApiPromptChainResponse":
        """
        Create a MaximApiPromptChainResponse instance from a dictionary.

        Args:
            data: Dictionary containing API response data

        Returns:
            MaximApiPromptChainResponse: New instance created from the dictionary data
        """
        return MaximApiPromptChainResponse(
            data=PromptChainVersionsAndRules.from_dict(data["data"]),
            error=data.get("error"),
        )


@dataclass
class PromptChainWithId(PromptChainVersionsAndRules):
    """
    Prompt chain versions and rules with associated prompt chain ID.

    Similar to VersionAndRulesWithPromptChainId but used in different contexts.
    Contains all version and rule information along with the prompt chain identifier.

    Attributes:
        promptChainId: Unique identifier of the prompt chain
        Inherits all attributes from PromptChainVersionsAndRules
    """
    promptChainId: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        """
        Create a PromptChainWithId instance from a dictionary.

        Args:
            obj: Dictionary containing prompt chain data with ID

        Returns:
            PromptChainWithId: New instance created from the dictionary data
        """
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
    """
    Custom JSON encoder for VersionAndRulesWithPromptChainId objects.

    Provides serialization support for VersionAndRulesWithPromptChainId
    instances by converting them to dictionaries.
    """
    def default(self, o):
        """
        Convert VersionAndRulesWithPromptChainId objects to dictionaries.

        Args:
            o: Object to encode

        Returns:
            dict: Dictionary representation of the object, or calls parent default
        """
        if isinstance(o, VersionAndRulesWithPromptChainId):
            return asdict(o)
        return super().default(o)


@dataclass
class MaximApiPromptChainsResponse:
    """
    Response wrapper for multiple prompt chains API calls.

    Encapsulates the response from API calls that return information
    about multiple prompt chains, including error handling.

    Attributes:
        data: List of prompt chains with their versions and rules
        error: Optional error information if the API call failed
    """
    data: List[PromptChainWithId]
    error: Optional[dict[str, Any]]

    @staticmethod
    def from_dict(incoming_data: Dict[str, Any]) -> "MaximApiPromptChainsResponse":
        """
        Create a MaximApiPromptChainsResponse instance from a dictionary.

        Args:
            incoming_data: Dictionary containing API response data for multiple chains

        Returns:
            MaximApiPromptChainsResponse: New instance created from the dictionary data
        """
        return MaximApiPromptChainsResponse(
            data=[PromptChainWithId.from_dict(item) for item in incoming_data["data"]],
            error=incoming_data.get("error"),
        )
