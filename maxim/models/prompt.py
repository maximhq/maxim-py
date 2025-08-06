import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union, Literal


# Type definitions for multimodal content support
class ImageURL(TypedDict):
    """
    This class represents an image URL.
    """

    url: str
    detail: Optional[str]


class CompletionRequestTextContent(TypedDict):
    """
    Represents text content in a completion request.

    Attribute:
        type: Content type identifier, always "text"
        text: The text content
    """

    type: Literal["text"]  # "text"
    text: str


class CompletionRequestImageUrlContent(TypedDict):
    """
    Represents an image URL with optional detail level for vision-enabled models.

    Attribute:
        type: Content type identifier, always "image_url"
        image_url: Image URL configuration with url and optional detail
    """

    type: Literal["image_url"]  # "image_url"
    image_url: ImageURL


# Union type for all possible content types
CompletionRequestContent = Union[
    CompletionRequestTextContent, CompletionRequestImageUrlContent
]


@dataclass
class FunctionCall:
    """
    This class represents a function call.
    """

    name: str
    arguments: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        name = obj["name"]
        arguments = obj["arguments"]
        return FunctionCall(name=name, arguments=arguments)


@dataclass
class ToolCall:
    """
    This class represents a tool call.
    """

    id: str
    type: str
    function: FunctionCall

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        id = obj["id"]
        type = obj["type"]
        function = FunctionCall.from_dict(obj["function"])
        return ToolCall(id=id, type=type, function=function)


@dataclass
class Message:
    """
    This class represents a message of a LLM response choice.

    Supports both string content and multimodal content
    including text and images for vision-enabled models.
    """

    role: str
    content: Union[str, List[CompletionRequestContent]]
    tool_calls: Optional[List[ToolCall]] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        role = obj["role"]
        content = obj["content"]

        # Handle both string content (backward compatibility) and array content (multimodal)
        if isinstance(content, str):
            # String content - keep as is for backward compatibility
            pass
        elif isinstance(content, list):
            # Array content - validate and convert to proper CompletionRequestContent objects
            validated_content = []
            for item in content:
                if not isinstance(item, dict) or "type" not in item:
                    raise TypeError("Invalid content item: missing 'type' field")

                if item["type"] == "text":
                    if "text" not in item:
                        raise TypeError("Text content missing required 'text' field")
                    validated_content.append(
                        CompletionRequestTextContent(
                            type=item["type"], text=item["text"]
                        )
                    )
                elif item["type"] == "image_url":
                    if "image_url" not in item or "url" not in item["image_url"]:
                        raise TypeError("Image content missing required fields")
                    validated_content.append(
                        CompletionRequestImageUrlContent(
                            type=item["type"],
                            image_url=ImageURL(
                                url=item["image_url"]["url"],
                                detail=item["image_url"].get("detail", "auto"),
                            ),
                        )
                    )
                else:
                    raise TypeError(f"Unsupported content type: {item['type']}")
            content = validated_content
        else:
            raise TypeError(f"Content must be string or list, got: {type(content).__name__}")

        tool_calls = (
            [ToolCall.from_dict(t) for t in obj["toolCalls"]]
            if "toolCalls" in obj
            else None
        )
        return Message(role=role, content=content, tool_calls=tool_calls)


@dataclass
class Choice:
    """
    This class represents a choice of a LLM response.
    """

    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    """
    This class represents the usage of a LLM response.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float


@dataclass
class PromptResponse:
    """
    This class represents a response of a prompt.
    """

    id: str
    provider: str
    model: str
    choices: List[Choice]
    usage: Usage
    model_params: Dict[str, Union[str, int, bool, Dict, None]] = field(
        default_factory=dict
    )

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        id = obj["id"]
        provider = obj["provider"]
        model = obj["model"]
        choices = [
            Choice(
                index=c["index"],
                message=Message.from_dict(c["message"]),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in obj["choices"]
        ]
        usage_dict = obj.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_dict.get("prompt_tokens", 0),
            completion_tokens=usage_dict.get("completion_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            latency=usage_dict.get("latency", 0.0),
        )
        model_params = obj.get("modelParams", {})
        return PromptResponse(
            id=id,
            provider=provider,
            model=model,
            choices=choices,
            usage=usage,
            model_params=model_params,
        )


class ChatCompletionMessageImageContent(TypedDict):
    """
    This class represents an image content of a chat completion message.
    """

    type: str
    image_url: ImageURL


class ChatCompletionMessageTextContent(TypedDict):
    """
    This class represents a text content of a chat completion message.
    """

    type: str
    text: str


class ChatCompletionMessage(TypedDict):
    """
    This class represents a chat completion message.
    """

    role: str
    content: Union[
        str,
        List[
            Union[ChatCompletionMessageImageContent, ChatCompletionMessageTextContent]
        ],
    ]


class Function(TypedDict):
    """
    This class represents a function.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(TypedDict):
    """
    This class represents a tool.
    """

    type: str
    function: Function


# Note: Any changes here should be done in RunnablePrompt as well
@dataclass
class Prompt:
    """
    This class represents a prompt.
    """

    prompt_id: str
    version: int
    version_id: str
    messages: List[Message]
    model_parameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: Optional[str] = None
    provider: Optional[str] = None
    deployment_id: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Prompt":
        return Prompt(
            prompt_id=data["promptId"],
            version=data["version"],
            version_id=data["versionId"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            model_parameters=data["modelParameters"],
            model=data.get("model"),
            provider=data.get("provider"),
            deployment_id=data.get("deploymentId"),
            tags=data.get("tags"),
        )


@dataclass
class RuleType:
    """
    This class represents a rule type.
    """

    field: str
    value: Union[str, int, List[str], bool, None]  # adding None here
    operator: str
    valueSource: Optional[str] = None
    exactMatch: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Dict):
        return RuleType(
            field=obj["field"],
            value=obj["value"],
            operator=obj["operator"],
            valueSource=obj.get("valueSource", None),
            exactMatch=obj.get("exactMatch", None),
        )


@dataclass
class RuleGroupType:
    """
    This class represents a rule group type.
    """

    rules: List[Union["RuleType", "RuleGroupType"]]
    combinator: str

    @staticmethod
    def from_dict(obj: Dict):
        rules = []
        for rule in obj["rules"]:
            if "rules" in rule:
                rules.append(RuleGroupType.from_dict(rule))
            else:
                rules.append(RuleType(**rule))
        return RuleGroupType(rules=rules, combinator=obj["combinator"])


@dataclass
class PromptDeploymentRules:
    """
    This class represents the deployment rules of a prompt.
    """

    version: int
    query: Optional[RuleGroupType] = None

    @staticmethod
    def from_dict(obj: Dict):
        query = obj.get("query", None)
        if query is not None:
            query = RuleGroupType.from_dict(query)
        return PromptDeploymentRules(version=obj["version"], query=query)


@dataclass
class VersionSpecificDeploymentConfig:
    """
    This class represents the deployment rules of a prompt version.
    """

    id: str
    timestamp: datetime
    rules: PromptDeploymentRules
    isFallback: bool = False

    @staticmethod
    def from_dict(obj: Dict):
        rules = PromptDeploymentRules.from_dict(obj["rules"])
        return VersionSpecificDeploymentConfig(
            id=obj["id"],
            timestamp=obj["timestamp"],
            rules=rules,
            isFallback=obj.get("isFallback", False),
        )


@dataclass
class PromptVersionConfig:
    """
    This class represents the config of a prompt version.
    """

    messages: List[Message]
    modelParameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: str
    provider: str
    deployment_id: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    @staticmethod
    def from_dict(obj: Dict):
        messages = [Message.from_dict(message) for message in obj["messages"]]
        return PromptVersionConfig(
            messages=messages,
            modelParameters=obj["modelParameters"],
            model=obj["model"],
            provider=obj["provider"],
            deployment_id=obj.get("deploymentId"),
            tags=obj.get("tags", None),
        )


@dataclass
class PromptVersion:
    """
    This class represents a prompt version.
    """

    id: str
    version: int
    promptId: str
    createdAt: str
    updatedAt: str
    deletedAt: Optional[str] = None
    description: Optional[str] = None
    config: Optional[PromptVersionConfig] = None

    @staticmethod
    def from_dict(obj: Dict):
        config = obj.get("config", None)
        if config:
            config = PromptVersionConfig.from_dict(config)
        return PromptVersion(
            id=obj["id"],
            version=obj["version"],
            promptId=obj["promptId"],
            createdAt=obj["createdAt"],
            updatedAt=obj["updatedAt"],
            deletedAt=obj.get("deletedAt", None),
            description=obj.get("description", None),
            config=config,
        )


@dataclass
class VersionsAndRules:
    """
    This class represents the versions and rules of a prompt.
    """

    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptVersion]
    folderId: Optional[str] = None
    fallbackVersion: Optional[PromptVersion] = None

    @staticmethod
    def from_dict(obj: Dict):
        rules = obj["rules"]
        # Decoding each rule
        for key in rules:
            rules[key] = [
                VersionSpecificDeploymentConfig.from_dict(rule) for rule in rules[key]
            ]
        versions = [PromptVersion.from_dict(version) for version in obj["versions"]]
        fallbackVersion = obj.get("fallbackVersion", None)
        if fallbackVersion:
            fallbackVersion = PromptVersion.from_dict(fallbackVersion)
        return VersionsAndRules(
            rules=rules,
            versions=versions,
            folderId=obj.get("folderId", None),
            fallbackVersion=fallbackVersion,
        )

    def to_json(self):
        return asdict(self)


@dataclass
class VersionAndRulesWithPromptId(VersionsAndRules):
    """
    This class represents the versions and rules of a prompt with a prompt id.
    """

    promptId: str = ""

    @staticmethod
    def from_dict(obj: Dict):
        promptId = obj["promptId"]
        del obj["promptId"]
        versionAndRules = VersionsAndRules.from_dict(obj)
        return VersionAndRulesWithPromptId(
            rules=versionAndRules.rules,
            versions=versionAndRules.versions,
            promptId=promptId,
            folderId=versionAndRules.folderId,
            fallbackVersion=versionAndRules.fallbackVersion,
        )


class VersionAndRulesWithPromptIdEncoder(json.JSONEncoder):
    """
    This class represents a JSON encoder for VersionAndRulesWithPromptId.
    """

    def default(self, o):
        if isinstance(o, VersionAndRulesWithPromptId):
            return asdict(o)
        return super().default(o)


@dataclass
class Error:
    """
    This class represents an error from Prompt.
    """

    message: str


@dataclass
class PromptData:
    """
    This class represents the data of a prompt.
    """

    promptId: str
    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptVersion]
    folderId: Optional[str] = None
    fallbackVersion: Optional[PromptVersion] = None


@dataclass
class MaximApiPromptResponse:
    """
    This class represents the response of a prompt.
    """

    data: VersionsAndRules
    error: Optional[Error] = None


@dataclass
class MaximApiPromptsResponse:
    """
    This class represents the response of a prompts.
    """

    data: List[PromptData]
    error: Optional[Error] = None


@dataclass
class MaximAPIResponse:
    """
    This class represents the response of a Maxim API.
    """

    error: Optional[Error] = None
