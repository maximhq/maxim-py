import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union


@dataclass
class FunctionCall():
    name: str
    arguments: str

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        name = obj['name']
        arguments = obj['arguments']
        return FunctionCall(name=name, arguments=arguments)

@dataclass
class ToolCall():
    id:str
    type:str
    function:FunctionCall

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        id = obj['id']
        type = obj['type']
        function = FunctionCall.from_dict(obj['function'])
        return ToolCall(id=id, type=type, function=function)

@dataclass
class Message():
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        role = obj['role']
        content = obj['content']
        tool_calls = [ToolCall.from_dict(t) for t in obj['toolCalls']] if 'toolCalls' in obj else None
        return Message(role=role, content=content, tool_calls=tool_calls)

@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float

@dataclass
class PromptResponse:
    id: str
    provider:str
    model:str
    choices: List[Choice]
    usage: Usage
    model_params: Dict[str, Union[str, int, bool, Dict, None]] = field(default_factory=dict)

    @staticmethod
    def from_dict(obj: Dict[str, Any]):
        id = obj['id']
        provider = obj['provider']
        model = obj['model']
        choices = [Choice(index=c['index'], message=Message.from_dict(c['message']), finish_reason=c.get('finish_reason', "stop")) for c in obj['choices']]
        usage = Usage(prompt_tokens=obj['usage']['prompt_tokens'], completion_tokens=obj['usage']['completion_tokens'], total_tokens=obj['usage']['total_tokens'], latency=obj['usage'].get('latency', 0.0))
        model_params = obj.get('modelParams', {})
        return PromptResponse(id=id, provider=provider, model=model, choices=choices, usage=usage, model_params=model_params)


class ImageURL(TypedDict):
    url: str
    detail: Optional[str]


class ChatCompletionMessageImageContent(TypedDict):
    type: str
    image_url: ImageURL


class ChatCompletionMessageTextContent(TypedDict):
    type: str
    text: str

class ChatCompletionMessage(TypedDict):
    role: str
    content: Union[str,List[Union[ChatCompletionMessageImageContent,ChatCompletionMessageTextContent]]]


class Function(TypedDict):
    name:str
    description:str
    parameters:Dict[str,Any]


class Tool(TypedDict):
    type: str
    function: Function


class ImageUrls(TypedDict):
    url: str
    detail: str


# Note: Any changes here should be done in RunnablePrompt as well
@dataclass
class Prompt:
    prompt_id: str
    version: int
    version_id: str
    messages: List[Message]
    model_parameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: Optional[str] = None
    provider: Optional[str] = None
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Prompt":
        return Prompt(
            prompt_id=data['promptId'],
            version=data['version'],
            version_id=data['versionId'],
            messages=[Message.from_dict(m) for m in data['messages']],
            model_parameters=data['modelParameters'],
            model=data['model'],
            provider=data['provider'],
            tags=data['tags']
        )


@dataclass
class RuleType():
    field: str
    value: Union[str, int, List[str], bool, None]  # adding None here
    operator: str
    valueSource: Optional[str] = None
    exactMatch: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Dict):
        return RuleType(field=obj['field'], value=obj['value'], operator=obj['operator'], valueSource=obj.get('valueSource', None), exactMatch=obj.get('exactMatch', None))


@dataclass
class RuleGroupType():
    rules: List[Union['RuleType', 'RuleGroupType']]
    combinator: str

    @staticmethod
    def from_dict(obj: Dict):
        rules = []
        for rule in obj['rules']:
            if 'rules' in rule:
                rules.append(RuleGroupType.from_dict(rule))
            else:
                rules.append(RuleType(**rule))
        return RuleGroupType(rules=rules, combinator=obj['combinator'])


@dataclass
class PromptDeploymentRules():
    version: int
    query: Optional[RuleGroupType] = None

    @staticmethod
    def from_dict(obj: Dict):
        query = obj.get('query', None)
        if query is not None:
            query = RuleGroupType.from_dict(query)
        return PromptDeploymentRules(version=obj['version'], query=query)


@dataclass
class VersionSpecificDeploymentConfig():
    id: str
    timestamp: datetime
    rules: PromptDeploymentRules
    isFallback: bool = False

    @staticmethod
    def from_dict(obj: Dict):
        rules = PromptDeploymentRules.from_dict(obj['rules'])
        return VersionSpecificDeploymentConfig(id=obj['id'], timestamp=obj['timestamp'], rules=rules, isFallback=obj.get('isFallback', False))


@dataclass
class PromptVersionConfig():
    messages: List[Message]
    modelParameters: Dict[str, Union[str, int, bool, Dict, None]]
    model: str
    provider: str
    tags: Optional[Dict[str, Union[str, int, bool, None]]] = None

    @staticmethod
    def from_dict(obj: Dict):
        messages = [Message.from_dict(message) for message in obj['messages']]
        return PromptVersionConfig(
            messages=messages,
            modelParameters=obj['modelParameters'],
            model=obj['model'],
            provider=obj['provider'],
            tags=obj.get('tags', None)
        )


@dataclass
class PromptVersion():
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
        config = obj.get('config', None)
        if config:
            config = PromptVersionConfig.from_dict(config)
        return PromptVersion(id=obj['id'], version=obj['version'], promptId=obj['promptId'], createdAt=obj['createdAt'], updatedAt=obj['updatedAt'], deletedAt=obj.get('deletedAt', None), description=obj.get('description', None), config=config)


@dataclass
class VersionsAndRules():
    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptVersion]
    folderId: Optional[str] = None
    fallbackVersion: Optional[PromptVersion] = None

    @staticmethod
    def from_dict(obj: Dict):
        rules = obj['rules']
        # Decoding each rule
        for key in rules:
            rules[key] = [VersionSpecificDeploymentConfig.from_dict(
                rule) for rule in rules[key]]
        versions = [PromptVersion.from_dict(version)
                    for version in obj['versions']]
        fallbackVersion = obj.get('fallbackVersion', None)
        if fallbackVersion:
            fallbackVersion = PromptVersion.from_dict(fallbackVersion)
        return VersionsAndRules(rules=rules, versions=versions,  folderId=obj.get('folderId', None), fallbackVersion=fallbackVersion)

    def to_json(self):
        return asdict(self)


@ dataclass
class VersionAndRulesWithPromptId(VersionsAndRules):
    promptId: str = ""

    @staticmethod
    def from_dict(obj: Dict):
        promptId = obj['promptId']
        del obj['promptId']
        versionAndRules = VersionsAndRules.from_dict(obj)
        return VersionAndRulesWithPromptId(rules=versionAndRules.rules, versions=versionAndRules.versions, promptId=promptId, folderId=versionAndRules.folderId, fallbackVersion=versionAndRules.fallbackVersion)


class VersionAndRulesWithPromptIdEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, VersionAndRulesWithPromptId):
            return asdict(o)
        return super().default(o)


@ dataclass
class Error():
    message: str


@ dataclass
class PromptData():
    promptId: str
    rules: Dict[str, List[VersionSpecificDeploymentConfig]]
    versions: List[PromptVersion]
    folderId: Optional[str] = None
    fallbackVersion: Optional[PromptVersion] = None


@ dataclass
class MaximApiPromptResponse():
    data: VersionsAndRules
    error: Optional[Error] = None


@ dataclass
class MaximApiPromptsResponse():
    data: List[PromptData]
    error: Optional[Error] = None


@ dataclass
class MaximAPIResponse():
    error: Optional[Error] = None
