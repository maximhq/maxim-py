import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class EvaluatorType(str, Enum):
    AI = "AI"
    PROGRAMMATIC = "Programmatic"
    STATISTICAL = "Statistical"
    API = "API"
    HUMAN = "Human"
    LOCAL = "Local"


@dataclass
class Evaluator:
    id: str
    name: str
    type: EvaluatorType
    builtin: bool
    reversed: Optional[bool] = False
    config: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in {
                "id": self.id,
                "name": self.name,
                "type": self.type.value,
                "builtin": self.builtin,
                "reversed": self.reversed,
                "config": self.config,
            }.items()
            if v is not None
        }

    def __json__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "builtin": self.builtin,
            "reversed": self.reversed,
            "config": self.config,
        }

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "Evaluator":
        return cls(
            id=data["id"],
            name=data["name"],
            type=EvaluatorType(data["type"]),
            builtin=data["builtin"],
            reversed=data.get("reversed"),
            config=data.get("config"),
        )


OperatorType = Literal[">=", "<", "<=", ">", "=", "!="]


@dataclass
class LocalEvaluatorReturn:
    score: Union[int, bool, str]
    reasoning: Optional[str] = None

    def __init__(self, score: Union[int, bool, str], reasoning: Optional[str] = None):
        self.score = score
        self.reasoning = reasoning

    def __json__(self):
        return {
            key: value
            for key, value in {
                "score": self.score,
                "reasoning": self.reasoning,
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "score": self.score,
                "reasoning": self.reasoning,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "LocalEvaluatorReturn":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "LocalEvaluatorReturn":
        return cls(
            score=data["score"],
            reasoning=data.get("reasoning"),
        )


@dataclass
class PassFailCriteriaOnEachEntry:
    score_should_be: OperatorType
    value: Union[bool, int]

    def __init__(self, score_should_be: OperatorType, value: Union[bool, int]):
        self.score_should_be = score_should_be
        self.value = value

    def __json__(self):
        return {
            key: value
            for key, value in {
                "scoreShouldBe": self.score_should_be,
                "value": self.value,
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "scoreShouldBe": self.score_should_be,
                "value": self.value,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PassFailCriteriaOnEachEntry":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "PassFailCriteriaOnEachEntry":
        return cls(
            score_should_be=data["scoreShouldBe"],
            value=data["value"],
        )


@dataclass
class PassFailCriteriaForTestrunOverall:
    overall_should_be: OperatorType
    value: int
    for_result: Literal["average", "percentageOfPassedResults"]

    def __init__(
        self,
        overall_should_be: OperatorType,
        value: int,
        for_result: Literal["average", "percentageOfPassedResults"],
    ):
        if isinstance(value, bool):
            raise ValueError("overall_should_be is required to be an int")
        self.overall_should_be = overall_should_be
        self.value = value
        self.for_result = for_result

    def __json__(self):
        return {
            key: value
            for key, value in {
                "overallShouldBe": self.overall_should_be,
                "value": self.value,
                "for": self.for_result,
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "overallShouldBe": self.overall_should_be,
                "value": self.value,
                "for": self.for_result,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PassFailCriteriaForTestrunOverall":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "PassFailCriteriaForTestrunOverall":
        return cls(
            overall_should_be=data["overallShouldBe"],
            value=data["value"],
            for_result=data["for"],
        )


@dataclass
class PassFailCriteria:
    on_each_entry: PassFailCriteriaOnEachEntry
    for_testrun_overall: PassFailCriteriaForTestrunOverall

    def __init__(
        self,
        on_each_entry_pass_if: PassFailCriteriaOnEachEntry,
        for_testrun_overall_pass_if: PassFailCriteriaForTestrunOverall,
    ):
        self.on_each_entry = on_each_entry_pass_if
        self.for_testrun_overall = for_testrun_overall_pass_if

    def __json__(self):
        return {
            key: value
            for key, value in {
                "onEachEntry": self.on_each_entry.__json__(),
                "forTestrunOverall": self.for_testrun_overall.__json__(),
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "onEachEntry": self.on_each_entry.to_dict(),
                "forTestrunOverall": self.for_testrun_overall.to_dict(),
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PassFailCriteria":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "PassFailCriteria":
        return cls(
            on_each_entry_pass_if=PassFailCriteriaOnEachEntry.dict_to_class(
                data["onEachEntry"]
            ),
            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall.dict_to_class(
                data["forTestrunOverall"]
            ),
        )


@dataclass
class LocalEvaluatorResultParameter:
    output: str
    context_to_evaluate: Optional[Union[str, List[str]]]

    def __init__(
        self, output: str, context_to_evaluate: Optional[Union[str, List[str]]]
    ):
        self.output = output
        self.context_to_evaluate = context_to_evaluate

    def __json__(self):
        return {
            key: value
            for key, value in {
                "output": self.output,
                "contextToEvaluate": self.context_to_evaluate,
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "output": self.output,
                "contextToEvaluate": self.context_to_evaluate,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "LocalEvaluatorResultParameter":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "LocalEvaluatorResultParameter":
        return cls(
            output=data["output"],
            context_to_evaluate=data.get("contextToEvaluate"),
        )


@dataclass
class LocalEvaluationResult:
    result: LocalEvaluatorReturn
    name: str
    pass_fail_criteria: PassFailCriteria

    def __init__(
        self,
        result: LocalEvaluatorReturn,
        name: str,
        pass_fail_criteria: PassFailCriteria,
    ):
        self.result = result
        self.name = name
        self.pass_fail_criteria = pass_fail_criteria

    def __json__(self):
        return {
            key: value
            for key, value in {
                "result": self.result.__json__(),
                "name": self.name,
                "passFailCriteria": self.pass_fail_criteria.__json__(),
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "result": self.result.to_dict(),
                "name": self.name,
                "passFailCriteria": self.pass_fail_criteria.to_dict(),
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "LocalEvaluationResult":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "LocalEvaluationResult":
        return cls(
            result=LocalEvaluatorReturn.dict_to_class(data["result"]),
            name=data["name"],
            pass_fail_criteria=PassFailCriteria.dict_to_class(data["passFailCriteria"]),
        )


@dataclass
class LocalEvaluationResultWithId(LocalEvaluationResult):
    id: str

    def __init__(
        self,
        result: LocalEvaluatorReturn,
        name: str,
        pass_fail_criteria: PassFailCriteria,
        id: str,
    ):
        super().__init__(result, name, pass_fail_criteria)
        self.id = id

    def __json__(self):
        return {
            key: value
            for key, value in {
                "result": self.result.__json__(),
                "name": self.name,
                "passFailCriteria": self.pass_fail_criteria.__json__(),
                "id": self.id,
            }.items()
            if value is not None
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "result": self.result.to_dict(),
                "name": self.name,
                "passFailCriteria": self.pass_fail_criteria.to_dict(),
                "id": self.id,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "LocalEvaluationResultWithId":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "LocalEvaluationResultWithId":
        return cls(
            result=LocalEvaluatorReturn.dict_to_class(data["result"]),
            name=data["name"],
            pass_fail_criteria=PassFailCriteria.dict_to_class(data["passFailCriteria"]),
            id=data["id"],
        )
