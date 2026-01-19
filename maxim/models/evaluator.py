import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Callable, TypeVar


class EvaluatorType(str, Enum):
    AI = "AI"
    PROGRAMMATIC = "Programmatic"
    STATISTICAL = "Statistical"
    API = "API"
    HUMAN = "Human"
    LOCAL = "Local"


@dataclass
class Evaluator:
    """
    This class represents an evaluator.
    """

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
    """
    This class represents the return value of a local evaluator.
    """

    score: Union[int, bool, str]
    reasoning: Optional[str] = None

    def __init__(self, score: Union[int, bool, str], reasoning: Optional[str] = None):
        """
        This class represents the return value of a local evaluator.

        Args:
            score: The score of the evaluator.
            reasoning: The reasoning of the evaluator.
        """
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
    """
    This class represents the pass fail criteria on each entry.
    """

    score_should_be: OperatorType
    value: Union[bool, int, float, None]

    def __init__(
        self, score_should_be: OperatorType, value: Union[bool, int, float, None]
    ):
        """
        This class represents the pass fail criteria on each entry.

        Args:
            score_should_be: The score should be.
            value: The value of the pass fail criteria.
        """
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
    """
    This class represents the pass fail criteria for the overall testrun.
    """

    overall_should_be: OperatorType
    value: int
    for_result: Literal["average", "percentageOfPassedResults"]

    def __init__(
        self,
        overall_should_be: OperatorType,
        value: int,
        for_result: Literal["average", "percentageOfPassedResults"],
    ):
        """
        This class represents the pass fail criteria for the overall testrun.

        Args:
            overall_should_be: The overall should be.
            value: The value of the pass fail criteria.
            for_result: The for result.
        """
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
    """
    This class represents the pass fail criteria.
    """

    on_each_entry: PassFailCriteriaOnEachEntry
    for_testrun_overall: PassFailCriteriaForTestrunOverall

    def __init__(
        self,
        on_each_entry_pass_if: PassFailCriteriaOnEachEntry,
        for_testrun_overall_pass_if: PassFailCriteriaForTestrunOverall,
    ):
        """
        This class represents the pass fail criteria.

        Args:
            on_each_entry_pass_if: The pass fail criteria on each entry.
            for_testrun_overall_pass_if: The pass fail criteria for the overall testrun.
        """
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
    """
    This class represents the result parameter of a local evaluator.
    """

    output: str
    context_to_evaluate: Optional[Union[str, List[str]]]

    def __init__(
        self, output: str, context_to_evaluate: Optional[Union[str, List[str]]]
    ):
        """
        This class represents the result parameter of a local evaluator.

        Args:
            output: The output of the local evaluator.
            context_to_evaluate: The context to evaluate.
        """
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
    """
    This class represents the result of a local evaluation.
    """

    result: LocalEvaluatorReturn
    name: str
    pass_fail_criteria: PassFailCriteria

    def __init__(
        self,
        result: LocalEvaluatorReturn,
        name: str,
        pass_fail_criteria: PassFailCriteria,
    ):
        """
        This class represents the result of a local evaluation.

        Args:
            result: The result of the local evaluation.
            name: The name of the local evaluation.
            pass_fail_criteria: The pass fail criteria of the local evaluation.
        """
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
    """
    This class represents the result of a local evaluation with an id.
    """

    id: str

    def __init__(
        self,
        result: LocalEvaluatorReturn,
        name: str,
        pass_fail_criteria: PassFailCriteria,
        id: str,
    ):
        """
        This class represents the result of a local evaluation with an id.

        Args:
            result: The result of the local evaluation.
            name: The name of the local evaluation.
            pass_fail_criteria: The pass fail criteria of the local evaluation.
            id: The id of the local evaluation.
        """
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


# =============================================================================
# Variable Mapping Types
# =============================================================================


class VersionInfo(Dict[str, Any]):
    """
    TypedDict-like class representing version information passed to variable mapping functions.
    
    Contains:
        - id: Optional version ID (workflow, prompt, or prompt chain)
        - type: The type of version ("workflow", "prompt", or "promptChain")
    """
    pass


@dataclass
class VariableMappingInput:
    """
    The output object passed to variableMapping functions.
    This matches the YieldedOutput type from testRun but is defined here to avoid circular imports.
    
    Attributes:
        data: The main output string from the run
        retrieved_context_to_evaluate: Optional context retrieved during the run
        messages: Optional list of messages from the run
        meta: Optional metadata including usage and cost information
    """
    data: str
    retrieved_context_to_evaluate: Optional[Union[str, List[str]]] = None
    messages: Optional[List[Any]] = None
    meta: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, checking standard fields first, then extra."""
        if key == "data":
            return self.data
        elif key == "retrieved_context_to_evaluate" or key == "retrieval":
            return self.retrieved_context_to_evaluate
        elif key == "messages":
            return self.messages
        elif key == "meta":
            return self.meta
        elif key == "output":
            return self.data
        elif self.extra and key in self.extra:
            return self.extra[key]
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {"data": self.data}
        if self.retrieved_context_to_evaluate is not None:
            result["retrieved_context_to_evaluate"] = self.retrieved_context_to_evaluate
        if self.messages is not None:
            result["messages"] = self.messages
        if self.meta is not None:
            result["meta"] = self.meta
        if self.extra is not None:
            result.update(self.extra)
        return result

    @classmethod
    def from_yielded_output(cls, output: Any, input_value: Optional[str] = None, 
                            context_to_evaluate: Optional[Union[str, List[str]]] = None) -> "VariableMappingInput":
        """Create a VariableMappingInput from a YieldedOutput-like object."""
        data = getattr(output, "data", "") if output else ""
        retrieved_context = getattr(output, "retrieved_context_to_evaluate", None) or context_to_evaluate
        
        meta_obj = getattr(output, "meta", None)
        meta: Optional[Dict[str, Any]] = None
        if meta_obj:
            usage_obj = getattr(meta_obj, "usage", None)
            cost_obj = getattr(meta_obj, "cost", None)
            meta = {}
            if usage_obj:
                meta["usage"] = usage_obj.to_dict() if hasattr(usage_obj, "to_dict") else usage_obj
            if cost_obj:
                meta["cost"] = cost_obj.to_dict() if hasattr(cost_obj, "to_dict") else cost_obj
        
        extra: Dict[str, Any] = {}
        if input_value is not None:
            extra["input"] = input_value
        if context_to_evaluate is not None:
            extra["retrieval"] = context_to_evaluate
        
        return cls(
            data=data,
            retrieved_context_to_evaluate=retrieved_context,
            messages=None,
            meta=meta,
            extra=extra if extra else None,
        )



# Forward reference types to avoid circular imports
LocalDataType = Dict[str, Any]

VariableMappingFunction = Callable[
    [VariableMappingInput, LocalDataType, Optional[VersionInfo]],
    Optional[str]
]

# Type alias for variable mapping dictionary
VariableMapping = Dict[str, VariableMappingFunction]


@dataclass
class PlatformEvaluator:
    """
    A platform evaluator (identified by name) with an optional variable mapping.
    Use this when you need to transform the output for a platform evaluator.
    
    Example:
        ```python
        from maxim.evaluators import PlatformEvaluator
        
        maxim.create_test_run("Test", workspace_id).with_evaluators(
            "Bias",  # Simple platform evaluator
            PlatformEvaluator(
                name="Bias",
                variable_mapping={
                    "output": lambda run, dataset, version: run.data.upper()
                }
            ),  # Platform evaluator with mapping
        )
        ```
    
    Attributes:
        name: The name of the platform evaluator
        variable_mapping: Optional dictionary of variable mapping functions
    """
    name: str
    variable_mapping: Optional[VariableMapping] = None
