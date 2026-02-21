import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Union

from ..evaluators import BaseEvaluator
from ..models.dataset import Data, LocalData, T, Variable
from .evaluator import (
    EvaluatorType,
    LocalEvaluationResultWithId,
    PlatformEvaluator,
)



@dataclass
class YieldedOutputTokenUsage:
    """
    This class represents the token usage of a yielded output. Users can pass custom token usage to the `yieldsOutput` function.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: Optional[float] = None

    def __json__(self):
        ret_dict: Dict[str, Union[int, float]] = {}
        if self.prompt_tokens is not None:
            ret_dict["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            ret_dict["completion_tokens"] = self.completion_tokens
        if self.total_tokens is not None:
            ret_dict["total_tokens"] = self.total_tokens
        if self.latency is not None:
            ret_dict["latency"] = self.latency
        return ret_dict

    def to_dict(self) -> Dict[str, Union[int, float]]:
        ret_dict: Dict[str, Union[int, float]] = {}
        if self.prompt_tokens is not None:
            ret_dict["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            ret_dict["completion_tokens"] = self.completion_tokens
        if self.total_tokens is not None:
            ret_dict["total_tokens"] = self.total_tokens
        if self.latency is not None:
            ret_dict["latency"] = self.latency
        return ret_dict

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]):
        prompt_tokens = data.get("promptTokens") if "promptTokens" in data else data.get("prompt_tokens", 0)
        completion_tokens = data.get("completionTokens") if "completionTokens" in data else data.get("completion_tokens", 0)
        total_tokens = data.get("totalTokens") if "totalTokens" in data else data.get("total_tokens", 0)
        latency = data.get("latency")
        
        return cls(
            prompt_tokens=prompt_tokens if prompt_tokens is not None else 0,
            completion_tokens=completion_tokens if completion_tokens is not None else 0,
            total_tokens=total_tokens if total_tokens is not None else 0,
            latency=latency,
        )


@dataclass
class YieldedOutputCost:
    """
    This class represents the cost of a yielded output. Users can pass custom cost to the `yieldsOutput` function.
    """

    input_cost: float
    output_cost: float
    total_cost: float

    def __json__(self):
        return {
            "input": self.input_cost,
            "output": self.output_cost,
            "total": self.total_cost,
        }

    def to_dict(self):
        return {
            "input": self.input_cost,
            "output": self.output_cost,
            "total": self.total_cost,
        }

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]):
        return cls(
            input_cost=data["input"],
            output_cost=data["output"],
            total_cost=data["total"],
        )


@dataclass
class YieldedOutputMeta:
    """
    This class represents the meta of a yielded output. Users can pass custom meta to the `yieldsOutput` function.
    """

    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None

    def __json__(self):
        return {
            "entityType": self.entity_type,
            "entityId": self.entity_id,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost": self.cost.to_dict() if self.cost else None,
        }

    def to_dict(self):
        return {
            "entityType": self.entity_type,
            "entityId": self.entity_id,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost": self.cost.to_dict() if self.cost else None,
        }

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        usage = data["usage"]
        cost = data["cost"]
        return cls(
            usage=(YieldedOutputTokenUsage.dict_to_class(usage) if usage else None),
            cost=(YieldedOutputCost.dict_to_class(cost) if cost else None),
        )

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]):
        usage = data.get("usage")
        cost = data.get("cost")
        return cls(
            usage=(YieldedOutputTokenUsage.dict_to_class(usage) if usage else None),
            cost=(YieldedOutputCost.dict_to_class(cost) if cost else None),
        )


@dataclass
class SimulationConversationTurn:
    """A single turn in the simulation conversation history."""
    turn: int
    request: Dict[str, Any]
    response: Dict[str, Any]
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "turn": self.turn,
            "request": self.request,
            "response": self.response,
        }
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
        return result

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "SimulationConversationTurn":
        return cls(
            turn=data["turn"],
            request=data["request"],
            response=data["response"],
            reasoning=data.get("reasoning"),
        )


@dataclass
class SimulationMeta:
    """
    Metadata returned from simulation endpoints.
    Contains the full simulation conversation and trace data.
    """
    session_id: Optional[str] = None
    simulation_id: Optional[str] = None
    messages: Optional[List[Any]] = None
    trace: Optional[List[Any]] = None
    turns: Optional[List[Any]] = None
    test_run_entry_id: Optional[str] = None
    last_turn: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None
    usage: Optional["YieldedOutputTokenUsage"] = None
    cost: Optional["YieldedOutputCost"] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.session_id is not None:
            result["sessionId"] = self.session_id
        if self.simulation_id is not None:
            result["simulationId"] = self.simulation_id
        if self.messages is not None:
            result["messages"] = [
                m.to_dict() if hasattr(m, "to_dict") else m for m in self.messages
            ]
        if self.trace is not None:
            result["trace"] = self.trace
        if self.turns is not None:
            result["turns"] = self.turns
        if self.test_run_entry_id is not None:
            result["testRunEntryId"] = self.test_run_entry_id
        if self.last_turn is not None:
            result["lastTurn"] = self.last_turn
        if self.stop_reason is not None:
            result["stopReason"] = self.stop_reason
        if self.usage is not None:
            result["usage"] = {
                "promptTokens": self.usage.prompt_tokens,
                "completionTokens": self.usage.completion_tokens,
                "totalTokens": self.usage.total_tokens,
                **({"latency": self.usage.latency} if self.usage.latency is not None else {}),
            }
        if self.cost is not None:
            result["cost"] = self.cost.to_dict()
        return result

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "SimulationMeta":
        usage_data = data.get("usage")
        cost_data = data.get("cost")
        return cls(
            session_id=data.get("sessionId"),
            simulation_id=data.get("simulationId"),
            messages=data.get("messages", []),
            trace=data.get("trace"),
            turns=data.get("turns"),
            test_run_entry_id=data.get("testRunEntryId"),
            last_turn=data.get("lastTurn"),
            stop_reason=data.get("stopReason"),
            usage=YieldedOutputTokenUsage.dict_to_class(usage_data) if usage_data else None,
            cost=YieldedOutputCost.dict_to_class(cost_data) if cost_data else None,
        )


@dataclass
class YieldedOutput:
    """
    Yielded output represents the output of `yieldsOutput` function.
    """

    data: str
    retrieved_context_to_evaluate: Optional[Union[str, list[str]]] = None
    simulation_outputs: Optional[List[str]] = None
    messages: Optional[List[Any]] = None
    simulation_meta: Optional[SimulationMeta] = None
    simulation_response: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Any]] = None
    meta: Optional[YieldedOutputMeta] = None


@dataclass
class EvaluatorArgs:
    """
    This class represents the arguments of an evaluator.
    """

    output: str
    input: Optional[str] = None
    expectedOutput: Optional[str] = None
    contextToEvaluate: Optional[Union[str, List[str]]] = None


@dataclass
class HumanEvaluationConfig:
    emails: List[str]
    instructions: Optional[str] = None
    requester: Optional[str] = None

    def __json__(self):
        return {
            "emails": self.emails,
            "instructions": self.instructions,
            "requester": self.requester,
        }

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "emails": self.emails,
                "instructions": self.instructions,
                "requester": self.requester,
            }.items()
            if v is not None
        }


class RunType(str, Enum):
    SINGLE = "SINGLE"
    COMPARISON = "COMPARISON"

    def __json__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "RunType":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid RunType: {value}")


@dataclass
class EvaluatorConfig:
    """
    This class represents the config of an evaluator.
    """

    id: str
    name: str
    type: EvaluatorType
    builtin: bool
    reversed: Optional[bool] = False
    config: Optional[Any] = None

    def __json__(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "builtin": self.builtin,
            "reversed": self.reversed,
            "config": self.config,
        }

    def to_dict(self):
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


@dataclass
class TestRun:
    """
    This class represents a test run.
    """

    id: str
    workspace_id: str
    eval_config: Dict[str, Any]
    human_evaluation_config: Optional[HumanEvaluationConfig] = None
    parent_test_run_id: Optional[str] = None
    environment_name: Optional[str] = None

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "id": self.id,
                "workspaceId": self.workspace_id,
                "evalConfig": self.eval_config,
                "humanEvaluationConfig": (
                    self.human_evaluation_config.to_dict()
                    if self.human_evaluation_config
                    else None
                ),
                "parentTestRunId": self.parent_test_run_id,
                "environmentName": self.environment_name,
            }.items()
            if v is not None
        }

    def __json__(self):
        return {
            "id": self.id,
            "workspaceId": self.workspace_id,
            "evalConfig": self.eval_config,
            "humanEvaluationConfig": (
                self.human_evaluation_config.__json__()
                if self.human_evaluation_config
                else None
            ),
            "parentTestRunId": self.parent_test_run_id,
            "environmentName": self.environment_name,
        }

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            workspace_id=data["workspaceId"],
            eval_config=data["evalConfig"],
            human_evaluation_config=(
                HumanEvaluationConfig(**data["humanEvaluationConfig"])
                if data.get("humanEvaluationConfig")
                else None
            ),
            parent_test_run_id=data.get("parentTestRunId"),
            environment_name=data.get("environmentName"),
        )

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRun":
        return cls(
            id=data["id"],
            workspace_id=data["workspaceId"],
            eval_config=data["evalConfig"],
            human_evaluation_config=(
                HumanEvaluationConfig(data["humanEvaluationConfig"])
                if data.get("humanEvaluationConfig")
                else None
            ),
            parent_test_run_id=data.get("parentTestRunId"),
            environment_name=data.get("environmentName"),
        )


@dataclass
class TestRunEntry:
    """
    This class represents an entry of a test run.
    """

    variables: Dict[str, Variable]
    output: Optional[str] = None
    input: Optional[str] = None
    expected_output: Optional[str] = None
    context_to_evaluate: Optional[Union[str, List[str]]] = None
    scenario: Optional[str] = None
    expected_steps: Optional[str] = None
    simulation_meta: Optional[SimulationMeta] = None
    local_evaluation_results: Optional[List[LocalEvaluationResultWithId]] = None
    sdk_variables: Optional[Dict[str, Dict[str, str]]] = None

    def to_dict(self):
        result = {}
        if self.output is not None:
            result["output"] = self.output
        if self.input is not None:
            result["input"] = self.input
        if self.expected_output is not None:
            result["expectedOutput"] = self.expected_output
        if self.context_to_evaluate is not None:
            result["contextToEvaluate"] = self.context_to_evaluate
        if self.scenario is not None:
            result["scenario"] = self.scenario
        if self.expected_steps is not None:
            result["expectedSteps"] = self.expected_steps
        if self.simulation_meta is not None:
            result["simulationMeta"] = self.simulation_meta.to_dict()
        if self.local_evaluation_results is not None:
            result["localEvaluationResults"] = (
                [
                    local_evaluation_result.to_dict()
                    for local_evaluation_result in self.local_evaluation_results
                ]
                if self.local_evaluation_results
                else None
            )
        if self.variables is not None:
            result["dataEntry"] = {
                key: variable.to_json() for key, variable in self.variables.items()
            }
        if self.sdk_variables is not None and len(self.sdk_variables) > 0:
            sdk_vars_formatted = {}
            for evaluator_id, var_mapping in self.sdk_variables.items():
                sdk_vars_formatted[evaluator_id] = {
                    "type": "json",
                    "payload": json.dumps(var_mapping),
                }
            if "meta" not in result:
                result["meta"] = {}
            result["meta"]["sdkVariables"] = sdk_vars_formatted

        return {key: value for key, value in result.items() if value is not None}

    def __json__(self):
        base_result = {
            key: value
            for key, value in {
                "output": self.output,
                "input": self.input,
                "expectedOutput": self.expected_output,
                "contextToEvaluate": self.context_to_evaluate,
                "scenario": self.scenario,
                "expectedSteps": self.expected_steps,
                "simulationMeta": self.simulation_meta.to_dict() if self.simulation_meta else None,
                "localEvaluationResults": (
                    [
                        local_evaluation_result.to_dict()
                        for local_evaluation_result in self.local_evaluation_results
                    ]
                    if self.local_evaluation_results
                    else None
                ),
                "dataEntry": (
                    {
                        key: variable.to_json()
                        for key, variable in self.variables.items()
                    }
                    if self.variables
                    else None
                ),
            }.items()
            if value is not None
        }
        
        if self.sdk_variables is not None and len(self.sdk_variables) > 0:
            sdk_vars_formatted = {}
            for evaluator_id, var_mapping in self.sdk_variables.items():
                sdk_vars_formatted[evaluator_id] = {
                    "type": "json",
                    "payload": json.dumps(var_mapping),
                }
            if "meta" not in base_result:
                base_result["meta"] = {}
            base_result["meta"]["sdkVariables"] = sdk_vars_formatted
        
        return base_result

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunEntry":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunEntry":
        local_evaluation_results = data.get("localEvaluationResults")
        data_entry = data.get("dataEntry", {})
        variables = {}
        for key, value in data_entry.items():
            if value is not None:
                variables[key] = Variable.from_json(value)

        # Parse sdk_variables from meta.sdkVariables if present
        sdk_variables = None
        meta = data.get("meta")
        if meta and "sdkVariables" in meta:
            sdk_variables = {}
            for evaluator_id, var_data in meta["sdkVariables"].items():
                if isinstance(var_data, dict) and "payload" in var_data:
                    try:
                        sdk_variables[evaluator_id] = json.loads(var_data["payload"])
                    except (json.JSONDecodeError, TypeError):
                        sdk_variables[evaluator_id] = var_data.get("payload", {})
                else:
                    sdk_variables[evaluator_id] = var_data

        return cls(
            output=data.get("output", None),
            input=data.get("input", None),
            expected_output=data.get("expectedOutput", None),
            context_to_evaluate=data.get("contextToEvaluate", None),
            scenario=data.get("scenario", None),
            expected_steps=data.get("expectedSteps", None),
            simulation_meta=(
                SimulationMeta.dict_to_class(data["simulationMeta"])
                if data.get("simulationMeta")
                else None
            ),
            variables=variables,
            local_evaluation_results=(
                [
                    LocalEvaluationResultWithId.dict_to_class(local_evaluation_result)
                    for local_evaluation_result in local_evaluation_results
                ]
                if local_evaluation_results
                else None
            ),
            sdk_variables=sdk_variables,
        )




@dataclass
class TestRunWithDatasetEntry(TestRun):
    """
    This class represents a test run with a dataset entry.
    """

    def __init__(self, test_run: TestRun, dataset_entry_id: str, dataset_id: str):
        super().__init__(
            id=test_run.id,
            workspace_id=test_run.workspace_id,
            eval_config=test_run.eval_config,
            human_evaluation_config=test_run.human_evaluation_config,
            parent_test_run_id=test_run.parent_test_run_id,
            environment_name=test_run.environment_name,
        )
        self.dataset_entry_id = dataset_entry_id
        self.dataset_id = dataset_id

    def __json__(self):
        base_json = super().__json__()
        base_json.update(
            {
                "datasetEntryId": self.dataset_entry_id,
                "datasetId": self.dataset_id,
            }
        )
        return base_json

    def to_dict(self):
        return {
            k: v
            for k, v in {
                "datasetEntryId": self.dataset_entry_id,
                "datasetId": self.dataset_id,
                "id": self.id,
                "workspaceId": self.workspace_id,
                "evalConfig": self.eval_config,
                "humanEvaluationConfig": (
                    self.human_evaluation_config.__json__()
                    if self.human_evaluation_config
                    else None
                ),
                "parentTestRunId": self.parent_test_run_id,
                "environmentName": self.environment_name,
            }.items()
            if v is not None
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunWithDatasetEntry":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunWithDatasetEntry":
        base_dict = super().dict_to_class(data)
        test_run = TestRunWithDatasetEntry(
            base_dict, data["datasetEntryId"], data["datasetId"]
        )
        return test_run


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"
    STOPPED = "STOPPED"


@dataclass
class TestRunStatus:
    """
    This class represents the status of a test run.
    """

    total_entries: int
    running_entries: int
    queued_entries: int
    failed_entries: int
    completed_entries: int
    stopped_entries: int
    test_run_status: RunStatus

    def to_dict(self):
        return {
            "totalEntries": self.total_entries,
            "runningEntries": self.running_entries,
            "queuedEntries": self.queued_entries,
            "failedEntries": self.failed_entries,
            "completedEntries": self.completed_entries,
            "stoppedEntries": self.stopped_entries,
            "testRunStatus": self.test_run_status.value,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunStatus":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunStatus":
        return cls(
            total_entries=data["total"],
            running_entries=data["running"],
            queued_entries=data["queued"],
            failed_entries=data["failed"],
            completed_entries=data["completed"],
            stopped_entries=data["stopped"],
            test_run_status=RunStatus(data["testRunStatus"]),
        )


@dataclass
class EvaluatorMeanScore:
    """
    This class represents the mean score of an evaluator. This helps users to specify the score of an custom evaluator.
    """

    score: Union[float, bool, str]
    out_of: Optional[float] = None
    is_pass: Optional[bool] = None

    def __json__(self):
        return {"score": self.score, "outOf": self.out_of, "pass": self.is_pass}

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluatorMeanScore":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "EvaluatorMeanScore":
        return cls(
            score=data["score"], out_of=data.get("outOf"), is_pass=data.get("pass")
        )


@dataclass
class TestRunTokenUsage:
    """
    This class represents the token usage of a test run.
    """

    total: int
    input: int
    completion: int

    def __json__(self):
        return {
            "total": self.total,
            "input": self.input,
            "completion": self.completion,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunTokenUsage":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunTokenUsage":
        return cls(
            total=data["total"],
            input=data["input"],
            completion=data["completion"],
        )


@dataclass
class TestRunCost:
    """
    This class represents the cost of a test run.
    """

    total: float
    input: float
    completion: float

    def __json__(self):
        return {
            "total": self.total,
            "input": self.input,
            "completion": self.completion,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunCost":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunCost":
        return cls(
            total=data["total"],
            input=data["input"],
            completion=data["completion"],
        )


@dataclass
class TestRunLatency:
    """
    This class represents the latency of a test run.
    """

    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    mean: float
    standard_deviation: float
    total: float

    def __json__(self):
        return {
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "standardDeviation": self.standard_deviation,
            "total": self.total,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunLatency":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunLatency":
        return cls(
            min=data["min"],
            max=data["max"],
            p50=data["p50"],
            p90=data["p90"],
            p95=data["p95"],
            p99=data["p99"],
            mean=data["mean"],
            standard_deviation=data["standardDeviation"],
            total=data["total"],
        )


@dataclass
class TestRunResultObj:
    """
    Object representing a result of a test run.
    """

    name: str
    individual_evaluator_mean_score: dict[str, EvaluatorMeanScore]
    usage: Optional[TestRunTokenUsage] = None
    cost: Optional[TestRunCost] = None
    latency: Optional[TestRunLatency] = None

    def __json__(self):
        return {
            "name": self.name,
            "individualEvaluatorMeanScore": {
                k: v.__json__() for k, v in self.individual_evaluator_mean_score.items()
            },
            "usage": self.usage.__json__() if self.usage else None,
            "cost": self.cost.__json__() if self.cost else None,
            "latency": self.latency.__json__() if self.latency else None,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunResultObj":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunResultObj":
        return cls(
            name=data["name"],
            individual_evaluator_mean_score={
                k: EvaluatorMeanScore.dict_to_class(v)
                for k, v in data["individualEvaluatorMeanScore"].items()
            },
            usage=(
                TestRunTokenUsage.dict_to_class(data["usage"])
                if data.get("usage")
                else None
            ),
            cost=(
                TestRunCost.dict_to_class(data["cost"]) if data.get("cost") else None
            ),
            latency=(
                TestRunLatency.dict_to_class(data["latency"])
                if data.get("latency")
                else None
            ),
        )


@dataclass
class TestRunResult:
    """
    This class represents the result of a test run.
    """

    link: str
    result: List[TestRunResultObj]

    def __json__(self):
        return {"link": self.link, "result": [r.__json__() for r in self.result]}

    @classmethod
    def from_json(cls, json_str: str) -> "TestRunResult":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "TestRunResult":
        return cls(
            link=data["link"],
            result=[TestRunResultObj.dict_to_class(r) for r in data["result"]],
        )


@dataclass
class RunResult:
    """
    This class represents the result of a comparison test run.
    """

    test_run_result: TestRunResult
    failed_entry_indices: List[int]


class TestRunLogger(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        """
        Log an informational message.

        Args:
            message (str): The message to be logged.
        """
        pass

    @abstractmethod
    def error(self, message: str, e: Optional[Exception] = None) -> None:
        """
        Log an error message.

        Args:
            message (str): The error message to be logged.
        """
        pass


class ConsoleLogger(TestRunLogger):
    def info(self, message: str) -> None:
        print(message)

    def error(self, message: str, e: Optional[Exception] = None) -> None:
        print(message, e)


@dataclass
class WorkflowConfig:
    id: str
    context_to_evaluate: Optional[str] = None

    def __json__(self):
        return {
            "id": self.id,
            "contextToEvaluate": self.context_to_evaluate,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowConfig":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        return cls(
            id=data["id"],
            context_to_evaluate=data.get("contextToEvaluate"),
        )


@dataclass
class PromptVersionConfig:
    id: str
    context_to_evaluate: Optional[str] = None

    def __json__(self):
        return {
            "id": self.id,
            "contextToEvaluate": self.context_to_evaluate,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PromptVersionConfig":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "PromptVersionConfig":
        return cls(
            id=data["id"],
            context_to_evaluate=data.get("contextToEvaluate"),
        )


@dataclass
class PromptChainVersionConfig:
    id: str
    context_to_evaluate: Optional[str] = None

    def __json__(self):
        return {
            "id": self.id,
            "contextToEvaluate": self.context_to_evaluate,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PromptChainVersionConfig":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "PromptChainVersionConfig":
        return cls(
            id=data["id"],
            context_to_evaluate=data.get("contextToEvaluate"),
        )


@dataclass
class SimulationConfig:
    """
    Configuration for simulation in test runs.
    """
    scenario: Optional[str] = None
    persona: Optional[Union[str, Dict[str, str]]] = None
    max_turns: Optional[int] = None
    tools: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    response_fields: Optional[List[str]] = None
    environment_id: Optional[str] = None
    stop_trigger: Optional[Dict[str, Any]] = None
    additional_instructions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.scenario is not None:
            result["scenario"] = self.scenario
        if self.persona is not None:
            result["persona"] = self.persona
        if self.max_turns is not None:
            result["maxTurns"] = self.max_turns
        if self.tools is not None:
            result["tools"] = self.tools
        if self.context is not None:
            result["context"] = self.context
        if self.response_fields is not None:
            result["responseFields"] = self.response_fields
        if self.environment_id is not None:
            result["environmentId"] = self.environment_id
        if self.stop_trigger is not None:
            result["stopTrigger"] = self.stop_trigger
        if self.additional_instructions is not None:
            result["additionalInstructions"] = self.additional_instructions
        return result

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "SimulationConfig":
        return cls(
            scenario=data.get("scenario"),
            persona=data.get("persona"),
            max_turns=data.get("maxTurns"),
            tools=data.get("tools"),
            context=data.get("context"),
            response_fields=data.get("responseFields"),
            environment_id=data.get("environmentId"),
            stop_trigger=data.get("stopTrigger"),
            additional_instructions=data.get("additionalInstructions"),
        )


@dataclass
class SimulationContext:
    """Context passed to the output function during simulation turns."""
    conversation_history: List[SimulationConversationTurn]
    current_user_input: Dict[str, Any]
    turn_number: int


@dataclass
class TestRunConfig(Generic[T]):
    """
    Configuration for a test run.

    Attributes:
        base_url (str): The base URL for the API.
        api_key (str): The API key for authentication.
        in_workspace_id (str): The ID of the workspace.
        workflow_id (Optional[str]): The ID of the workflow.
        prompt_version_id (Optional[str]): The ID of the prompt version.
        prompt_chain_version_id (Optional[str]): The ID of the prompt chain version.
        name (str): The name of the test run.
        tags (Optional[list[str]]): The tags for the test run.
        data_structure (Optional[T]): The structure of the test data.
        data (Optional[Union[str, DataValue[T], Callable[[int], Optional[DataValue[T]]]]]): The test data or a function to retrieve it.
        test_config_id (Optional[str]): The ID of the test configuration.
        platform_evaluators (List[PlatformEvaluatorType[T]]): List of platform evaluators to use.
    """

    base_url: str
    api_key: str
    in_workspace_id: str
    name: str
    evaluators: List[Union[str, BaseEvaluator, PlatformEvaluator]]
    workflow: Optional[WorkflowConfig] = None
    prompt_version: Optional[PromptVersionConfig] = None
    prompt_chain_version: Optional[PromptChainVersionConfig] = None
    tags: Optional[list[str]] = None
    data_structure: Optional[T] = None
    data: Optional[Data] = None
    test_config_id: Optional[str] = None
    logger: TestRunLogger = ConsoleLogger()
    human_evaluation_config: Optional[HumanEvaluationConfig] = None
    output_function: Optional[
        Callable[..., Union[YieldedOutput, Awaitable[YieldedOutput]]]
    ] = None
    concurrency: Optional[int] = None
    environment_name: Optional[str] = None
    simulation_config: Optional[SimulationConfig] = None


@dataclass
class ExecuteWorkflowForDataResponse:
    output: Optional[str]
    context_to_evaluate: Optional[str]
    latency: float

    def __json__(self):
        return {
            "output": self.output,
            "contextToEvaluate": self.context_to_evaluate,
            "latency": self.latency,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "ExecuteWorkflowForDataResponse":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecuteWorkflowForDataResponse":
        return cls(
            output=data["output"],
            context_to_evaluate=data.get("contextToEvaluate"),
            latency=data["latency"],
        )


@dataclass
class ExecutePromptForDataResponse:
    output: Optional[str]
    context_to_evaluate: Optional[str]
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None

    def __json__(self):
        return {
            "output": self.output,
            "contextToEvaluate": self.context_to_evaluate,
            "usage": self.usage.__json__() if self.usage else None,
            "cost": self.cost.__json__() if self.cost else None,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutePromptForDataResponse":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecutePromptForDataResponse":
        return cls(
            output=data["output"],
            context_to_evaluate=data.get("contextToEvaluate"),
            usage=(
                YieldedOutputTokenUsage.dict_to_class(data["usage"])
                if data.get("usage")
                else None
            ),
            cost=(
                YieldedOutputCost.dict_to_class(data["cost"])
                if data.get("cost")
                else None
            ),
        )


@dataclass
class ExecutePromptChainForDataResponse:
    output: Optional[str]
    context_to_evaluate: Optional[str]
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None

    def __json__(self):
        return {
            "output": self.output,
            "contextToEvaluate": self.context_to_evaluate,
            "usage": self.usage.__json__() if self.usage else None,
            "cost": self.cost.__json__() if self.cost else None,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutePromptChainForDataResponse":
        data = json.loads(json_str)
        return cls.dict_to_class(data)

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecutePromptChainForDataResponse":
        return cls(
            output=data["output"],
            context_to_evaluate=data.get("contextToEvaluate"),
            usage=(
                YieldedOutputTokenUsage.dict_to_class(data["usage"])
                if data.get("usage")
                else None
            ),
            cost=(
                YieldedOutputCost.dict_to_class(data["cost"])
                if data.get("cost")
                else None
            ),
        )


@dataclass
class ExecuteSimulationStartResponse:
    """Response from simulation POST (start); used to poll for result via GET."""

    workspace_id: str
    test_run_entry_id: str

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecuteSimulationStartResponse":
        return cls(
            workspace_id=data["workspaceId"],
            test_run_entry_id=data["testRunEntryId"],
        )


@dataclass
class ExecuteSimulationPromptForDataResponse:
    output: Optional[str] = None
    outputs: Optional[List[str]] = None
    messages: Optional[List[Any]] = None
    trace: Optional[List[Any]] = None
    context_to_evaluate: Optional[str] = None
    session_id: Optional[str] = None
    simulation_id: Optional[str] = None
    test_run_entry_id: Optional[str] = None
    latency: Optional[float] = None
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecuteSimulationPromptForDataResponse":
        outputs = data.get("outputs")
        output = data.get("output")
        if outputs is not None and output is None and outputs:
            output = outputs[-1]
        return cls(
            output=output,
            outputs=outputs,
            messages=data.get("messages"),
            trace=data.get("trace"),
            context_to_evaluate=data.get("contextToEvaluate"),
            session_id=data.get("sessionId"),
            simulation_id=data.get("simulationId"),
            test_run_entry_id=data.get("testRunEntryId"),
            latency=data.get("latency"),
            usage=(
                YieldedOutputTokenUsage.dict_to_class(data["usage"])
                if data.get("usage")
                else None
            ),
            cost=(
                YieldedOutputCost.dict_to_class(data["cost"])
                if data.get("cost")
                else None
            ),
        )


@dataclass
class ExecuteSimulationWorkflowForDataResponse:
    output: Optional[str] = None
    outputs: Optional[List[str]] = None
    messages: Optional[List[Any]] = None
    trace: Optional[List[Any]] = None
    turns: Optional[List[Any]] = None
    context_to_evaluate: Optional[str] = None
    session_id: Optional[str] = None
    simulation_id: Optional[str] = None
    test_run_entry_id: Optional[str] = None
    latency: Optional[float] = None
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "ExecuteSimulationWorkflowForDataResponse":
        outputs = data.get("outputs")
        output = data.get("output")
        if outputs is not None and output is None and outputs:
            output = outputs[-1]
        return cls(
            output=output,
            outputs=outputs,
            messages=data.get("messages"),
            trace=data.get("trace"),
            turns=data.get("turns"),
            context_to_evaluate=data.get("contextToEvaluate"),
            session_id=data.get("sessionId"),
            simulation_id=data.get("simulationId"),
            test_run_entry_id=data.get("testRunEntryId"),
            latency=data.get("latency"),
            usage=(
                YieldedOutputTokenUsage.dict_to_class(data["usage"])
                if data.get("usage")
                else None
            ),
            cost=(
                YieldedOutputCost.dict_to_class(data["cost"])
                if data.get("cost")
                else None
            ),
        )


@dataclass
class LocalExecutionResponse:
    """Response from the local-execution simulation endpoints."""
    test_run_entry_id: Optional[str]
    user_input: Optional[Dict[str, Any]]
    turn_number: int
    is_complete: bool
    stop_reason: Optional[str] = None
    usage: Optional[YieldedOutputTokenUsage] = None
    cost: Optional[YieldedOutputCost] = None
    session_id: Optional[str] = None
    simulation_id: Optional[str] = None

    @classmethod
    def dict_to_class(cls, data: Dict[str, Any]) -> "LocalExecutionResponse":
        usage_data = data.get("usage")
        cost_data = data.get("cost")
        user_input_raw = data.get("userInput")
        if isinstance(user_input_raw, str):
            user_input_raw = {"input": user_input_raw}
        return cls(
            test_run_entry_id=data.get("testRunEntryId"),
            user_input=user_input_raw,
            turn_number=data.get("turnNumber", 0),
            is_complete=data.get("isComplete", False),
            stop_reason=data.get("stopReason"),
            usage=YieldedOutputTokenUsage.dict_to_class(usage_data) if usage_data else None,
            cost=YieldedOutputCost.dict_to_class(cost_data) if cost_data else None,
            session_id=data.get("sessionId"),
            simulation_id=data.get("simulationId"),
        )
