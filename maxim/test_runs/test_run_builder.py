import asyncio
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Union, final
from uuid import uuid4

from maxim.logger import Logger
from maxim.logger.components import TraceConfigDict

from ..apis import MaximAPI
from ..dataset import sanitize_data_structure
from ..evaluators import BaseEvaluator
from ..models import (
    DatasetRow,
    Evaluator,
    EvaluatorType,
    HumanEvaluationConfig,
    PromptResponse,
    RunResult,
    RunStatus,
    RunType,
    T,
    TestRun,
    TestRunConfig,
    TestRunEntry,
    TestRunLogger,
    TestRunStatus,
    TestRunWithDatasetEntry,
    YieldedOutput,
    YieldedOutputMeta,
)
from ..models.dataset import Data, LocalData
from ..models.evaluator import (
    LocalEvaluationResultWithId,
    LocalEvaluatorResultParameter,
    PlatformEvaluator,
    VariableMappingInput,
    VersionInfo,
)
from ..models.test_run import (
    ExecuteSimulationPromptForDataResponse,
    ExecuteSimulationWorkflowForDataResponse,
    LocalExecutionResponse,
    PromptChainVersionConfig,
    PromptVersionConfig,
    SimulationConfig,
    SimulationContext,
    SimulationConversationTurn,
    SimulationMeta,
    WorkflowConfig,
    YieldedOutputCost,
    YieldedOutputTokenUsage,
)
from ..test_runs.run_utils import (
    get_input_expected_output_and_context_from_row,
    get_variables_from_row,
    process_awaitable,
    run_local_evaluations,
)
from ..test_runs.sanitization_utils import sanitize_data, sanitize_evaluators
from ..test_runs.utils import (
    EvaluatorNameToIdAndPassFailCriteria,
    get_evaluator_config_from_evaluator_name_and_pass_fail_criteria,
    get_local_evaluator_name_to_id_and_pass_fail_criteria_map,
)
from ..utils import Semaphore


def calculate_polling_interval(
    timeout_minutes: float, is_ai_evaluator_in_use: bool = False
) -> int:
    points = [
        (10, 5),
        (15, 5),
        (30, 10),
        (60, 15),
        (120, 30),
        (1440, 120),
    ]

    lower_point = points[0]
    upper_point = points[-1]
    for i in range(len(points) - 1):
        if points[i][0] <= timeout_minutes <= points[i + 1][0]:
            lower_point = points[i]
            upper_point = points[i + 1]
            break

    x1, y1 = lower_point
    x2, y2 = upper_point
    if x1 == x2:
        return y1

    t = (timeout_minutes - x1) / (x2 - x1)
    p = 2
    interpolated_value = y1 + (y2 - y1) * pow(t, p)

    return min(max(round(interpolated_value), 15 if is_ai_evaluator_in_use else 5), 120)


SIMULATION_POLL_TIMEOUT_MINUTES = 30


def get_all_keys_by_value(obj: Optional[dict[Any, Any]], value: Any) -> List[str]:
    if obj is None:
        return []
    return [key for key, val in obj.items() if val == value]


@dataclass
class ProcessedEntry:
    entry: TestRunEntry
    meta: Optional[YieldedOutputMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "meta": self.meta.to_dict() if self.meta else None,
        }


@final
class TestRunBuilder(Generic[T]):
    """
    Builder for test runs.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        name: str,
        workspace_id: str,
        evaluators: List[Union[str, BaseEvaluator, PlatformEvaluator]],
    ):
        """
        Constructor
        """
        self._config = TestRunConfig(
            base_url=base_url,
            api_key=api_key,
            name=name,
            evaluators=evaluators,
            in_workspace_id=workspace_id,
        )
        self._maxim_apis = MaximAPI(base_url, api_key)

    def _compute_sdk_variables_for_platform_evaluators(
        self,
        yielded_output: YieldedOutput,
        row: LocalData,
        input_value: Optional[str],
        context_to_evaluate: Optional[Union[str, List[str]]],
        evaluator_name_to_id_map: Dict[str, EvaluatorNameToIdAndPassFailCriteria],
        logger: TestRunLogger,
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Compute sdk_variables for platform evaluators that have variable_mapping.
        
        Args:
            yielded_output: The output from prompt/workflow execution
            row: The dataset row
            input_value: The input string
            context_to_evaluate: Context for evaluation
            evaluator_name_to_id_map: Map of evaluator names to their IDs
            logger: Logger instance
            
        Returns:
            Dict mapping evaluator IDs to their variable mappings, or None if empty
        """
        sdk_variables: Dict[str, Dict[str, str]] = {}
        
        # Build VariableMappingInput from yielded_output
        variable_mapping_input = VariableMappingInput.from_yielded_output(
            output=yielded_output,
            input_value=input_value,
            context_to_evaluate=context_to_evaluate,
        )
        
        # Determine version info based on execution type
        version_info: Optional[VersionInfo] = None
        if self._config.workflow is not None:
            version_info = VersionInfo({"id": self._config.workflow.id, "type": "workflow"})
        elif self._config.prompt_version is not None:
            version_info = VersionInfo({"id": self._config.prompt_version.id, "type": "prompt"})
        elif self._config.prompt_chain_version is not None:
            version_info = VersionInfo({"id": self._config.prompt_chain_version.id, "type": "promptChain"})
        
        # Process platform evaluators with variable_mapping
        for evaluator in self._config.evaluators:
            if isinstance(evaluator, PlatformEvaluator) and evaluator.variable_mapping:
                evaluator_info = evaluator_name_to_id_map.get(evaluator.name)
                if evaluator_info:
                    evaluator_id = evaluator_info.id
                    mapping_result: Dict[str, str] = {}
                    
                    for var_name, mapping_fn in evaluator.variable_mapping.items():
                        try:
                            result = mapping_fn(variable_mapping_input, row, version_info)
                            if result is not None:
                                mapping_result[var_name] = result
                        except Exception as e:
                            logger.error(f"Error in variable mapping for key '{var_name}': {e}")
                    
                    if mapping_result:
                        sdk_variables[evaluator_id] = mapping_result
        
        return sdk_variables if sdk_variables else None

    def _poll_simulation_result(
        self,
        fetch_status: Callable[[], Dict[str, Any]],
        parse_data: Callable[[Dict[str, Any]], Union[ExecuteSimulationPromptForDataResponse, ExecuteSimulationWorkflowForDataResponse]],
    ) -> Union[ExecuteSimulationPromptForDataResponse, ExecuteSimulationWorkflowForDataResponse]:
        """Poll simulation GET until runStatus is COMPLETE or FAILED. Returns parsed data when complete."""
        polling_interval = calculate_polling_interval(
            SIMULATION_POLL_TIMEOUT_MINUTES, is_ai_evaluator_in_use=False
        )
        max_iterations = math.ceil(
            (SIMULATION_POLL_TIMEOUT_MINUTES * 60) / polling_interval
        )
        for _ in range(max_iterations):
            resp = fetch_status()
            status = resp.get("status") or resp.get("runStatus") or resp.get("run_status")
            if status == RunStatus.FAILED.value:
                raise Exception("Simulation failed")
            if status == RunStatus.COMPLETE.value or status == RunStatus.STOPPED.value:
                data = resp.get("data")
                if data is None:
                    data = {
                        k: v
                        for k, v in resp.items()
                        if k not in ("status", "runStatus", "run_status")
                    }
                if not data:
                    raise Exception("Simulation completed but no data returned")
                return parse_data(data)
            time.sleep(polling_interval)
        raise Exception(
            f"Simulation did not complete within {SIMULATION_POLL_TIMEOUT_MINUTES} minutes"
        )

    @staticmethod
    def _get_nested_field_value(obj: Any, field_path: str) -> Any:
        """Get a value from a nested object by dot-separated path."""
        keys = field_path.split(".")
        value = obj
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif hasattr(value, key):
                value = getattr(value, key)
            else:
                return None
        return value

    def _run_simulation_with_local_output(
        self,
        row: LocalData,
        test_run_id: str,
        workspace_id: str,
        dataset_entry_id: Optional[str],
        input_val: Optional[str],
        scenario: Optional[str],
        expected_steps: Optional[str],
        context_to_evaluate: Optional[Union[str, List[str]]],
        output_function: Callable[..., Union[YieldedOutput, Awaitable[YieldedOutput]]],
        logger: TestRunLogger,
    ) -> YieldedOutput:
        """
        Run a turn-by-turn local simulation loop.
        Calls the backend /local-execution endpoint for each turn to get the AI user input,
        then calls the user's output_function with conversation context.
        """
        simulation_config = self._config.simulation_config
        if simulation_config is None:
            raise ValueError("simulation_config is required for local simulation")

        if simulation_config.stop_trigger is not None:
            if not isinstance(simulation_config.stop_trigger, dict):
                raise ValueError("stop_trigger must be a dict")
            field = simulation_config.stop_trigger.get("field")
            if not field or not isinstance(field, str):
                raise ValueError("stop_trigger.field must be a non-empty string")

        max_turns = simulation_config.max_turns or 10
        conversation_history: List[SimulationConversationTurn] = []
        simulation_outputs: List[str] = []
        test_run_entry_id: Optional[str] = None
        session_id: Optional[str] = None
        simulation_id: Optional[str] = None
        stop_reason: Optional[str] = None
        is_complete = False
        turn_number = 0

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_input_cost: float = 0
        total_output_cost: float = 0
        total_cost: float = 0

        # Resolve persona with priority: dataset column > simulation config
        dataset_persona: Optional[str] = None
        for key, value in row.items():
            if key.lower() == "persona" and value is not None:
                persona_str = str(value).strip()
                if persona_str:
                    dataset_persona = persona_str
                    break

        simconfig_persona: Optional[str] = None
        if simulation_config.persona is not None and dataset_persona is None:
            if isinstance(simulation_config.persona, str):
                simconfig_persona = simulation_config.persona
            elif isinstance(simulation_config.persona, dict) and simulation_config.persona.get("type") == "DATASET_COLUMN":
                col_name = simulation_config.persona.get("payload", "")
                if col_name:
                    ref_value = str(row.get(col_name, "")).strip() if row.get(col_name) is not None else None
                    simconfig_persona = ref_value if ref_value else None

        resolved_persona = dataset_persona if dataset_persona is not None else simconfig_persona

        resolved_simulation_config = SimulationConfig(
            max_turns=max_turns,
            persona=resolved_persona,
            tools=simulation_config.tools,
            context=simulation_config.context,
            stop_trigger=simulation_config.stop_trigger,
            response_fields=simulation_config.response_fields,
            environment_id=simulation_config.environment_id,
            additional_instructions=simulation_config.additional_instructions,
            custom_simulator=simulation_config.custom_simulator,
        )

        # Build entry dict for the backend
        data_entry: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                data_entry[key] = value
            elif isinstance(value, str):
                data_entry[key] = value
            else:
                data_entry[key] = str(value) if value is not None else None

        # Convert data_entry to variable format (matching JS SDK pattern)
        variable_entry = self._maxim_apis._convert_data_entry_to_variable_format(data_entry)

        try:
            simulation_start_time = time.time()
            while turn_number < max_turns and not is_complete:
                turn_number += 1

                entry_payload: Optional[Dict[str, Any]] = None
                if turn_number == 1:
                    entry_payload = {
                        "input": input_val,
                        "scenario": scenario,
                        "expectedSteps": expected_steps,
                        "contextToEvaluate": context_to_evaluate,
                        "dataEntry": variable_entry,
                    }
                    if resolved_persona is not None:
                        entry_payload["persona"] = resolved_persona
                    
                turn_result = self._maxim_apis.execute_simulation_local_execution(
                    test_run_id=test_run_id,
                    workspace_id=workspace_id,
                    simulation_config=resolved_simulation_config,
                    dataset_entry_id=dataset_entry_id if turn_number == 1 else None,
                    entry=entry_payload,
                    conversation_history=conversation_history if turn_number > 1 else None,
                    test_run_entry_id=test_run_entry_id,
                )

                if turn_number == 1:
                    if turn_result.test_run_entry_id is None:
                        raise ValueError("test_run_entry_id is required on first turn but was not returned by backend")
                    test_run_entry_id = turn_result.test_run_entry_id
                    session_id = turn_result.session_id
                    simulation_id = turn_result.simulation_id

                if turn_result.usage:
                    total_prompt_tokens += turn_result.usage.prompt_tokens
                    total_completion_tokens += turn_result.usage.completion_tokens
                    total_tokens += turn_result.usage.total_tokens
                if turn_result.cost:
                    total_input_cost += turn_result.cost.input_cost
                    total_output_cost += turn_result.cost.output_cost
                    total_cost += turn_result.cost.total_cost

                if turn_result.stop_reason:
                    stop_reason = turn_result.stop_reason
                    logger.info(f"Simulation stopped: {stop_reason}")
                    is_complete = True
                    break

                if turn_result.is_complete:
                    is_complete = True
                    break

                user_input = turn_result.user_input or {}

                sim_context = SimulationContext(
                    conversation_history=list(conversation_history),
                    current_user_input=user_input,
                    turn_number=turn_number,
                    total_cost=float(total_cost),
                    total_tokens=int(total_tokens),
                )
                assistant_output = output_function(row, sim_context)
                if isinstance(assistant_output, Awaitable):
                    assistant_output = asyncio.run(process_awaitable(assistant_output))

                response: Dict[str, Any] = {
                    "output": assistant_output.data,
                    "tool_calls": assistant_output.tool_calls or [],
                }

                simulation_outputs.append(assistant_output.data)

                normalized_request: Dict[str, Any] = {"input": user_input.get("input", "") if isinstance(user_input, dict) else str(user_input)}

                conversation_history.append(SimulationConversationTurn(
                    turn=turn_number,
                    request=normalized_request,
                    response=response,
                ))

                if simulation_config.stop_trigger is not None:
                    if not isinstance(simulation_config.stop_trigger, dict):
                        raise TypeError("stop_trigger must be a dict")
                    field = simulation_config.stop_trigger.get("field")
                    if not field or not isinstance(field, str):
                        raise ValueError("stop_trigger.field must be a non-empty string")

                    if simulation_config.stop_trigger:
                        field_value = self._get_nested_field_value(assistant_output, simulation_config.stop_trigger.get("field", ""))
                        if field_value == simulation_config.stop_trigger.get("value"):
                            is_complete = True
                            break

            total_latency = (time.time() - simulation_start_time) * 1000

            last_turn: Optional[Dict[str, Any]] = None
            if conversation_history:
                last = conversation_history[-1]
                last_turn = {
                    "turn": last.turn,
                    "request": last.request,
                    "response": last.response,
                }

            return YieldedOutput(
                data=simulation_outputs[-1] if simulation_outputs else "",
                simulation_outputs=simulation_outputs,
                simulation_meta=SimulationMeta(
                    test_run_entry_id=test_run_entry_id,
                    session_id=session_id,
                    simulation_id=simulation_id,
                    messages=conversation_history,
                    last_turn=last_turn,
                    stop_reason=stop_reason,
                    usage=YieldedOutputTokenUsage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                        latency=total_latency,
                    ),
                    cost=YieldedOutputCost(
                        input_cost=total_input_cost,
                        output_cost=total_output_cost,
                        total_cost=total_cost,
                    ),
                ),
            )
        except Exception:
            if test_run_entry_id:
                try:
                    self._maxim_apis.update_simulation_status(test_run_entry_id, "FAILED")
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to mark simulation as failed (testRunEntryId: {test_run_entry_id}): {cleanup_error}"
                    )
            raise

    def __process_entry(
        self,
        index: int,
        input: Optional[str],
        expected_output: Optional[str],
        context_to_evaluate: Optional[Union[str, List[str]]],
        scenario: Optional[str],
        expected_steps: Optional[str],
        output_function: Optional[
            Callable[..., Union[YieldedOutput, Awaitable[YieldedOutput]]]
        ],
        get_row: Callable[[int], Optional[LocalData]],
        logger: TestRunLogger,
        evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
            str, EvaluatorNameToIdAndPassFailCriteria
        ],
        test_run_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        dataset_entry_id: Optional[str] = None,
        entry_id: Optional[str] = None,
        connected_trace_id: Optional[str] = None,
    ) -> ProcessedEntry:
        """
        Process a single test run entry

        Args:
            index (int): The index of the entry
            keys (dict[str, Optional[str]]): Mapping of column names to keys in the data
            output_function (Callable): Function to generate output
            get_row (Callable[[int], Optional[dict[str, Any]]]): Function to retrieve a row from the dataset
            logger (TestRunLogger): Logger instance

        Returns:
            ProcessedEntry: Contains processed entry and metadata
        """
        row = get_row(index)
        if row is None:
            raise ValueError(f"Dataset entry {index} is missing")

        output: Optional[Union[Awaitable[YieldedOutput], YieldedOutput]]
        if self._config.simulation_config is not None and output_function is not None:
            if not test_run_id:
                raise ValueError("test_run_id is required for local simulation")
            if not workspace_id:
                raise ValueError("workspace_id is required for local simulation")
            output = self._run_simulation_with_local_output(
                row=row,
                test_run_id=test_run_id,
                workspace_id=workspace_id,
                dataset_entry_id=dataset_entry_id,
                input_val=input,
                scenario=scenario,
                expected_steps=expected_steps,
                context_to_evaluate=context_to_evaluate,
                output_function=output_function,
                logger=logger,
            )
        elif output_function is not None:
            if (
                output_function is self._config.output_function_with_tracing
                and connected_trace_id is not None
            ):
                output = output_function(row, connected_trace_id)
            else:
                output = output_function(row)
        elif self._config.workflow is not None:
            # Check if we need to use simulation endpoints (simulationConfig + local evaluators)
            has_local_evaluators = any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
            use_simulation_endpoints = self._config.simulation_config is not None
            
            if use_simulation_endpoints:
                if not test_run_id:
                    raise ValueError("test_run_id is required for simulation endpoints")
                if not workspace_id:
                    raise ValueError("workspace_id is required for simulation endpoints")

                data_entry: Dict[str, Union[str, List[str], None, Dict[str, Any]]] = {}
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        data_entry[key] = value
                    elif isinstance(value, str):
                        data_entry[key] = value
                    else:
                        data_entry[key] = str(value) if value is not None else None

                context_to_evaluate_for_simulation = context_to_evaluate or self._config.workflow.context_to_evaluate

                start_resp = self._maxim_apis.execute_simulation_workflow_start(
                    test_run_id=test_run_id,
                    workflow_id=self._config.workflow.id,
                    workspace_id=workspace_id,
                    simulation_config=self._config.simulation_config,
                    dataset_entry_id=dataset_entry_id,
                    input=input,
                    scenario=scenario,
                    expected_steps=expected_steps,
                    context_to_evaluate=context_to_evaluate_for_simulation,
                    data_entry=data_entry,
                )
                fetch = lambda: self._maxim_apis.get_simulation_workflow_status(
                    start_resp.workspace_id, start_resp.test_run_entry_id
                )
                parse = lambda d: ExecuteSimulationWorkflowForDataResponse.dict_to_class(d)
                simulation_output = self._poll_simulation_result(fetch, parse)

                outputs = simulation_output.outputs or []
                output_str = outputs[-1] if outputs else (simulation_output.output or "")
                output = YieldedOutput(
                    data=output_str,
                    simulation_outputs=outputs if outputs else None,
                    retrieved_context_to_evaluate=simulation_output.context_to_evaluate,
                    messages=simulation_output.messages,
                    simulation_meta=SimulationMeta(
                        session_id=simulation_output.session_id,
                        simulation_id=simulation_output.simulation_id,
                        messages=simulation_output.messages or [],
                        trace=simulation_output.trace,
                        turns=simulation_output.turns,
                    ),
                    meta=YieldedOutputMeta(
                        entity_type="WORKFLOW",
                        entity_id=self._config.workflow.id,
                        usage=simulation_output.usage,
                        cost=simulation_output.cost,
                    ),
                )
            else:
                workflow_output = self._maxim_apis.execute_workflow_for_data(
                    self._config.workflow.id, row, self._config.workflow.context_to_evaluate
                )
                output = YieldedOutput(
                    data=(
                        workflow_output.output if workflow_output.output is not None else ""
                    ),
                    retrieved_context_to_evaluate=workflow_output.context_to_evaluate,
                    meta=YieldedOutputMeta(
                        entity_type="WORKFLOW",
                        entity_id=self._config.workflow.id,
                        usage=YieldedOutputTokenUsage(
                            latency=workflow_output.latency,
                            completion_tokens=0,
                            prompt_tokens=0,
                            total_tokens=0,
                        ),
                    ),
                )
        elif self._config.prompt_version is not None:
            variables = (
                get_variables_from_row(row, self._config.data_structure)
                if self._config.data_structure
                else {}
            )
            
            # Check if we need to use simulation endpoints (simulationConfig + local evaluators)
            has_local_evaluators = any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
            use_simulation_endpoints = self._config.simulation_config is not None
            
            if use_simulation_endpoints:
                if not test_run_id:
                    raise ValueError("test_run_id is required for simulation endpoints")
                if not workspace_id:
                    raise ValueError("workspace_id is required for simulation endpoints")

                data_entry: Dict[str, Union[str, List[str], None, Dict[str, Any]]] = {}
                for key, variable in variables.items():
                    if variable.type == "text":
                        data_entry[key] = variable.payload
                    elif variable.type == "file":
                        data_entry[key] = variable.to_json()
                    else:
                        data_entry[key] = str(variable.payload) if variable.payload is not None else None

                context_to_evaluate_for_simulation = context_to_evaluate or self._config.prompt_version.context_to_evaluate

                start_resp = self._maxim_apis.execute_simulation_prompt_start(
                    test_run_id=test_run_id,
                    prompt_version_id=self._config.prompt_version.id,
                    workspace_id=workspace_id,
                    simulation_config=self._config.simulation_config,
                    dataset_entry_id=dataset_entry_id,
                    input=input,
                    scenario=scenario,
                    expected_steps=expected_steps,
                    context_to_evaluate=context_to_evaluate_for_simulation,
                    data_entry=data_entry,
                )
                fetch = lambda: self._maxim_apis.get_simulation_prompt_status(
                    start_resp.workspace_id, start_resp.test_run_entry_id
                )
                parse = lambda d: ExecuteSimulationPromptForDataResponse.dict_to_class(d)
                simulation_output = self._poll_simulation_result(fetch, parse)

                outputs = simulation_output.outputs or []
                output_str = outputs[-1] if outputs else (simulation_output.output or "")
                output = YieldedOutput(
                    data=output_str,
                    simulation_outputs=outputs if outputs else None,
                    retrieved_context_to_evaluate=simulation_output.context_to_evaluate,
                    messages=simulation_output.messages,
                    simulation_meta=SimulationMeta(
                        session_id=simulation_output.session_id,
                        simulation_id=simulation_output.simulation_id,
                        messages=simulation_output.messages or [],
                        trace=simulation_output.trace,
                    ),
                    meta=YieldedOutputMeta(
                        entity_type="PROMPT",
                        entity_id=self._config.prompt_version.id,
                        usage=simulation_output.usage,
                        cost=simulation_output.cost,
                    ),
                )
            else:
                prompt_output = self._maxim_apis.execute_prompt_for_data(
                    self._config.prompt_version.id,
                    input if input is not None else "",
                    variables,
                    self._config.prompt_version.context_to_evaluate,
                )
                output = YieldedOutput(
                    data=prompt_output.output if prompt_output.output is not None else "",
                    retrieved_context_to_evaluate=prompt_output.context_to_evaluate,
                    meta=YieldedOutputMeta(
                        entity_type="PROMPT",
                        entity_id=self._config.prompt_version.id,
                        usage=prompt_output.usage,
                        cost=prompt_output.cost,
                    ),
                )
        elif self._config.prompt_chain_version is not None:
            variables = (
                get_variables_from_row(row, self._config.data_structure)
                if self._config.data_structure
                else {}
            )
            prompt_chain_output = self._maxim_apis.execute_prompt_chain_for_data(
                self._config.prompt_chain_version.id,
                input if input is not None else "",
                variables,
                self._config.prompt_chain_version.context_to_evaluate,
            )
            output = YieldedOutput(
                data=(
                    prompt_chain_output.output
                    if prompt_chain_output.output is not None
                    else ""
                ),
                retrieved_context_to_evaluate=prompt_chain_output.context_to_evaluate,
                meta=YieldedOutputMeta(
                    entity_type="PROMPT_CHAIN",
                    entity_id=self._config.prompt_chain_version.id,
                    usage=prompt_chain_output.usage,
                    cost=prompt_chain_output.cost,
                ),
            )
        else:
            raise ValueError(
                "Found no output function to execute, please make sure you have either `yields_output`, `with_prompt_version_id`, `with_prompt_chain_version_id` or `with_workflow_id` set."
            )

        yielded_output: YieldedOutput

        if isinstance(output, Awaitable):
            yielded_output = asyncio.run(process_awaitable(output))
        else:
            yielded_output = output

        if yielded_output is not None and yielded_output.retrieved_context_to_evaluate is not None:
            if context_to_evaluate is not None:
                logger.info(
                    "Overriding context_to_evaluate from output over dataset entry"
                )
            context_to_evaluate = yielded_output.retrieved_context_to_evaluate

        local_evaluators: List[BaseEvaluator] = []
        for evaluator in self._config.evaluators:
            if isinstance(evaluator, BaseEvaluator):
                local_evaluators.append(evaluator)
            else:
                continue

        local_evaluation_results_awaitable = run_local_evaluations(
            local_evaluators,
            row,
            LocalEvaluatorResultParameter(
                output=yielded_output.data if yielded_output is not None else "",
                context_to_evaluate=context_to_evaluate,
                simulation_outputs=(
                    yielded_output.simulation_outputs
                    if yielded_output is not None
                    else None
                ),
            ),
        )
        local_evaluation_results = asyncio.run(
            process_awaitable(local_evaluation_results_awaitable)
        )

        local_evaluation_results_with_ids: List[LocalEvaluationResultWithId] = []
        for local_evaluation_result in local_evaluation_results:
            local_evaluation_results_with_ids.append(
                LocalEvaluationResultWithId(
                    result=local_evaluation_result.result,
                    id=evaluator_name_to_id_and_pass_fail_criteria_map[
                        local_evaluation_result.name
                    ].id,
                    name=local_evaluation_result.name,
                    pass_fail_criteria=local_evaluation_result.pass_fail_criteria,
                    output=local_evaluation_result.output,
                    simulation_outputs=local_evaluation_result.simulation_outputs,
                )
            )

        sdk_variables: Dict[str, Dict[str, str]] = {}
        
        variable_mapping_input = VariableMappingInput.from_yielded_output(
            output=yielded_output,
            input_value=input,
            context_to_evaluate=context_to_evaluate,
        )
        
        version_info: Optional[VersionInfo] = None
        if self._config.workflow is not None:
            version_info = VersionInfo({"id": self._config.workflow.id, "type": "workflow"})
        elif self._config.prompt_version is not None:
            version_info = VersionInfo({"id": self._config.prompt_version.id, "type": "prompt"})
        elif self._config.prompt_chain_version is not None:
            version_info = VersionInfo({"id": self._config.prompt_chain_version.id, "type": "promptChain"})
        
        # Process all evaluators with variable_mapping
        for evaluator in self._config.evaluators:
            variable_mapping = None
            evaluator_name = None
            
            if isinstance(evaluator, BaseEvaluator) and evaluator.variable_mapping:
                variable_mapping = evaluator.variable_mapping
                # BaseEvaluator can have multiple names, use first one for the mapping
                evaluator_name = evaluator.names[0] if evaluator.names else None
            elif isinstance(evaluator, PlatformEvaluator) and evaluator.variable_mapping:
                variable_mapping = evaluator.variable_mapping
                evaluator_name = evaluator.name
            
            if variable_mapping and evaluator_name:
                evaluator_info = evaluator_name_to_id_and_pass_fail_criteria_map.get(evaluator_name)
                if evaluator_info:
                    evaluator_id = evaluator_info.id
                    mapping_result: Dict[str, str] = {}
                    
                    for var_name, mapping_fn in variable_mapping.items():
                        try:
                            result = mapping_fn(variable_mapping_input, row, version_info)
                            if result is not None:
                                mapping_result[var_name] = result
                        except Exception as e:
                            logger.error(f"Error in variable mapping for key '{var_name}': {e}")
                    
                    if mapping_result:
                        sdk_variables[evaluator_id] = mapping_result

        return ProcessedEntry(
            entry=TestRunEntry(
                id=entry_id,
                output=yielded_output.data if yielded_output is not None else None,
                input=input,
                expected_output=expected_output,
                context_to_evaluate=context_to_evaluate,
                scenario=scenario,
                expected_steps=expected_steps,
                simulation_meta=yielded_output.simulation_meta if yielded_output is not None else None,
                variables=(
                    get_variables_from_row(row, self._config.data_structure)
                    if self._config.data_structure
                    else {}
                ),
                local_evaluation_results=local_evaluation_results_with_ids,
                sdk_variables=sdk_variables if sdk_variables else None,
                connected_trace_id=connected_trace_id if connected_trace_id is not None else None,
            ),
            meta=yielded_output.meta if yielded_output is not None else None,
        )


    def with_data_structure(self, data: T) -> "TestRunBuilder[T]":
        """
        Set the data structure for the test run

        Args:
            data (T): The data structure to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        sanitize_data_structure(data)
        self._config.data_structure = data
        return self

    def with_data(self, data: Data) -> "TestRunBuilder[T]":
        """
        Set the data for the test run

        Args:
            data (DataValue[T]): The data to use or the ID of the dataset and dataset split to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        if self._config.data_structure is not None:
            sanitize_data(data, self._config.data_structure)
        self._config.data = data
        return self

    def with_evaluators(
        self, *evaluators: Union[str, BaseEvaluator, PlatformEvaluator]
    ) -> "TestRunBuilder[T]":
        """
        Add evaluators to the test run

        Args:
            *evaluators (Union[str, BaseEvaluator, PlatformEvaluator]): The evaluators to add

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        evaluators_list: List[Union[BaseEvaluator, PlatformEvaluator, str]] = []
        for evaluator in evaluators:
            evaluators_list.append(evaluator)
        sanitize_evaluators(evaluators_list)
        self._config.evaluators = evaluators_list
        return self

    
    def with_tags(self, tags: list[str]) -> "TestRunBuilder[T]":
        """
        Set the tags for the test run

        Args:
            tags (list[str]): The tags to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        self._config.tags = tags
        return self

    def with_human_evaluation_config(
        self, config: HumanEvaluationConfig
    ) -> "TestRunBuilder[T]":
        """
        Set the human evaluation configuration for the test run

        Args:
            config (HumanEvaluationConfig): The human evaluation configuration to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        email_regex = re.compile(
            r"^(?!\.)(?!.*\.\.)([A-Z0-9_\'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$",
            re.IGNORECASE,
        )
        invalid_emails = [
            email for email in config.emails if not email_regex.match(email)
        ]
        if len(invalid_emails) > 0:
            raise ValueError(f"Invalid email addresses: {', '.join(invalid_emails)}")
        self._config.human_evaluation_config = config
        return self

    def with_environment(self, environment_name: str) -> "TestRunBuilder[T]":
        """
        Set the environment name for the test run

        Args:
            environment_name (str): The name of the environment to use
        """
        self._config.environment_name = environment_name
        return self

    def with_workflow_id(
        self, workflow_id: Optional[str], context_to_evaluate: Optional[str] = None
    ) -> "TestRunBuilder[T]":
        """
        Set the workflow ID for the test run. Optionally, you can also set the context to evaluate for the workflow. (Note: setting the context to evaluate will end up overriding the CONTEXT_TO_EVALUATE dataset column value)

        Args:
            workflow_id (str): The ID of the workflow to use
            context_to_evaluate (Optional[str]): The context to evaluate for the workflow (variable name essentially).

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If a prompt version ID, prompt chain version ID or output function is already set for this run builder
        """
        if self._config.prompt_version is not None:
            raise ValueError(
                "Prompt version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.prompt_chain_version is not None:
            raise ValueError(
                "Prompt chain version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.output_function is not None:
            raise ValueError(
                "yields_output is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if workflow_id is None or not isinstance(workflow_id, str):
            raise ValueError("Workflow id is required for a test run. Please provide a valid workflow id.")
        self._config.workflow = WorkflowConfig(
            id=workflow_id,
            context_to_evaluate=context_to_evaluate,
        )
        return self

    def with_prompt_version_id(
        self, prompt_version_id: str, context_to_evaluate: Optional[str] = None
    ) -> "TestRunBuilder[T]":
        """
        Set the prompt version ID for the test run. Optionally, you can also set the context to evaluate for the prompt. (Note: setting the context to evaluate will end up overriding the CONTEXT_TO_EVALUATE dataset column value)

        Args:
            prompt_version_id (str): The ID of the prompt version to use
            context_to_evaluate (Optional[str]): The context to evaluate for the prompt (variable name essentially).

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If a workflow ID, prompt chain version ID or output function is already set for this run builder
        """
        if self._config.workflow is not None:
            raise ValueError(
                "Workflow id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.prompt_chain_version is not None:
            raise ValueError(
                "Prompt chain version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.output_function is not None:
            raise ValueError(
                "yields_output is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        self._config.prompt_version = PromptVersionConfig(
            id=prompt_version_id,
            context_to_evaluate=context_to_evaluate,
        )
        return self

    def with_prompt_chain_version_id(
        self, prompt_chain_version_id: str, context_to_evaluate: Optional[str] = None
    ) -> "TestRunBuilder[T]":
        """
        Set the prompt chain version ID for the test run. Optionally, you can also set the context to evaluate for the prompt chain. (Note: setting the context to evaluate will end up overriding the CONTEXT_TO_EVALUATE dataset column value)

        Args:
            prompt_chain_version_id (str): The ID of the prompt chain version to use
            context_to_evaluate (Optional[str]): The context to evaluate for the prompt chain (variable name essentially).

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If a workflow ID, prompt version ID or output function is already set for this run builder
        """
        if self._config.workflow is not None:
            raise ValueError(
                "Workflow id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.prompt_version is not None:
            raise ValueError(
                "Prompt version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.output_function is not None:
            raise ValueError(
                "yields_output is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        self._config.prompt_chain_version = PromptChainVersionConfig(
            id=prompt_chain_version_id,
            context_to_evaluate=context_to_evaluate,
        )
        return self

    def with_simulation_config(
        self, simulation_config: SimulationConfig
    ) -> "TestRunBuilder[T]":
        """
        Set the simulation configuration for the test run. Use this to run AI-simulated multi-turn conversations.

        When used with yields_output(), the SDK runs your output function locally in a turn-by-turn loop.
        You may optionally set with_prompt_version_id() or with_workflow_id() for prompt/workflow-based simulation,
        or omit both for SDK-only simulation (no prompt or workflow dependency).

        Args:
            simulation_config (SimulationConfig): The simulation configuration.

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If simulation config is used with promptChainVersion
        """
        if self._config.prompt_chain_version is not None:
            raise ValueError(
                "Simulation config cannot be used with withPromptChainVersionId. Use withWorkflowId or withPromptVersionId instead."
            )
        self._config.simulation_config = simulation_config
        return self

    def yields_output(
        self,
        output_function: Callable[
            ..., Union[YieldedOutput, Awaitable[YieldedOutput]]
        ],
    ) -> "TestRunBuilder[T]":
        """
        Set the output function for the test run.

        When combined with with_simulation_config(), this enables local-execution simulation
        where your output function is called turn-by-turn with simulation context.
        You may optionally set with_prompt_version_id() or with_workflow_id(), or omit both for SDK-only simulation.

        Args:
            output_function: The output function to use. Accepts (data) for test_runs and for simulation (data, simulation_context).
        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If incompatible configuration is already set
        """
        # Only validate mutual exclusivity when simulation_config is None.
        if self._config.simulation_config is not None:
            if self._config.prompt_version is not None or self._config.workflow is not None:
                raise ValueError(
                    "simulation_config with yields_output cannot be used with with_prompt_version_id or with_workflow_id. "
                    "For local-execution simulation, omit with_prompt_version_id and with_workflow_id (SDK-only simulation)."
            )
        if self._config.simulation_config is None:
            if self._config.workflow is not None:
                raise ValueError(
                    "Workflow id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
                )
            if self._config.prompt_chain_version is not None:
                raise ValueError(
                    "Prompt chain version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
                )
            if self._config.prompt_version is not None:
                raise ValueError(
                    "Prompt version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
                )
            if self._config.output_function_with_tracing is not None:
                raise ValueError(
                    "output_function_with_tracing is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
                )
        self._config.output_function = output_function
        return self
    
    def yields_output_with_tracing(
        self,
        output_function: Callable[
            [LocalData, str], Union[YieldedOutput, Awaitable[YieldedOutput]]
        ],
        maxim_logger: Logger,
        disable_default_trace_creation: bool = False,
    ) -> "TestRunBuilder[T]":
        """
        Set the output function for the test run with tracing.
        The output function receives (row, trace_id) - pass trace_id to LLM calls
        (e.g. via x-maxim-trace-id header) to associate them with the test run entry trace.

        Args:
            output_function: Callable receiving (row, trace_id) returning YieldedOutput.
                Use trace_id when making LLM/API calls to link traces.

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If a workflow ID, prompt chain version ID or prompt version ID is already set for this run builder
        """
        if self._config.workflow is not None:
            raise ValueError(
                "Workflow id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
            )
        if self._config.prompt_chain_version is not None:
            raise ValueError(
                "Prompt chain version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
            )
        if self._config.prompt_version is not None:
            raise ValueError(
                "Prompt version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
            )
        if self._config.output_function is not None:
            raise ValueError(
                "output_function is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id, yields_output or yields_output_with_tracing in a test run."
            )
        self._config.output_function_with_tracing = output_function
        self._config.internal_maxim_logger = maxim_logger
        self._config.disable_default_trace_creation = disable_default_trace_creation
        return self


    def with_concurrency(self, concurrency: int) -> "TestRunBuilder[T]":
        """
        Set the concurrency level for the test run

        Args:
            concurrency (int): The concurrency level to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        self._config.concurrency = concurrency
        return self

    def with_logger(self, logger: TestRunLogger) -> "TestRunBuilder[T]":
        """
        Set the logger for the test run

        Args:
            logger (TestRunLogger): The logger to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        self._config.logger = logger
        return self

    def _run_test_with_local_data(
        self,
        test_run: TestRun,
        get_row: Callable[[int], Optional[LocalData]],
        on_entry_failed: Callable[[int], None],
        on_dataset_finished: Callable[[], None],
        evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
            str, EvaluatorNameToIdAndPassFailCriteria
        ],
    ):
        """
        Run the test with local data

        Args:
            test_run (TestRun): The test run to execute
            get_row (Callable[[int], Optional[Dict[str, Any]]]): Function to retrieve a row from the dataset
            on_entry_failed (Callable[[int], None]): Callback for when an entry fails
            on_dataset_finished (Callable[[], None]): Callback for when the dataset is finished
        """
        semaphore = Semaphore.get(
            f"test_run:{test_run.workspace_id}:{test_run.id}",
            self._config.concurrency or 10,
        )
        data_structure = self._config.data_structure
        try:
            input_key = get_all_keys_by_value(data_structure, "INPUT")[0]
        except IndexError:
            input_key = None
        try:
            expectedOutputKey = get_all_keys_by_value(
                data_structure, "EXPECTED_OUTPUT"
            )[0]
        except IndexError:
            expectedOutputKey = None
        try:
            contextToEvaluateKey = get_all_keys_by_value(
                data_structure, "CONTEXT_TO_EVALUATE"
            )[0]
        except IndexError:
            contextToEvaluateKey = None
        try:
            scenarioKey = get_all_keys_by_value(
                data_structure, "SCENARIO"
            )[0]
        except IndexError:
            scenarioKey = None
        try:
            expectedStepsKey = get_all_keys_by_value(
                data_structure, "EXPECTED_STEPS"
            )[0]
        except IndexError:
            expectedStepsKey = None

        def process_row(
            index: int,
            row: LocalData,
            evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
                str, EvaluatorNameToIdAndPassFailCriteria
            ],
        ) -> None:
            try:
                if row is None:
                    raise ValueError(f"Dataset entry {index} is missing")
                input, expected_output, context_to_evaluate, scenario, expected_steps = (
                    get_input_expected_output_and_context_from_row(
                        input_key, expectedOutputKey, contextToEvaluateKey, scenarioKey, expectedStepsKey, row
                    )
                )
                if (
                    any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
                    or self._config.output_function is not None
                    or self._config.output_function_with_tracing is not None
                     or (
                        self._config.simulation_config is not None
                        and (self._config.workflow is not None or self._config.prompt_version is not None)
                    )
                ):
                    test_run_entry = None
                    if self._config.simulation_config is None:
                        test_run_entry = self._maxim_apis.create_test_run_entry(
                        test_run=test_run
                    )
                    
                    trace_id = str(uuid4())
                    if not self._config.disable_default_trace_creation and self._config.internal_maxim_logger:
                        try:
                            test_run_entry_trace = self._config.internal_maxim_logger.trace(config=TraceConfigDict(id=trace_id, name=f"Test Run Entry {index + 1}"))
                            test_run_entry_trace.add_tag("testRunEntryId", test_run_entry.get("id", trace_id))
                            test_run_entry_trace.add_tag("testRunId", test_run.id)
                        except Exception as e:
                            self._config.logger.error(f"Error creating trace for test run entry {index + 1}: {e}")

                    result = self.__process_entry(
                        entry_id=test_run_entry.get("id", None) if test_run_entry is not None else None,
                        connected_trace_id=trace_id,
                        index=index,
                        input=input,
                        expected_output=expected_output,
                        context_to_evaluate=context_to_evaluate,
                        scenario=scenario,
                        expected_steps=expected_steps,
                        output_function=self._config.output_function or self._config.output_function_with_tracing,
                        get_row=lambda index: row,
                        logger=self._config.logger,
                        evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                        test_run_id=test_run.id,
                        workspace_id=test_run.workspace_id,
                        dataset_entry_id=None,
                    )
                    is_local_sim = (
                        self._config.simulation_config is not None
                        and self._config.output_function is not None
                    )
                    self._maxim_apis.push_test_run_entry(
                        test_run=test_run,
                        entry=result.entry,
                        run_config=(
                            None if is_local_sim else (
                                {
                                    "cost": (
                                        result.meta.cost.to_dict()
                                        if result.meta.cost is not None
                                        else None
                                    ),
                                    "usage": (
                                        result.meta.usage.to_dict()
                                        if result.meta.usage is not None
                                        else None
                                    ),
                                }
                                if result.meta is not None
                                else None
                            )
                        ),
                        local_simulation=is_local_sim or None,
                    )
                else:
                    # Check if we have platform evaluators with variable_mapping
                    has_platform_evaluator_with_mapping = any(
                        isinstance(e, PlatformEvaluator) and e.variable_mapping 
                        for e in self._config.evaluators
                    )
                    
                    sdk_variables: Optional[Dict[str, Dict[str, str]]] = None
                    
                    if has_platform_evaluator_with_mapping:
                        # We need to execute the prompt/workflow to get output for variable mapping
                        yielded_output: Optional[YieldedOutput] = None
                        
                        if self._config.prompt_version is not None:
                            # Convert Variable dict to string dict for run_prompt_version
                            variable_dict = (
                                get_variables_from_row(row, self._config.data_structure)
                                if self._config.data_structure
                                else {}
                            )
                            # Convert Variable objects to strings
                            variables_str = {
                                k: v.payload if hasattr(v, 'payload') else str(v)
                                for k, v in variable_dict.items()
                            }
                            
                            prompt_response = self._maxim_apis.run_prompt_version(
                                self._config.prompt_version.id,
                                input if input is not None else "",
                                None,  # image_urls
                                variables_str,
                            )
                            
                            if prompt_response is not None:
                                # Extract output from the response
                                output_text = ""
                                if prompt_response.choices and len(prompt_response.choices) > 0:
                                    content = prompt_response.choices[0].message.content
                                    if isinstance(content, str):
                                        output_text = content
                                    elif isinstance(content, list):
                                        # Multimodal content - extract text parts
                                        output_text = " ".join(
                                            item.get("text", "") for item in content 
                                            if isinstance(item, dict) and item.get("type") == "text"
                                        )
                                
                                yielded_output = YieldedOutput(
                                    data=output_text,
                                    retrieved_context_to_evaluate=self._config.prompt_version.context_to_evaluate,
                                    meta=YieldedOutputMeta(
                                        entity_type="PROMPT",
                                        entity_id=self._config.prompt_version.id,
                                        usage=YieldedOutputTokenUsage(
                                            prompt_tokens=prompt_response.usage.prompt_tokens,
                                            completion_tokens=prompt_response.usage.completion_tokens,
                                            total_tokens=prompt_response.usage.total_tokens,
                                            latency=prompt_response.usage.latency,
                                        ),
                                    ),
                                )
                        elif self._config.prompt_chain_version is not None:
                            # Convert Variable dict to string dict
                            variable_dict = (
                                get_variables_from_row(row, self._config.data_structure)
                                if self._config.data_structure
                                else {}
                            )
                            variables_str = {
                                k: v.payload if hasattr(v, 'payload') else str(v)
                                for k, v in variable_dict.items()
                            }
                            
                            agent_response = self._maxim_apis.run_prompt_chain_version(
                                self._config.prompt_chain_version.id,
                                input if input is not None else "",
                                variables_str,
                            )
                            
                            if agent_response is not None:
                                meta = getattr(agent_response, "meta", None)
                                usage = getattr(meta, "usage", None) if meta is not None else None
                                yielded_output = YieldedOutput(
                                    data=agent_response.response or "",
                                    retrieved_context_to_evaluate=self._config.prompt_chain_version.context_to_evaluate,
                                    meta=YieldedOutputMeta(
                                        entity_type="PROMPT_CHAIN",
                                        entity_id=self._config.prompt_chain_version.id,
                                        usage=YieldedOutputTokenUsage(
                                            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                                            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                                            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                                            latency=getattr(usage, "latency", 0) if usage else 0,
                                        ),
                                    ),
                                )
                        elif self._config.workflow is not None:
                            workflow_output = self._maxim_apis.execute_workflow_for_data(
                                self._config.workflow.id, row, self._config.workflow.context_to_evaluate
                            )
                            yielded_output = YieldedOutput(
                                data=workflow_output.output if workflow_output.output is not None else "",
                                retrieved_context_to_evaluate=workflow_output.context_to_evaluate,
                                meta=YieldedOutputMeta(
                                    entity_type="WORKFLOW",
                                    entity_id=self._config.workflow.id,
                                    usage=YieldedOutputTokenUsage(
                                        latency=workflow_output.latency,
                                        completion_tokens=0,
                                        prompt_tokens=0,
                                        total_tokens=0,
                                    ),
                                ),
                            )
                        
                        if yielded_output is not None:
                            # Compute sdk_variables for platform evaluators with variable_mapping
                            sdk_variables = self._compute_sdk_variables_for_platform_evaluators(
                                yielded_output, row, input, context_to_evaluate, evaluator_name_to_id_and_pass_fail_criteria_map, self._config.logger
                            )
                    
                    # Push the entry with sdk_variables if computed
                    self._maxim_apis.push_test_run_entry(
                        test_run=test_run,
                        entry=TestRunEntry(
                            variables=(
                                get_variables_from_row(row, self._config.data_structure)
                                if self._config.data_structure
                                else {}
                            ),
                            input=input,
                            expected_output=expected_output,
                            context_to_evaluate=context_to_evaluate,
                            sdk_variables=sdk_variables,
                            expected_steps=expected_steps,
                            scenario=scenario
                        ),
                    )
            except Exception as e:
                self._config.logger.error(
                    f"Error while running data entry at index [{index}]: {str(e)}"
                )
                on_entry_failed(index)
            finally:
                semaphore.release()

        def process_all_entries() -> None:
            threads = []
            index = 0
            while True:
                try:
                    semaphore.acquire()
                    # getting the entry
                    row = get_row(index)
                    if row is None:
                        on_dataset_finished()
                        semaphore.release()
                        break
                    # sanitizing data
                    try:
                        if self._config.data_structure is None:
                            raise ValueError(
                                "Data structure is required to run a test with local data as a function"
                            )
                        sanitize_data(row, self._config.data_structure)
                    except ValueError as e:
                        self._config.logger.error(
                            f"Invalid data entry at index [{index}]: {str(e)}"
                        )
                        on_entry_failed(index)
                        # Release semaphore since we're not starting a thread for this entry
                        semaphore.release()
                        index += 1
                        continue
                    current_index = index
                    index += 1
                    thread = threading.Thread(
                        target=process_row,
                        args=(
                            current_index,
                            row,
                            evaluator_name_to_id_and_pass_fail_criteria_map,
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                except Exception as e:
                    self._config.logger.error(
                        f"Error while running data entry at index [{index}]: {str(e)}",
                    )
                    on_entry_failed(index)
                    semaphore.release()

        thread = threading.Thread(target=process_all_entries, args=())
        thread.start()

    def _run_test_with_dataset_id_or_split_id(
        self,
        test_run: TestRun,
        dataset_id: str,
        on_entry_failed: Callable[[int], None],
        on_dataset_finished: Callable[[], None],
        evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
            str, EvaluatorNameToIdAndPassFailCriteria
        ],
    ) -> None:
        """
        Run the test with a dataset ID

        Args:
            test_run (TestRun): The test run to execute
            dataset_id (str): The ID of the dataset to use
            on_entry_failed (Callable[[int], None]): Callback for when an entry fails
            on_dataset_finished (Callable[[], None]): Callback for when the dataset is finished
        """
        semaphore = Semaphore.get(
            f"test_run:{test_run.workspace_id}:{test_run.id}",
            self._config.concurrency or 10,
        )
        data_structure = self._maxim_apis.get_dataset_structure(dataset_id)
        self._maxim_apis.attach_dataset_to_test_run(
            test_run_id=test_run.id, dataset_id=dataset_id
        )
        try:
            input_key = get_all_keys_by_value(data_structure, "INPUT")[0]
        except IndexError:
            input_key = None
        try:
            expectedOutputKey = get_all_keys_by_value(
                data_structure, "EXPECTED_OUTPUT"
            )[0]
        except IndexError:
            expectedOutputKey = None
        try:
            contextToEvaluateKey = get_all_keys_by_value(
                data_structure, "CONTEXT_TO_EVALUATE"
            )[0]
        except IndexError:
            contextToEvaluateKey = None
        try:
            scenarioKey = get_all_keys_by_value(
                data_structure, "SCENARIO"
            )[0]
        except IndexError:
            scenarioKey = None
        try:
            expectedStepsKey = get_all_keys_by_value(
                data_structure, "EXPECTED_STEPS"
            )[0]
        except IndexError:
            expectedStepsKey = None

        def process_dataset_entry(
            index: int,
            row: DatasetRow,
            dataset_id: str,
            evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
                str, EvaluatorNameToIdAndPassFailCriteria
            ],
        ) -> None:
            try:
                row_data: LocalData = row.to_dict()["data"]
                if row_data is None:
                    raise ValueError(f"Dataset entry {index} is missing")
                input, expected_output, context_to_evaluate, scenario, expected_steps = (
                    get_input_expected_output_and_context_from_row(
                        input_key, expectedOutputKey, contextToEvaluateKey, scenarioKey, expectedStepsKey, row_data
                    )
                )

                if (
                    any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
                    or self._config.output_function is not None
                    or self._config.output_function_with_tracing is not None
                    or (
                        self._config.simulation_config is not None
                        and (self._config.workflow is not None or self._config.prompt_version is not None)
                    )
                ):
                    test_run_entry = None
                    if self._config.simulation_config is None:
                        test_run_entry = self._maxim_apis.create_test_run_entry(
                            test_run=test_run
                        )

                    trace_id = str(uuid4())
                    if not self._config.disable_default_trace_creation and self._config.internal_maxim_logger:
                        try:
                            test_run_entry_trace = self._config.internal_maxim_logger.trace(config=TraceConfigDict(id=trace_id, name=f"Test Run Entry {index + 1}"))
                            test_run_entry_trace.add_tag("testRunEntryId", test_run_entry.get("id", trace_id))
                            test_run_entry_trace.add_tag("testRunId", test_run.id)
                        except Exception as e:
                            self._config.logger.error(f"Error creating trace for test run entry {index + 1}: {e}")

                    # processing the entry
                    result = self.__process_entry(
                        entry_id=test_run_entry.get("id", None) if test_run_entry is not None else None,
                        connected_trace_id=trace_id,
                        index=index,
                        input=input,
                        expected_output=expected_output,
                        context_to_evaluate=context_to_evaluate,
                        scenario=scenario,
                        expected_steps=expected_steps,
                        output_function=self._config.output_function or self._config.output_function_with_tracing,
                        get_row=lambda index: row.to_dict()["data"],
                        logger=self._config.logger,
                        evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                        test_run_id=test_run.id,
                        workspace_id=test_run.workspace_id,
                        dataset_entry_id=row.id,
                    )
                    is_local_sim = (
                        self._config.simulation_config is not None
                        and self._config.output_function is not None
                    )
                    self._maxim_apis.push_test_run_entry(
                        test_run=TestRunWithDatasetEntry(
                            test_run=test_run,
                            dataset_id=dataset_id,
                            dataset_entry_id=row.id,
                        ),
                        entry=result.entry,
                        run_config=(
                            None if is_local_sim else (
                                {
                                    "cost": (
                                        result.meta.cost.to_dict()
                                        if result.meta.cost is not None
                                        else None
                                    ),
                                    "usage": (
                                        result.meta.usage.to_dict()
                                        if result.meta.usage is not None
                                        else None
                                    ),
                                }
                                if result.meta
                                else None
                            )
                        ),
                        local_simulation=is_local_sim or None,
                    )
                else:
                    # Check if we have platform evaluators with variable_mapping
                    has_platform_evaluator_with_mapping = any(
                        isinstance(e, PlatformEvaluator) and e.variable_mapping 
                        for e in self._config.evaluators
                    )
                    
                    sdk_variables: Optional[Dict[str, Dict[str, str]]] = None
                    
                    if has_platform_evaluator_with_mapping:
                        # We need to execute the prompt/workflow to get output for variable mapping
                        yielded_output: Optional[YieldedOutput] = None
                        
                        if self._config.prompt_version is not None:
                            # Convert Variable dict to string dict for run_prompt_version
                            variable_dict = (
                                get_variables_from_row(row_data, data_structure)
                                if data_structure
                                else {}
                            )
                            # Convert Variable objects to strings
                            variables_str = {
                                k: v.payload if hasattr(v, 'payload') else str(v)
                                for k, v in variable_dict.items()
                            }
                            
                            prompt_response = self._maxim_apis.run_prompt_version(
                                self._config.prompt_version.id,
                                input if input is not None else "",
                                None,  # image_urls
                                variables_str,
                            )
                            
                            if prompt_response is not None:
                                # Extract output from the response
                                output_text = ""
                                if prompt_response.choices and len(prompt_response.choices) > 0:
                                    content = prompt_response.choices[0].message.content
                                    if isinstance(content, str):
                                        output_text = content
                                    elif isinstance(content, list):
                                        # Multimodal content - extract text parts
                                        output_text = " ".join(
                                            item.get("text", "") for item in content 
                                            if isinstance(item, dict) and item.get("type") == "text"
                                        )
                                
                                yielded_output = YieldedOutput(
                                    data=output_text,
                                    retrieved_context_to_evaluate=self._config.prompt_version.context_to_evaluate,
                                    meta=YieldedOutputMeta(
                                        entity_type="PROMPT",
                                        entity_id=self._config.prompt_version.id,
                                        usage=YieldedOutputTokenUsage(
                                            prompt_tokens=prompt_response.usage.prompt_tokens,
                                            completion_tokens=prompt_response.usage.completion_tokens,
                                            total_tokens=prompt_response.usage.total_tokens,
                                            latency=prompt_response.usage.latency,
                                        ),
                                    ),
                                )
                        elif self._config.prompt_chain_version is not None:
                            # Convert Variable dict to string dict
                            variable_dict = (
                                get_variables_from_row(row_data, data_structure)
                                if data_structure
                                else {}
                            )
                            variables_str = {
                                k: v.payload if hasattr(v, 'payload') else str(v)
                                for k, v in variable_dict.items()
                            }
                            
                            agent_response = self._maxim_apis.run_prompt_chain_version(
                                self._config.prompt_chain_version.id,
                                input if input is not None else "",
                                variables_str,
                            )
                            
                            if agent_response is not None:
                                meta = getattr(agent_response, "meta", None)
                                usage = getattr(meta, "usage", None) if meta is not None else None
                                yielded_output = YieldedOutput(
                                    data=agent_response.response or "",
                                    retrieved_context_to_evaluate=self._config.prompt_chain_version.context_to_evaluate,
                                    meta=YieldedOutputMeta(
                                        entity_type="PROMPT_CHAIN",
                                        entity_id=self._config.prompt_chain_version.id,
                                        usage=YieldedOutputTokenUsage(
                                            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                                            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                                            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                                            latency=getattr(usage, "latency", 0) if usage else 0,
                                        ),
                                    ),
                                )

                        elif self._config.workflow is not None:
                            workflow_output = self._maxim_apis.execute_workflow_for_data(
                                self._config.workflow.id, row_data, self._config.workflow.context_to_evaluate
                            )
                            yielded_output = YieldedOutput(
                                data=workflow_output.output if workflow_output.output is not None else "",
                                retrieved_context_to_evaluate=workflow_output.context_to_evaluate,
                                meta=YieldedOutputMeta(
                                    entity_type="WORKFLOW",
                                    entity_id=self._config.workflow.id,
                                    usage=YieldedOutputTokenUsage(
                                        latency=workflow_output.latency,
                                        completion_tokens=0,
                                        prompt_tokens=0,
                                        total_tokens=0,
                                    ),
                                ),
                            )
                        
                        if yielded_output is not None:
                            # Compute sdk_variables
                            sdk_variables = self._compute_sdk_variables_for_platform_evaluators(
                                yielded_output=yielded_output,
                                row=row_data,
                                input_value=input,
                                context_to_evaluate=context_to_evaluate,
                                evaluator_name_to_id_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                                logger=self._config.logger,
                            )
                            
                            # Push with output and sdk_variables
                            self._maxim_apis.push_test_run_entry(
                                test_run=TestRunWithDatasetEntry(
                                    test_run=test_run,
                                    dataset_id=dataset_id,
                                    dataset_entry_id=row.id,
                                ),
                                entry=TestRunEntry(
                                    variables=(
                                        get_variables_from_row(row_data, data_structure)
                                        if data_structure
                                        else {}
                                    ),
                                    input=input,
                                    expected_output=expected_output,
                                    context_to_evaluate=context_to_evaluate,
                                    output=yielded_output.data,
                                    sdk_variables=sdk_variables,
                                    scenario=scenario,
                                    expected_steps=expected_steps,
                                ),
                                run_config=(
                                    {
                                        "cost": (
                                            yielded_output.meta.cost.to_dict()
                                            if yielded_output.meta and yielded_output.meta.cost is not None
                                            else None
                                        ),
                                        "usage": (
                                            yielded_output.meta.usage.to_dict()
                                            if yielded_output.meta and yielded_output.meta.usage is not None
                                            else None
                                        ),
                                    }
                                    if yielded_output.meta is not None
                                    else None
                                ),
                            )
                        else:
                            # No yielded output, push without sdk_variables
                            self._maxim_apis.push_test_run_entry(
                                test_run=TestRunWithDatasetEntry(
                                    test_run=test_run,
                                    dataset_id=dataset_id,
                                    dataset_entry_id=row.id,
                                ),
                                entry=TestRunEntry(
                                    variables=(
                                        get_variables_from_row(row_data, data_structure)
                                        if data_structure
                                        else {}
                                    ),
                                    input=input,
                                    expected_output=expected_output,
                                    context_to_evaluate=context_to_evaluate,
                                    scenario=scenario,
                                    expected_steps=expected_steps,
                                ),
                            )
                    else:
                        # No platform evaluators with variable_mapping, push directly
                        self._maxim_apis.push_test_run_entry(
                            test_run=TestRunWithDatasetEntry(
                                test_run=test_run,
                                dataset_id=dataset_id,
                                dataset_entry_id=row.id,
                            ),
                            entry=TestRunEntry(
                                variables=(
                                    get_variables_from_row(row_data, data_structure)
                                    if data_structure
                                    else {}
                                ),
                                input=input,
                                expected_output=expected_output,
                                context_to_evaluate=context_to_evaluate,
                                expected_steps=expected_steps,
                                scenario=scenario,
                            ),
                        )
            except Exception as e:
                self._config.logger.error(
                    f"Error while running data entry at index [{index}]: {str(e)}",
                )
                on_entry_failed(index)
                raise e
            finally:
                # The work is complete, so we can release the semaphore
                semaphore.release()

        def process_all_dataset_entries(dataset_id: str) -> None:
            threads = []
            index = 0
            total_rows = self._maxim_apis.get_dataset_total_rows(dataset_id)
            for index in range(total_rows):
                try:
                    semaphore.acquire()
                    # getting the entry
                    row = self._maxim_apis.get_dataset_row(dataset_id, index)
                    if row is None:
                        semaphore.release()
                        break
                    thread = threading.Thread(
                        target=process_dataset_entry,
                        args=(
                            index,
                            row,
                            dataset_id,
                            evaluator_name_to_id_and_pass_fail_criteria_map,
                        ),
                    )
                    thread.start()
                    threads.append(thread)
                except Exception as e:
                    self._config.logger.error(
                        f"Error while running data entry at index [{index}]: {str(e)}"
                    )
                    on_entry_failed(index)
                    semaphore.release() # Releasing semaphore here since thread won't be started
            on_dataset_finished()

            for thread in threads:
                thread.join()

        thread = threading.Thread(
            target=process_all_dataset_entries, args=(dataset_id,)
        )
        thread.start()

    def run(self, timeout_in_minutes: Optional[int] = 10) -> Optional[RunResult]:
        """
        Run the test

        Args:
            timeout_in_minutes (Optional[int]): The timeout in minutes. Defaults to 10.

        Returns:
            RunResult: The result of the test run
        """
        try:
            errors: list[str] = []
            self._config.logger.info(message="Validating test run config...")
            if self._config.name == "":
                errors.append("Name is required to run a test.")
            if self._config.in_workspace_id == "":
                errors.append("Workspace id is required to run a test.")
            if (
                self._config.output_function is None
                and self._config.workflow is None
                and self._config.prompt_version is None
                and self._config.prompt_chain_version is None
                and self._config.output_function_with_tracing is None
            ):
                errors.append(
                    "One of output function (by calling yields_output) or output function with tracing (by calling yields_output_with_tracing) or workflow id (by calling with_workflow_id) or prompt version id (by calling with_prompt_version_id) or prompt chain version id (by calling with_prompt_chain_version_id) is required to run a test."
                )
            if self._config.data is None:
                errors.append("Dataset id is required to run a test.")
            if self._config.simulation_config is not None:
                if self._config.output_function_with_tracing is not None:
                    errors.append(
                        "Simulation config cannot currently be used with yieldsOutputWithTracing. Use withWorkflowId or withPromptVersionId instead."
                    )
                if self._config.output_function is not None:
                    if self._config.prompt_chain_version is not None:
                        errors.append(
                            "Simulation config with yields_output cannot use with_prompt_chain_version_id. Use with_prompt_version_id or with_workflow_id, or omit both for SDK-only simulation."
                        )
                    if self._config.prompt_version and self._config.workflow:
                        errors.append(
                            "Simulation config with yields_output cannot use both with_prompt_version_id and with_workflow_id. Set at most one (or neither for SDK-only simulation)."
                        )
                else:
                    if self._config.prompt_chain_version is not None:
                        errors.append(
                            "Simulation config cannot be used with withPromptChainVersionId. Use withWorkflowId or withPromptVersionId instead."
                        )
                    if not self._config.workflow and not self._config.prompt_version:
                        errors.append(
                            "Simulation config requires either withWorkflowId or withPromptVersionId to be set."
                        )
                if (
                    self._config.simulation_config.response_fields
                    and len(self._config.simulation_config.response_fields) > 0
                    and not self._config.workflow
                ):
                    errors.append(
                        "responseFields in simulationConfig can only be used with withWorkflowId, not with withPromptVersionId."
                    )
            if len(errors) > 0:
                raise ValueError(
                    "Missing required configuration for test\n" + "\n".join(errors)
                )
            self._config.logger.info(message="Sanitizing data...")
            sanitize_data_structure(self._config.data_structure)
            if isinstance(self._config.data, List):
                if self._config.data_structure:
                    sanitize_data(self._config.data, self._config.data_structure)
            self._config.logger.info(message="Sanitizing evaluators...")
            sanitize_evaluators(self._config.evaluators)
            evaluator_configs: List[Evaluator] = []
            evaluator_name_to_id_and_pass_fail_criteria_map = (
                get_local_evaluator_name_to_id_and_pass_fail_criteria_map(
                    self._config.evaluators
                )
            )
            for evaluator in self._config.evaluators or []:
                if isinstance(evaluator, str):
                    try:
                        self._config.logger.info(
                            message=f"Verifying if {evaluator} is added to the workspace.."
                        )
                        evaluator_config = self._maxim_apis.fetch_platform_evaluator(
                            name=evaluator, in_workspace_id=self._config.in_workspace_id
                        )
                        evaluator_configs.append(evaluator_config)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to fetch evaluator {evaluator}"
                        ) from e
                elif isinstance(evaluator, PlatformEvaluator):
                    # PlatformEvaluator - fetch from platform API like string evaluators
                    try:
                        self._config.logger.info(
                            message=f"Verifying if {evaluator.name} is added to the workspace.."
                        )
                        evaluator_config = self._maxim_apis.fetch_platform_evaluator(
                            name=evaluator.name, in_workspace_id=self._config.in_workspace_id
                        )
                        evaluator_configs.append(evaluator_config)
                        # Add to the name->id map if not already present
                        if evaluator.name not in evaluator_name_to_id_and_pass_fail_criteria_map:
                            evaluator_name_to_id_and_pass_fail_criteria_map[evaluator.name] = EvaluatorNameToIdAndPassFailCriteria(
                                id=evaluator_config.id,
                                pass_fail_criteria=None
                            )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to fetch evaluator {evaluator.name}"
                        ) from e
                elif isinstance(evaluator, BaseEvaluator):
                    for name in evaluator.names:
                        evaluator_config = get_evaluator_config_from_evaluator_name_and_pass_fail_criteria(
                            id=evaluator_name_to_id_and_pass_fail_criteria_map[name].id,
                            name=name,
                            pass_fail_criteria=evaluator_name_to_id_and_pass_fail_criteria_map[
                                name
                            ].pass_fail_criteria,
                        )
                        evaluator_configs.append(evaluator_config)


            if any(
                evaluator.type.value == EvaluatorType.HUMAN.value
                for evaluator in evaluator_configs
            ):
                if self._config.human_evaluation_config is None:
                    raise ValueError(
                        "Human evaluator found in evaluators, but no human evaluation config was provided."
                    )

            name = self._config.name
            data = self._config.data
            workspace_id = self._config.in_workspace_id
            human_evaluation_config = self._config.human_evaluation_config
            failed_entry_indices = []
            all_entries_processed = threading.Event()

            def mark_all_entries_processed() -> None:
                nonlocal all_entries_processed
                all_entries_processed.set()

            try:
                self._config.logger.info(f"Creating test run: {name}")
                requires_local_run = False
                if (
                    any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
                    or self._config.output_function is not None
                ):
                    requires_local_run = True

                # When workflow + simulation_config, default response_fields to ["response"] if not set
                simulation_config_to_send = self._config.simulation_config
                if (
                    self._config.workflow is not None
                    and simulation_config_to_send is not None
                    and (not simulation_config_to_send.response_fields or len(simulation_config_to_send.response_fields) == 0)
                ):
                    simulation_config_to_send = SimulationConfig(
                        max_turns=simulation_config_to_send.max_turns,
                        tools=simulation_config_to_send.tools,
                        context=simulation_config_to_send.context,
                        response_fields=["response"],
                        environment_id=simulation_config_to_send.environment_id,
                        stop_trigger=simulation_config_to_send.stop_trigger,
                        additional_instructions=simulation_config_to_send.additional_instructions,
                    )

                test_run = self._maxim_apis.create_test_run(
                    name=name,
                    workspace_id=workspace_id,
                    run_type=RunType.SINGLE,
                    workflow_id=(
                        self._config.workflow.id
                        if self._config.workflow is not None
                        else None
                    ),
                    prompt_version_id=(
                        self._config.prompt_version.id
                        if self._config.prompt_version is not None
                        else None
                    ),
                    prompt_chain_version_id=(
                        self._config.prompt_chain_version.id
                        if self._config.prompt_chain_version is not None
                        else None
                    ),
                    evaluator_config=evaluator_configs,
                    human_evaluation_config=human_evaluation_config or None,
                    requires_local_run=requires_local_run,
                    tags=(
                        list(self._config.tags or [])
                        + [f"repoId:{self._config.internal_maxim_logger.id}"]
                        if self._config.internal_maxim_logger
                        else self._config.tags or None
                    ),
                    simulation_config=simulation_config_to_send,
                    connected_repo_id=self._config.internal_maxim_logger.id if self._config.internal_maxim_logger else None,
                )
                if self._config.environment_name is not None:
                    test_run.environment_name = self._config.environment_name
                try:
                    if data is not None:
                        if isinstance(data, str):
                            self._run_test_with_dataset_id_or_split_id(
                                test_run=test_run,
                                dataset_id=data,
                                on_entry_failed=failed_entry_indices.append,
                                on_dataset_finished=mark_all_entries_processed,
                                evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                            )
                        elif isinstance(data, list):
                            self._run_test_with_local_data(
                                test_run,
                                lambda index: (
                                    data[index] if index < len(data) else None
                                ),
                                failed_entry_indices.append,
                                mark_all_entries_processed,
                                evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                            )
                        elif isinstance(data, Callable):
                            self._run_test_with_local_data(
                                test_run,
                                data,
                                failed_entry_indices.append,
                                mark_all_entries_processed,
                                evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                            )
                        else:
                            raise ValueError("Invalid data")

                    self._maxim_apis.mark_test_run_processed(test_run.id)

                    self._config.logger.info(
                        f"You can view your test run here: {self._config.base_url}/workspace/{self._config.in_workspace_id}/testrun/{test_run.id}"
                    )
                    self._config.logger.info(
                        "You can safely quit this session or wait to see the final output in console."
                    )
                except Exception as e:
                    self._maxim_apis.mark_test_run_failed(test_run.id)
                    raise e
                poll_count = 0
                polling_interval = calculate_polling_interval(
                    timeout_in_minutes or 10,
                    is_ai_evaluator_in_use=any(
                        e.type == "AI" for e in evaluator_configs
                    )
                    or False,
                )
                max_iterations = math.ceil(
                    (round(timeout_in_minutes or 10) * 60) / polling_interval
                )
                # Here we will check if we failed to push all entries

                self._config.logger.info("Waiting for test run to complete...")
                self._config.logger.info(
                    f"Polling interval: {polling_interval} seconds"
                )
                status: Optional[TestRunStatus] = None
                sync_check_count = 0
                while True:
                    sync_check_count += 1
                    status = self._maxim_apis.get_test_run_status(test_run.id)
                    if (
                        status is not None
                        and sync_check_count > 5
                        and status.total_entries == 0
                    ):
                        self._config.logger.info(
                            "No entries were pushed to the test run. Exiting..."
                        )
                        break
                    status_dict = status.to_dict()
                    status_line = " | ".join(
                        f"{key}: {value}"
                        for key, value in status_dict.items()
                        if key != "testRunStatus"
                    )
                    box_width = max(50, len(status_line) + 4)
                    header_width = len(
                        f" Test run status: {status.test_run_status.value} "
                    )
                    box_width = max(box_width, header_width + 4)

                    header = (
                        f" Test run status: {status.test_run_status.value} ".center(
                            box_width
                        )
                    )
                    self._config.logger.info("┌" + "─" * box_width + "┐")
                    self._config.logger.info(f"│{header}│")
                    self._config.logger.info("├" + "─" * box_width + "┤")

                    status_line = " | ".join(
                        f"{key}: {value}"
                        for key, value in status_dict.items()
                        if key != "testRunStatus"
                    )
                    self._config.logger.info(f"│ {status_line:<{box_width - 2}} │")
                    self._config.logger.info("└" + "─" * box_width + "┘\n")
                    if poll_count > max_iterations:
                        raise Exception(
                            f"Test run is taking over timeout period ({round(timeout_in_minutes or 10)} minutes) to complete, please check the report on our web portal directly: {self._config.base_url}/workspace/{self._config.in_workspace_id}/testrun/{test_run.id}"
                        )

                    # Test run is failed - we break the loop
                    if (
                        status.test_run_status.value == RunStatus.FAILED.value
                        or status.test_run_status.value == RunStatus.STOPPED.value
                    ):
                        break

                    if (
                        status.test_run_status.value == RunStatus.COMPLETE.value
                        and all_entries_processed.is_set()
                    ):
                        # We will check if we sent all the entries
                        if status.total_entries != 0 and (
                            status.total_entries
                            == status.completed_entries
                            + status.failed_entries
                            + status.stopped_entries
                        ):
                            self._config.logger.info(
                                "All entries processed. Test run completed."
                            )
                            break
                    # Polling again
                    time.sleep(polling_interval)
                    poll_count += 1

                if status.test_run_status.value == RunStatus.FAILED:
                    raise Exception(
                        f"Test run failed, please check the report on our web portal: {self._config.base_url}/workspace/{self._config.in_workspace_id}/testrun/{test_run.id}"
                    )

                if status.test_run_status.value == RunStatus.STOPPED:
                    raise Exception(
                        f"Test run was stopped, please check the report on our web portal: {self._config.base_url}/workspace/{self._config.in_workspace_id}/testrun/{test_run.id}"
                    )

                test_run_result = self._maxim_apis.get_test_run_final_result(
                    test_run.id
                )
                test_run_result.link = self._config.base_url + test_run_result.link
                self._config.logger.info(
                    f'Test run "{name}" completed successfully!🎉 \nView the report here: {test_run_result.link}'
                )
                return RunResult(
                    test_run_result=test_run_result,
                    failed_entry_indices=failed_entry_indices,
                )

            except Exception as e:
                self._config.logger.error("\n\n💥 Error while running test: ", e)
                raise

        except Exception as e:
            self._config.logger.error("\n\n💥 Error while running test: ", e)
            raise
