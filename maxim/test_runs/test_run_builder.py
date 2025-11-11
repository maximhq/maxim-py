import asyncio
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Union, final

from ..apis import MaximAPI
from ..dataset import sanitize_data_structure
from ..evaluators import BaseEvaluator
from ..models import (
    DatasetRow,
    Evaluator,
    EvaluatorType,
    HumanEvaluationConfig,
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
)
from ..models.test_run import (
    PromptChainVersionConfig,
    PromptVersionConfig,
    WorkflowConfig,
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
        evaluators: List[Union[str, BaseEvaluator]],
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

    def __process_entry(
        self,
        index: int,
        input: Optional[str],
        expected_output: Optional[str],
        context_to_evaluate: Optional[Union[str, List[str]]],
        output_function: Optional[
            Callable[[LocalData], Union[YieldedOutput, Awaitable[YieldedOutput]]]
        ],
        get_row: Callable[[int], Optional[LocalData]],
        logger: TestRunLogger,
        evaluator_name_to_id_and_pass_fail_criteria_map: Dict[
            str, EvaluatorNameToIdAndPassFailCriteria
        ],
    ) -> ProcessedEntry:
        """
        Process a single test run entry

        Args:
            index (int): The index of the entry
            keys (dict[str, Optional[str]]): Mapping of column names to keys in the data
            output_function (Callable[[dict[str, Any]], YieldedOutput]): Function to generate output
            get_row (Callable[[int], Optional[dict[str, Any]]]): Function to retrieve a row from the dataset
            logger (TestRunLogger): Logger instance

        Returns:
            ProcessedEntry: Contains processed entry and metadata
        """
        row = get_row(index)
        if row is None:
            raise ValueError(f"Dataset entry {index} is missing")

        output: Optional[Union[Awaitable[YieldedOutput], YieldedOutput]]
        if output_function is not None:
            output = output_function(row)
        elif self._config.workflow is not None:
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

        if yielded_output is not None:
            if (
                yielded_output.retrieved_context_to_evaluate is not None
                and context_to_evaluate is not None
            ):
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
                )
            )

        return ProcessedEntry(
            entry=TestRunEntry(
                output=yielded_output.data if yielded_output is not None else None,
                input=input,
                expected_output=expected_output,
                context_to_evaluate=context_to_evaluate,
                variables=(
                    get_variables_from_row(row, self._config.data_structure)
                    if self._config.data_structure
                    else {}
                ),
                local_evaluation_results=local_evaluation_results_with_ids,
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
        self, *evaluators: Union[str, BaseEvaluator]
    ) -> "TestRunBuilder[T]":
        """
        Add evaluators to the test run

        Args:
            *evaluators (str): The evaluators to add

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining
        """
        evaluators_list: List[Union[BaseEvaluator, str]] = []
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

    def yields_output(
        self,
        output_function: Callable[
            [LocalData], Union[YieldedOutput, Awaitable[YieldedOutput]]
        ],
    ) -> "TestRunBuilder[T]":
        """
        Set the output function for the test run

        Args:
            output_function (Callable[[T], Union[YieldedOutput, Awaitable[YieldedOutput]]]): The output function to use

        Returns:
            TestRunBuilder[T]: The current TestRunBuilder instance for method chaining

        Raises:
            ValueError: If a workflow ID, prompt chain version ID or prompt version ID is already set for this run builder
        """
        if self._config.workflow is not None:
            raise ValueError(
                "Workflow id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.prompt_chain_version is not None:
            raise ValueError(
                "Prompt chain version id is already set for this run builder. You can use either one of with_prompt_version_id, with_prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        if self._config.prompt_version is not None:
            raise ValueError(
                "Prompt version id is already set for this run builder. You can use either one of with_prompt_version_id, prompt_chain_version_id, with_workflow_id or yields_output in a test run."
            )
        self._config.output_function = output_function
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
                input, expected_output, context_to_evaluate = (
                    get_input_expected_output_and_context_from_row(
                        input_key, expectedOutputKey, contextToEvaluateKey, row
                    )
                )
                if (
                    any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
                    or self._config.output_function is not None
                ):
                    result = self.__process_entry(
                        index=index,
                        input=input,
                        expected_output=expected_output,
                        context_to_evaluate=context_to_evaluate,
                        output_function=self._config.output_function,
                        get_row=lambda index: row,
                        logger=self._config.logger,
                        evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                    )
                    # pushing this entry to test run
                    self._maxim_apis.push_test_run_entry(
                        test_run=test_run,
                        entry=result.entry,
                        run_config=(
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
                        ),
                    )
                else:
                    # pushing directly
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
                        ),
                    )
            except Exception as e:
                self._config.logger.error(
                    f"Error while running data entry at index [{index}]: {str(e)}"
                )
                on_entry_failed(index)

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
                        continue
                    index += 1
                    thread = threading.Thread(
                        target=process_row,
                        args=(
                            index,
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
                finally:
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
                input, expected_output, context_to_evaluate = (
                    get_input_expected_output_and_context_from_row(
                        input_key, expectedOutputKey, contextToEvaluateKey, row_data
                    )
                )

                if (
                    any(isinstance(e, BaseEvaluator) for e in self._config.evaluators)
                    or self._config.output_function is not None
                ):
                    # processing the entry
                    result = self.__process_entry(
                        index=index,
                        input=input,
                        expected_output=expected_output,
                        context_to_evaluate=context_to_evaluate,
                        output_function=self._config.output_function,
                        get_row=lambda index: row.to_dict()["data"],
                        logger=self._config.logger,
                        evaluator_name_to_id_and_pass_fail_criteria_map=evaluator_name_to_id_and_pass_fail_criteria_map,
                    )
                    # pushing this entry to test run
                    self._maxim_apis.push_test_run_entry(
                        test_run=TestRunWithDatasetEntry(
                            test_run=test_run,
                            dataset_id=dataset_id,
                            dataset_entry_id=row.id,
                        ),
                        entry=result.entry,
                        run_config=(
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
                        ),
                    )
                else:
                    # pushing directly
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
                        ),
                    )
            except Exception as e:
                self._config.logger.error(
                    f"Error while running data entry at index [{index}]: {str(e)}",
                )
                on_entry_failed(index)
                raise e

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
                finally:
                    semaphore.release()
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
            ):
                errors.append(
                    "One of output function (by calling yields_output) or workflow id (by calling with_workflow_id) or prompt version id (by calling with_prompt_version_id) or prompt chain version id (by calling with_prompt_chain_version_id) is required to run a test."
                )
            if self._config.data is None:
                errors.append("Dataset id is required to run a test.")
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
                else:
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
                    tags=self._config.tags or None,
                )
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

        except Exception as e:
            self._config.logger.error("\n\n💥 Error while running test: ", e)
