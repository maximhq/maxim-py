import os
import time
import unittest
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from maxim import Config, Maxim
from maxim.evaluators import BaseEvaluator
from maxim.models import (
    Data,
    LocalData,
    LocalEvaluatorResultParameter,
    LocalEvaluatorReturn,
    PassFailCriteria,
    SimulationConfig,
    TestRunLogger,
    YieldedOutput,
)
from maxim.models.evaluator import (
    PassFailCriteriaForTestrunOverall,
    PassFailCriteriaOnEachEntry,
)

load_dotenv()

# Load config from environment variables
api_key: str = os.environ.get("MAXIM_API_KEY", "")
if not api_key:
    raise ValueError("Missing MAXIM_API_KEY environment variable")
workspace_id: str = os.environ.get("MAXIM_WORKSPACE_ID", "")
if not workspace_id:
    raise ValueError("Missing MAXIM_WORKSPACE_ID environment variable")
prompt_version_id: str = os.environ.get("MAXIM_PROMPT_VERSION_ID", "")
if not prompt_version_id:
    raise ValueError("Missing MAXIM_PROMPT_VERSION_ID environment variable")
dataset_id: str = os.environ.get("MAXIM_DATASET_ID", "")
if not dataset_id:
    raise ValueError("Missing MAXIM_DATASET_ID environment variable")
workflow_id: str = os.environ.get("MAXIM_WORKFLOW_ID", "")
if not workflow_id:
    raise ValueError("Missing MAXIM_WORKFLOW_ID environment variable")
base_url: str = os.environ.get("MAXIM_BASE_URL", "https://app.getmaxim.ai")

# Data structure
data_structure = {
    "Input": "INPUT",
    "Expected Steps": "EXPECTED_STEPS",
    "Scenario": "SCENARIO",
}

# Data structure with a file column (FILE_URL_VARIABLE = file URL string per row)
data_structure_with_file = {
    "Input": "INPUT",
    "Expected Steps": "EXPECTED_STEPS",
    "Scenario": "SCENARIO",
    "Document": "FILE_URL_VARIABLE",
}
# Manual test data (matches JS SDK)
manual_data: Data = [
    {
        "Input": "What is the significance of the 'Pale Blue Dot' in 'Cosmos'?",
        "Expected Steps": """1.The "Pale Blue Dot" refers to an image of Earth taken by the Voyager 1 spacecraft from a distance of about 3.7 billion miles. In "Cosmos," Carl Sagan reflects on the image to illustrate the fragility and insignificance of Earth in the vastness of the universe, emphasizing the need for humility and unity among humanity.
\t\t\n2. Yes. Sagan connects the "Pale Blue Dot" idea to humanity's tendency to think of itself as central or unique. By showing how tiny Earth is, he challenges that view and instead presents humans as part of a much larger cosmic story, made from the same matter as stars and governed by the same natural laws.""",
        "Scenario": "Question about Pale Blue Dot",
    },
    {
        "Input": "Explain quantum entanglement in simple terms.",
        "Expected Steps": """1. Quantum entanglement is a phenomenon where two or more particles become connected in such a way that the state of one particle instantly affects the state of another, regardless of the distance between them.
\t\t\n2. This connection happens instantaneously, faster than light, which challenges our classical understanding of physics.""",
        "Scenario": "Science explanation request",
    },
]

# Manual test data with a file column (FILE_URL_VARIABLE = file URL string per row)
manual_data_with_files: Data = [
    {
        "Input": "Summarize the content of the linked document.",
        "Expected Steps": "1. Read the document.\n2. Provide a brief summary.",
        "Scenario": "Document summary with file",
        "Document": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    },
    {
        "Input": "What does the linked file contain?",
        "Expected Steps": "1. Access the file.\n2. Describe its contents.",
        "Scenario": "File content query",
        "Document": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    },
]


def _standard_pass_fail_criteria() -> Dict[str, Any]:
    return {
        "on_each_entry": PassFailCriteriaOnEachEntry(score_should_be=">=", value=1),
        "for_testrun_overall": PassFailCriteriaForTestrunOverall(
            overall_should_be=">=",
            value=100,
            for_result="percentageOfPassedResults",
        ),
    }


# ----- Local evaluators -----

# Single local evaluator (output-length-validator)
class LocalSingleEvaluator(BaseEvaluator):
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        length_score = (
            1 if len(result.output) >= len(data.get("Expected Steps", "")) else 0
        )
        return {
            "output-length-validator": LocalEvaluatorReturn(
                score=length_score,
                reasoning=(
                    "Output length is sufficient"
                    if length_score == 1
                    else "Output length is insufficient"
                ),
            ),
        }


def local_single_evaluator() -> LocalSingleEvaluator:
    s = _standard_pass_fail_criteria()
    return LocalSingleEvaluator(
        pass_fail_criteria={
            "output-length-validator": PassFailCriteria(
                on_each_entry_pass_if=s["on_each_entry"],
                for_testrun_overall_pass_if=s["for_testrun_overall"],
            ),
        }
    )


# Simulation outputs evaluator (simulation-steps-validator)
class SimulationOutputsEvaluator(BaseEvaluator):
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        if not result.simulation_outputs or len(result.simulation_outputs) == 0:
            return {
                "simulation-steps-validator": LocalEvaluatorReturn(
                    score=0, reasoning="No simulation outputs available"
                ),
            }
        expected_lines = [
            line for line in data.get("Expected Steps", "").split("\n")
            if line.strip()
        ]
        expected_steps_count = len(expected_lines)
        actual_steps_count = len(result.simulation_outputs)
        steps_match = actual_steps_count >= expected_steps_count
        return {
            "simulation-steps-validator": LocalEvaluatorReturn(
                score=1 if steps_match else 0,
                reasoning=(
                    f"Simulation produced {actual_steps_count} steps, meeting expected {expected_steps_count} steps"
                    if steps_match
                    else f"Simulation produced {actual_steps_count} steps, but expected at least {expected_steps_count} steps"
                ),
            ),
        }


def simulation_outputs_evaluator() -> SimulationOutputsEvaluator:
    s = _standard_pass_fail_criteria()
    return SimulationOutputsEvaluator(
        pass_fail_criteria={
            "simulation-steps-validator": PassFailCriteria(
                on_each_entry_pass_if=s["on_each_entry"],
                for_testrun_overall_pass_if=s["for_testrun_overall"],
            ),
        }
    )


# Combined local evaluator (length-check + contains-input)
class LocalCombinedEvaluator(BaseEvaluator):
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        output = result.output
        expected_steps = data.get("Expected Steps", "")
        input_text = data.get("Input", "")
        length_score = 1 if len(output) >= len(expected_steps) else 0
        contains_score = 1 if input_text in output else 0
        return {
            "length-check": LocalEvaluatorReturn(
                score=length_score, reasoning="Output length is sufficient"
            ),
            "contains-input": LocalEvaluatorReturn(
                score=contains_score, reasoning="Output contains input"
            ),
        }


def local_combined_evaluator() -> LocalCombinedEvaluator:
    s = _standard_pass_fail_criteria()
    return LocalCombinedEvaluator(
        pass_fail_criteria={
            "length-check": PassFailCriteria(
                on_each_entry_pass_if=s["on_each_entry"],
                for_testrun_overall_pass_if=s["for_testrun_overall"],
            ),
            "contains-input": PassFailCriteria(
                on_each_entry_pass_if=s["on_each_entry"],
                for_testrun_overall_pass_if=s["for_testrun_overall"],
            ),
        }
    )


# Boolean evaluator (has-question-mark)
class LocalBooleanEvaluator(BaseEvaluator):
    def evaluate(
        self, result: LocalEvaluatorResultParameter, data: LocalData
    ) -> Dict[str, LocalEvaluatorReturn]:
        has_q = "?" in result.output
        return {
            "has-question-mark": LocalEvaluatorReturn(
                score=has_q,
                reasoning=(
                    "Contains question mark" if has_q else "Missing question mark"
                ),
            ),
        }


def local_boolean_evaluator() -> LocalBooleanEvaluator:
    return LocalBooleanEvaluator(
        pass_fail_criteria={
            "has-question-mark": PassFailCriteria(
                on_each_entry_pass_if=PassFailCriteriaOnEachEntry(
                    score_should_be="=", value=True
                ),
                for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                    overall_should_be=">=",
                    value=80,
                    for_result="percentageOfPassedResults",
                ),
            ),
        }
    )


# ----- Custom logger -----


class TestLogger(TestRunLogger):
    def __init__(self, test_case: str):
        self.test_case = test_case

    def error(self, message: str, e: Optional[Exception] = None) -> None:
        out = f"[{self.test_case}][ERROR] {message}"
        if e is not None:
            out += f" {e!s}"
        print(out)

    def info(self, message: str) -> None:
        print(f"[{self.test_case}][INFO] {message}")

    def processed(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        print(f"[{self.test_case}][PROCESSED] {message}")


def _base_simulation_config() -> SimulationConfig:
    return SimulationConfig(max_turns=3)


# ----- Test suite -----


class TestSimulation(unittest.TestCase):
    def setUp(self) -> None:
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        config_obj = Config(
            api_key=api_key, base_url=base_url, debug=True, raise_exceptions=True
        )
        self.maxim = Maxim(config_obj)

    def tearDown(self) -> None:
        if hasattr(self, "maxim"):
            self.maxim.cleanup()
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

    # ----- Valid Combinations - Prompt Version -----

    def test_prompt_dataset_local_single(self) -> None:
        test_case = "prompt-dataset-local-single"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_local_combined(self) -> None:
        test_case = "prompt-dataset-local-combined"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_combined_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_maxim_eval(self) -> None:
        test_case = "prompt-dataset-maxim-eval"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators("Faithfulness")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_local_maxim(self) -> None:
        test_case = "prompt-dataset-local-maxim"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_single_evaluator(), "Faithfulness")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_manual_with_files(self) -> None:
        """Manual dataset with a file column (FILE_URL_VARIABLE); each row includes a document URL."""
        test_case = "prompt-manual-with-files"
        logger = TestLogger(test_case)
        logger.info("Starting test, Manual data with files (file URL column)")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure_with_file)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(manual_data_with_files)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_manual_local_single(self) -> None:
        test_case = "prompt-manual-local-single"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(manual_data)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_manual_local_combined(self) -> None:
        test_case = "prompt-manual-local-combined"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(manual_data)
            .with_evaluators(local_combined_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_manual_multiple_local(self) -> None:
        test_case = "prompt-manual-multiple-local"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(manual_data)
            .with_evaluators(local_single_evaluator(), local_boolean_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_multiple_maxim(self) -> None:
        test_case = "prompt-dataset-multiple-maxim"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators("Faithfulness", "Consistency")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_local_combined_maxim(self) -> None:
        test_case = "prompt-dataset-local-combined-maxim"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Prompt Version")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_combined_evaluator(), "Faithfulness")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    # ----- Valid Combinations - Workflow -----

    def test_workflow_dataset_local_single(self) -> None:
        test_case = "workflow-dataset-local-single"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Workflow")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data(dataset_id)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_dataset_local_combined(self) -> None:
        test_case = "workflow-dataset-local-combined"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Workflow")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data(dataset_id)
            .with_evaluators(local_combined_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_dataset_maxim_eval(self) -> None:
        test_case = "workflow-dataset-maxim-eval"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Workflow")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data(dataset_id)
            .with_evaluators("Faithfulness")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_manual_local_single(self) -> None:
        test_case = "workflow-manual-local-single"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Workflow")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data(manual_data)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_manual_local_maxim(self) -> None:
        test_case = "workflow-manual-local-maxim"
        logger = TestLogger(test_case)
        logger.info("Starting test, Valid Combinations - Workflow")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data(manual_data)
            .with_evaluators(local_single_evaluator(), "Faithfulness")
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    # ----- Simulation Config Variations -----

    def test_prompt_dataset_custom_sim_config(self) -> None:
        test_case = "prompt-dataset-custom-sim-config"
        logger = TestLogger(test_case)
        logger.info("Starting test, Simulation Config Variations")
        sim_config = SimulationConfig(
            max_turns=5,
            scenario="Test scenario",
            persona="Helpful assistant",
        )
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(sim_config)
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_dataset_minimal_sim_config(self) -> None:
        test_case = "workflow-dataset-minimal-sim-config"
        logger = TestLogger(test_case)
        logger.info("Starting test, Simulation Config Variations")
        sim_config = SimulationConfig(max_turns=2)
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(sim_config)
            .with_workflow_id(workflow_id)
            .with_data(dataset_id)
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    # ----- Invalid Combinations - Error Handling -----

    def test_invalid_no_prompt_or_workflow(self) -> None:
        test_case = "invalid-no-prompt-or-workflow"
        logger = TestLogger(test_case)
        logger.info("Starting test, Invalid Combinations")
        with self.assertRaises(Exception):
            (
                self.maxim.create_test_run(
                    f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                    workspace_id,
                )
                .with_data_structure(data_structure)
                .with_simulation_config(_base_simulation_config())
                .with_data(dataset_id)
                .with_evaluators(local_single_evaluator())
                .with_logger(logger)
                .run(3)
            )
        print(f"✅ {test_case}: Correctly rejected")

    def test_invalid_sim_with_yields_output(self) -> None:
        test_case = "invalid-sim-with-yields-output"
        logger = TestLogger(test_case)

        def yields_fn(_data: LocalData) -> YieldedOutput:
            return YieldedOutput(data="test")

        logger.info("Starting test, Invalid Combinations")
        with self.assertRaises(Exception):
            (
                self.maxim.create_test_run(
                    f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                    workspace_id,
                )
                .with_data_structure(data_structure)
                .with_simulation_config(_base_simulation_config())
                .with_data(manual_data)
                .with_evaluators(local_single_evaluator())
                .with_logger(logger)
                .yields_output(yields_fn)
                .run(3)
            )
        print(f"✅ {test_case}: Correctly rejected")

    def test_invalid_no_data(self) -> None:
        test_case = "invalid-no-data"
        logger = TestLogger(test_case)
        logger.info("Starting test, Invalid Combinations")
        with self.assertRaises(Exception):
            (
                self.maxim.create_test_run(
                    f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                    workspace_id,
                )
                .with_data_structure(data_structure)
                .with_simulation_config(_base_simulation_config())
                .with_prompt_version_id(prompt_version_id)
                .with_evaluators(local_single_evaluator())
                .with_logger(logger)
                .run(3)
            )
        print(f"✅ {test_case}: Correctly rejected")

    def test_invalid_both_prompt_workflow(self) -> None:
        test_case = "invalid-both-prompt-workflow"
        logger = TestLogger(test_case)
        logger.info("Starting test, Invalid Combinations")
        with self.assertRaises(ValueError):
            (
                self.maxim.create_test_run(
                    f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                    workspace_id,
                )
                .with_data_structure(data_structure)
                .with_simulation_config(_base_simulation_config())
                .with_prompt_version_id(prompt_version_id)
                .with_workflow_id(workflow_id)
                .with_data(dataset_id)
                .with_evaluators(local_single_evaluator())
                .with_logger(logger)
                .run(3)
            )
        print(f"✅ {test_case}: Correctly rejected")

    # ----- Edge Cases -----

    def test_prompt_dataset_boolean_eval(self) -> None:
        test_case = "prompt-dataset-boolean-eval"
        logger = TestLogger(test_case)
        logger.info("Starting test, Edge Cases")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(dataset_id)
            .with_evaluators(local_boolean_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_workflow_single_manual_entry(self) -> None:
        test_case = "workflow-single-manual-entry"
        logger = TestLogger(test_case)
        logger.info("Starting test, Edge Cases")
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_workflow_id(workflow_id)
            .with_data([manual_data[0]])
            .with_evaluators(local_single_evaluator())
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")

    def test_prompt_dataset_all_eval_types(self) -> None:
        test_case = "prompt-dataset-all-eval-types"
        logger = TestLogger(test_case)
        logger.info("Starting test, Edge Cases")
        # Use manual_data to avoid "Entry with same name already exists" from
        # platform dataset rows that may have duplicate Input/Scenario values.
        result = (
            self.maxim.create_test_run(
                f"SDK Test: {test_case} - {int(time.time() * 1000)}",
                workspace_id,
            )
            .with_data_structure(data_structure)
            .with_simulation_config(_base_simulation_config())
            .with_prompt_version_id(prompt_version_id)
            .with_data(manual_data)
            .with_evaluators(
                local_single_evaluator(),
                simulation_outputs_evaluator(),
            )
            .with_logger(logger)
            .run(3)
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.test_run_result)
        self.assertIsNotNone(result.test_run_result.link)
        print(f"✅ {test_case}: {result.test_run_result.link}")


if __name__ == "__main__":
    unittest.main()
