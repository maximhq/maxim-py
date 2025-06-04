import json
import logging
import os
import unittest
from typing import Dict, Optional

from maxim import Config, Maxim
from maxim.evaluators import BaseEvaluator
from maxim.models import (
    Data,
    LocalData,
    LocalEvaluatorResultParameter,
    LocalEvaluatorReturn,
    PassFailCriteria,
    TestRunLogger,
    YieldedOutput,
)
from maxim.models.evaluator import (
    PassFailCriteriaForTestrunOverall,
    PassFailCriteriaOnEachEntry,
)

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "dev"

apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
workspaceId = data[env]["workspaceId"]
datasetId = data[env]["datasetId"]
workflowId = data[env]["workflowId"]


class TestTestRuns(unittest.TestCase):
    def setUp(self):
        config = Config(
            api_key=apiKey, base_url=baseUrl, debug=True, raise_exceptions=True
        )
        self.maxim = Maxim(config)

    def test_create_test_run(self):
        def processor(data) -> YieldedOutput:
            return YieldedOutput(data="test")

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}")

        self.maxim.create_test_run(
            name="test run", in_workspace_id=workspaceId
        ).with_data(datasetId).with_concurrency(2).with_evaluators("Bias").with_logger(
            Logger()
        ).yields_output(processor).run()

    def test_run_with_multiple_evaluators(self):
        def processor(data) -> YieldedOutput:
            return YieldedOutput(data="test")

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}")

        self.maxim.create_test_run(
            name="test run", in_workspace_id=workspaceId
        ).with_data(datasetId).with_concurrency(2).with_evaluators(
            "Clarity", "Conciseness", "Faithfulness", "Bias"
        ).with_logger(Logger()).yields_output(processor).run()

    def test_create_test_with_workflow_id(self) -> None:
        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}")

        self.maxim.create_test_run(
            name="workflow-run-from-sdk", in_workspace_id=workspaceId
        ).with_workflow_id(workflowId).with_concurrency(2).with_logger(
            Logger()
        ).with_data(datasetId).with_evaluators("Bias").run()

    def test_create_test_run_dict(self):
        def processor(data: Data) -> YieldedOutput:
            return YieldedOutput(data="test")

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}")

        self.maxim.create_test_run(
            name="test run", in_workspace_id=workspaceId
        ).with_data(datasetId).with_concurrency(2).with_evaluators("Bias").with_logger(
            Logger()
        ).yields_output(processor).run()

    def test_create_test_run_with_local_data_and_workflow_id(self):
        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}", e)

        data: Data = [
            {
                "input": "test",
            },
            {
                "input": "example",
            },
            {
                "input": "sample",
            },
            {
                "input": "data",
            },
            {
                "input": "entry",
            },
        ]

        self.maxim.create_test_run(
            name="test run", in_workspace_id=workspaceId
        ).with_data_structure({"input": "INPUT"}).with_data(data).with_concurrency(
            2
        ).with_evaluators("Bias").with_logger(Logger()).with_workflow_id(
            workflowId
        ).run()

    def test_create_test_run_with_local_data_and_workflow_id_and_local_evaluators(self):
        def processor(data) -> YieldedOutput:
            return YieldedOutput(data="test")

        class MyCustomEvaluator(BaseEvaluator):
            def evaluate(
                self, result: LocalEvaluatorResultParameter, data: LocalData
            ) -> Dict[str, LocalEvaluatorReturn]:
                return {
                    "abc": LocalEvaluatorReturn(score=1),
                    "cde": LocalEvaluatorReturn(score=False, reasoning="Just chillll"),
                }

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}", e)

        data: Data = [
            {"input": "test"},
            {
                "input": "example",
            },
            {
                "input": "sample",
            },
            {
                "input": "data",
            },
            {
                "input": "entry",
            },
        ]

        self.maxim.create_test_run(
            name="test run", in_workspace_id=workspaceId
        ).with_data_structure({"input": "INPUT"}).with_data(data).with_concurrency(
            2
        ).with_evaluators(
            "Bias",
            MyCustomEvaluator(
                pass_fail_criteria={
                    "abc": PassFailCriteria(
                        for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                            ">", 3, "average"
                        ),
                        on_each_entry_pass_if=PassFailCriteriaOnEachEntry(">", 1),
                    ),
                    "cde": PassFailCriteria(
                        for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                            overall_should_be="!=", value=2, for_result="average"
                        ),
                        on_each_entry_pass_if=PassFailCriteriaOnEachEntry("=", True),
                    ),
                }
            ),
        ).with_logger(Logger()).with_workflow_id(workflowId).run()

    def test_create_test_run_with_local_data(self):
        def processor(data) -> YieldedOutput:
            return YieldedOutput(data="test")

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}", e)

        data: Data = [
            {
                "input": "test",
            },
            {
                "input": "example",
            },
            {
                "input": "sample",
            },
            {
                "input": "data",
            },
            {
                "input": "entry",
            },
        ]
        result = (
            self.maxim.create_test_run(name="test run", in_workspace_id=workspaceId)
            .with_data_structure({"input": "INPUT"})
            .with_data(data)
            .with_concurrency(2)
            .with_evaluators("Bias")
            .with_logger(Logger())
            .yields_output(processor)
            .run()
        )
        self.assertIsNotNone(result)
        if result is not None:
            self.assertListEqual(result.failed_entry_indices, [])

    def tearDown(self):
        pass
