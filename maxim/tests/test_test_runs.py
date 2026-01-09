import json
import logging
import os
import unittest
from typing import Dict, Optional
import time

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
    YieldedOutputMeta,
    YieldedOutputCost,
    YieldedOutputTokenUsage,
)
from maxim.models.evaluator import (
    PassFailCriteriaForTestrunOverall,
    PassFailCriteriaOnEachEntry,
)

with open(str(f"{os.getcwd()}/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

logging.basicConfig(level=logging.INFO)
env = "prod"

apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
workspaceId = data[env]["workspaceId"]
datasetId = data[env]["datasetId"]
workflowId = data[env]["workflowId"]
promptVersionId = data[env]["promptVersionId"]
promptChainVersionId = data[env]["promptChainVersionId"]


class TestTestRuns(unittest.TestCase):
    def setUp(self):
        # Clear singleton instance if it exists
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

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
        ).yields_output(
            processor
        ).run()

    def test_run_with_multiple_evaluators(self):
        def processor(data) -> YieldedOutput:
            time.sleep(4)
            print(f"Processor called with data: {data}")
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
        ).yields_output(
            processor
        ).run()

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
        ).with_data(
            datasetId
        ).with_evaluators(
            "Bias"
        ).run()

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
        ).yields_output(
            processor
        ).run()

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
        ).with_evaluators(
            "Bias"
        ).with_logger(
            Logger()
        ).with_workflow_id(
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
        ).with_logger(
            Logger()
        ).with_workflow_id(
            workflowId
        ).run()

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

    def test_create_test_run_with_image_variables_and_prompt_workflow(self):
        """Test prompt workflow and chains with image variable columns and other variable types"""

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables and prompt workflow",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators("Bias", "Clarity")
            .with_workflow_id(workflowId)  # Test with prompt workflow
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_and_prompt_version(self):
        """Test prompt version with image variable columns and other variable types"""

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables and prompt version",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators("Bias", "Clarity")
            .with_prompt_version_id(promptVersionId)  # Test with prompt version
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_and_prompt_chain_version(self):
        """Test prompt chain version with image variable columns and other variable types"""

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables and prompt chain version",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators("Bias", "Clarity")
            .with_prompt_chain_version_id(
                promptChainVersionId
            )  # Test with prompt chain version
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_and_local_yields_output(self):
        """Test local yields_output function with image variable columns and other variable types"""

        def processor(data) -> YieldedOutput:
            # Process the input data and return a response
            input_text = data.get("input", "")
            context = data.get("context", "")
            return YieldedOutput(
                data=f"Processed: {input_text} with context: {context}"
            )

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}", e)

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables and local yields output",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators("Bias", "Clarity")
            .with_logger(Logger())
            .yields_output(processor)  # Test with local yields_output function
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_prompt_workflow_and_local_evaluators(
        self,
    ):
        """Test prompt workflow with image variable columns and local evaluators"""

        class MyCustomEvaluator(BaseEvaluator):
            def evaluate(
                self, result: LocalEvaluatorResultParameter, data: LocalData
            ) -> Dict[str, LocalEvaluatorReturn]:
                return {
                    "image_relevance": LocalEvaluatorReturn(
                        score=1, reasoning="Image is relevant to the input"
                    ),
                    "output_quality": LocalEvaluatorReturn(
                        score=True, reasoning="Output meets quality standards"
                    ),
                }

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables, prompt workflow and local evaluators",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators(
                "Bias",
                MyCustomEvaluator(
                    pass_fail_criteria={
                        "image_relevance": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                ">", 0.8, "average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(">", 0.5),
                        ),
                        "output_quality": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                overall_should_be="=", value=0.8, for_result="average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(
                                "=", True
                            ),
                        ),
                    }
                ),
            )
            .with_workflow_id(workflowId)  # Test with prompt workflow
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_prompt_version_and_local_evaluators(
        self,
    ):
        """Test prompt version with image variable columns and local evaluators"""

        class MyCustomEvaluator(BaseEvaluator):
            def evaluate(
                self, result: LocalEvaluatorResultParameter, data: LocalData
            ) -> Dict[str, LocalEvaluatorReturn]:
                return {
                    "image_relevance": LocalEvaluatorReturn(
                        score=1, reasoning="Image is relevant to the input"
                    ),
                    "output_quality": LocalEvaluatorReturn(
                        score=True, reasoning="Output meets quality standards"
                    ),
                }

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables, prompt version and local evaluators",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators(
                "Bias",
                MyCustomEvaluator(
                    pass_fail_criteria={
                        "image_relevance": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                ">", 0.8, "average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(">", 0.5),
                        ),
                        "output_quality": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                overall_should_be="=", value=0.8, for_result="average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(
                                "=", True
                            ),
                        ),
                    }
                ),
            )
            .with_prompt_version_id(promptVersionId)  # Test with prompt version
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_prompt_chain_version_and_local_evaluators(
        self,
    ):
        """Test prompt chain version with image variable columns and local evaluators"""

        class MyCustomEvaluator(BaseEvaluator):
            def evaluate(
                self, result: LocalEvaluatorResultParameter, data: LocalData
            ) -> Dict[str, LocalEvaluatorReturn]:
                return {
                    "image_relevance": LocalEvaluatorReturn(
                        score=1, reasoning="Image is relevant to the input"
                    ),
                    "output_quality": LocalEvaluatorReturn(
                        score=True, reasoning="Output meets quality standards"
                    ),
                }

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables, prompt chain version and local evaluators",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators(
                "Bias",
                MyCustomEvaluator(
                    pass_fail_criteria={
                        "image_relevance": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                ">", 0.8, "average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(">", 0.5),
                        ),
                        "output_quality": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                overall_should_be="=", value=0.8, for_result="average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(
                                "=", True
                            ),
                        ),
                    }
                ),
            )
            .with_prompt_chain_version_id(
                promptChainVersionId
            )  # Test with prompt chain version
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def test_create_test_run_with_image_variables_local_yields_output_and_local_evaluators(
        self,
    ):
        """Test local yields_output function with image variable columns and local evaluators"""

        def processor(data) -> YieldedOutput:
            # Process the input data and return a response
            input_text = data.get("input", "")
            context = data.get("context", "")
            return YieldedOutput(
                data=f"Processed: {input_text} with context: {context}",
                meta=YieldedOutputMeta(
                    cost=YieldedOutputCost(
                        input_cost=0.0001,
                        output_cost=0.0002,
                        total_cost=0.0003,
                    ),
                    usage=YieldedOutputTokenUsage(
                        prompt_tokens=100,
                        completion_tokens=200,
                        total_tokens=300,
                        latency=100,
                    ),
                ),
            )

        class Logger(TestRunLogger):
            def info(self, message: str) -> None:
                print(f"{message}")

            def error(self, message: str, e: Optional[Exception] = None) -> None:
                print(f"{message}", e)

        class MyCustomEvaluator(BaseEvaluator):
            def evaluate(
                self, result: LocalEvaluatorResultParameter, data: LocalData
            ) -> Dict[str, LocalEvaluatorReturn]:
                return {
                    "image_relevance": LocalEvaluatorReturn(
                        score=1, reasoning="Image is relevant to the input"
                    ),
                    "output_quality": LocalEvaluatorReturn(
                        score=True, reasoning="Output meets quality standards"
                    ),
                }

        # Test data with multiple variable types including image URLs
        data: Data = [
            {
                "input": "Analyze this image and text",
                "image_url": "https://images.unsplash.com/photo-1494871262121-49703fd34e2b?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGV4dHxlbnwwfHwwfHx8MA%3D%3D",
                "context": "Product description analysis",
                "expected_output": "Detailed analysis result",
            },
            {
                "input": "Compare these visual elements",
                "image_url": "https://www.thoughtco.com/thmb/i3i0DhTooFFhVLjnBcwJhT5z9Q0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/what-are-the-elements-of-art-182704_FINAL-9a30cee7896f4d3a9e078274851d5382.png",
                "context": "Visual comparison task",
                "expected_output": "Comparison summary",
            },
            {
                "input": "Extract text from image",
                "image_url": "https://tmm.chicagodistributioncenter.com/IsbnImages/9780226827025.jpg",
                "context": "OCR processing",
                "expected_output": "Extracted text content",
            },
        ]

        # Data structure with various column types including FILE_URL_VARIABLE for images
        data_structure = {
            "input": "INPUT",
            "image_url": "FILE_URL_VARIABLE",
            "context": "VARIABLE",
            "expected_output": "EXPECTED_OUTPUT",
        }

        result = (
            self.maxim.create_test_run(
                name="test run with image variables, local yields output and local evaluators",
                in_workspace_id=workspaceId,
            )
            .with_data_structure(data_structure)
            .with_data(data)
            .with_concurrency(1)  # Lower concurrency for image processing
            .with_evaluators(
                "Bias",
                MyCustomEvaluator(
                    pass_fail_criteria={
                        "image_relevance": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                ">", 0.8, "average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(">", 0.5),
                        ),
                        "output_quality": PassFailCriteria(
                            for_testrun_overall_pass_if=PassFailCriteriaForTestrunOverall(
                                overall_should_be="=", value=0.8, for_result="average"
                            ),
                            on_each_entry_pass_if=PassFailCriteriaOnEachEntry(
                                "=", True
                            ),
                        ),
                    }
                ),
            )
            .with_logger(Logger())
            .yields_output(processor)  # Test with local yields_output function
            .run()
        )

        self.assertIsNotNone(result)
        if result is not None:
            print(
                f"Test run completed with {len(result.failed_entry_indices)} failed entries"
            )

    def tearDown(self):
        # Clean up the Maxim instance
        if hasattr(self, "maxim"):
            self.maxim.cleanup()

        # Clear singleton instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
