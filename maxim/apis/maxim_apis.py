import contextlib
import json
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util import Retry, Timeout

from ..models import (
    AgentResponse,
    ChatCompletionMessage,
    DatasetEntry,
    DatasetRow,
    Evaluator,
    ExecutePromptForDataResponse,
    ExecuteWorkflowForDataResponse,
    Folder,
    HumanEvaluationConfig,
    ImageUrls,
    PromptResponse,
    RunType,
    SignedURLResponse,
    TestRun,
    TestRunEntry,
    TestRunResult,
    TestRunStatus,
    TestRunWithDatasetEntry,
    Tool,
    VersionAndRulesWithPromptChainId,
    VersionAndRulesWithPromptId,
)
from ..scribe import scribe
from ..version import current_version


class ConnectionPool:
    """
    Manages HTTP connection pooling for efficient network requests.

    This class provides a reusable connection pool with retry logic
    for handling transient network errors.
    """

    def __init__(self):
        """
        Initialize a new connection pool with retry configuration.
        """
        self.session = requests.Session()
        retries = Retry(
            connect=5,
            read=3,
            redirect=1,
            status=3,
            backoff_factor=0.4,
            status_forcelist=frozenset({413, 429, 500, 502, 503, 504}),
        )
        self.http = PoolManager(
            num_pools=2,
            maxsize=3,
            retries=retries,
            timeout=Timeout(connect=10, read=10),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    @contextlib.contextmanager
    def get_session(self):
        """
        Context manager that yields the session and ensures it's closed after use.

        Yields:
            requests.Session: The HTTP session object
        """
        yield self.session

    @contextlib.contextmanager
    def get_connection(self):
        """
        Context manager that yields the connection pool and ensures it's cleared after use.

        Yields:
            PoolManager: The HTTP connection pool
        """
        try:
            yield self.http
        finally:
            self.http.clear()


class MaximAPI:
    """
    Client for interacting with the Maxim API.

    This class provides methods for all available Maxim API endpoints,
    handling authentication, request formatting, and error handling.
    """

    connection_pool: ConnectionPool

    def __init__(self, base_url: str, api_key: str):
        """
        Initialize a new Maxim API client.

        Args:
            base_url: The base URL for the Maxim API
            api_key: The API key for authentication
        """
        self.connection_pool = ConnectionPool()
        self.base_url = base_url
        self.api_key = api_key

    def __make_network_call(
        self,
        method: str,
        endpoint: str,
        body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Make a network request to the Maxim API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            body: Request body as a string
            headers: Additional HTTP headers

        Returns:
            bytes: Response content

        Raises:
            Exception: If the request fails
        """
        if headers is None:
            headers = {}
        headers["x-maxim-api-key"] = self.api_key
        headers["x-maxim-sdk-version"] = current_version
        url = f"{self.base_url}{endpoint}"
        with self.connection_pool.get_session() as session:
            response = session.request(method, url, data=body, headers=headers)
            response.raise_for_status()
            if "x-lt-maxim-sdk-version" in response.headers:
                if response.headers["x-lt-maxim-sdk-version"] != current_version:
                    latest_version = response.headers["x-lt-maxim-sdk-version"]
                    latest_version_parts = list(map(int, latest_version.split(".")))
                    current_version_parts = list(map(int, current_version.split(".")))
                    if latest_version_parts > current_version_parts:
                        scribe().warning(
                            f"\033[33m[MaximSDK] SDK version is out of date. Please update to the latest version. Current version: {current_version}, Latest version: {latest_version}\033[0m",
                        )
            return response.content

    def get_prompt(self, id: str) -> VersionAndRulesWithPromptId:
        """
        Get a prompt by ID.

        Args:
            id: The prompt ID

        Returns:
            VersionAndRulesWithPromptId: The prompt details

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET", endpoint=f"/api/sdk/v4/prompts?promptId={id}"
            )
            data = json.loads(res.decode())["data"]
            return VersionAndRulesWithPromptId.from_dict(data)
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_prompts(self) -> List[VersionAndRulesWithPromptId]:
        """
        Get all prompts.

        Returns:
            List[VersionAndRulesWithPromptId]: List of all prompts

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(method="GET", endpoint="/api/sdk/v4/prompts")
            return [
                VersionAndRulesWithPromptId.from_dict(data)
                for data in json.loads(res)["data"]
            ]
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def getPromptChain(self, id: str) -> VersionAndRulesWithPromptChainId:
        """
        Get a prompt chain by ID.

        Args:
            id: The prompt chain ID

        Returns:
            VersionAndRulesWithPromptChainId: The prompt chain details

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET", endpoint=f"/api/sdk/v4/prompt-chains?promptChainId={id}"
            )
            json_response = json.loads(res.decode())
            return VersionAndRulesWithPromptChainId.from_dict(obj=json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_prompt_chains(self) -> List[VersionAndRulesWithPromptChainId]:
        """
        Get all prompt chains.

        Returns:
            List[VersionAndRulesWithPromptChainId]: List of all prompt chains

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET", endpoint="/api/sdk/v4/prompt-chains"
            )
            json_response = json.loads(res.decode())
            return [
                VersionAndRulesWithPromptChainId.from_dict(elem)
                for elem in json_response["data"]
            ]
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def run_prompt(
        self,
        model: str,
        messages: List[ChatCompletionMessage],
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ):
        """
        Run a custom prompt with the specified model and messages.

        Args:
            model: The model to use
            messages: List of chat messages
            tools: Optional list of tools to use
            **kwargs: Additional parameters to pass to the API

        Returns:
            PromptResponse: The response from the model

        Raises:
            Exception: If the request fails
        """
        try:
            payload: dict[str, Any] = {
                "type": "custom",
                "model": model,
                "messages": messages,
                "tools": tools,
            }
            if kwargs is not None:
                for key, value in kwargs.items():
                    if value is not None:
                        payload[key] = value
            payload = {k: v for k, v in payload.items() if v is not None}
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v4/prompts/run",
                body=json.dumps(payload),
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
            return PromptResponse.from_dict(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def run_prompt_version(
        self,
        prompt_version_id: str,
        input: str,
        image_urls: Optional[List[ImageUrls]],
        variables: Optional[dict[str, str]],
    ) -> Optional[PromptResponse]:
        """
        Run a specific prompt version with the given input.

        Args:
            prompt_version_id: The ID of the prompt version to run
            input: The input text for the prompt
            image_urls: Optional list of image URLs to include
            variables: Optional dictionary of variables to use

        Returns:
            Optional[PromptResponse]: The response from the prompt

        Raises:
            Exception: If the request fails
        """
        try:
            payload = {
                "type": "maxim",
                "promptVersionId": prompt_version_id,
                "input": input,
                "imageUrls": image_urls,
                "variables": variables or {},
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v4/prompts/run",
                body=json.dumps(payload),
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
            return PromptResponse.from_dict(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def run_prompt_chain_version(
        self,
        prompt_chain_version_id: str,
        input: str,
        variables: Optional[dict[str, str]],
    ) -> Optional[AgentResponse]:
        """
        Run a specific prompt chain version with the given input.

        Args:
            prompt_chain_version_id: The ID of the prompt chain version to run
            input: The input text for the prompt chain
            variables: Optional dictionary of variables to use

        Returns:
            Optional[AgentResponse]: The response from the prompt chain

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v4/agents/run",
                body=json.dumps(
                    {
                        "versionId": prompt_chain_version_id,
                        "input": input,
                        "variables": variables or {},
                    }
                ),
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
            return AgentResponse.from_dict(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_folder(self, id: str) -> Folder:
        """
        Get a folder by ID.

        Args:
            id: The folder ID

        Returns:
            Folder: The folder details

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET", endpoint=f"/api/sdk/v3/folders?folderId={id}"
            )
            json_response = json.loads(res.decode())
            if "tags" not in json_response:
                json_response["tags"] = {}
            return Folder.from_dict(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_folders(self) -> List[Folder]:
        """
        Get all folders.

        Returns:
            List[Folder]: List of all folders

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(method="GET", endpoint="/api/sdk/v3/folders")
            json_response = json.loads(res.decode())
            for elem in json_response["data"]:
                if "tags" not in elem:
                    elem["tags"] = {}
            return [Folder.from_dict(elem) for elem in json_response["data"]]
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def add_dataset_entries(
        self, dataset_id: str, dataset_entries: List[DatasetEntry]
    ) -> dict[str, Any]:
        """
        Add entries to a dataset.

        Args:
            dataset_id: The ID of the dataset
            dataset_entries: List of dataset entries to add

        Returns:
            dict[str, Any]: Response from the API

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v3/datasets/entries",
                body=json.dumps(
                    {
                        "datasetId": dataset_id,
                        "entries": [entry.to_json() for entry in dataset_entries],
                    }
                ),
            )
            return json.loads(res.decode())
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_dataset_total_rows(self, dataset_id: str) -> int:
        """
        Get the total number of rows in a dataset.

        Args:
            dataset_id: The ID of the dataset

        Returns:
            int: The total number of rows

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/datasets/total-rows?datasetId={dataset_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return json_response["data"]
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_dataset_row(self, dataset_id: str, row_index: int) -> Optional[DatasetRow]:
        """
        Get a specific row from a dataset.

        Args:
            dataset_id: The ID of the dataset
            row_index: The index of the row to retrieve

        Returns:
            Optional[DatasetRow]: The dataset row, or None if not found

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v2/datasets/row?datasetId={dataset_id}&row={row_index}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return DatasetRow.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
        except Exception as e:
            raise Exception(e)

    def get_dataset_structure(self, dataset_id: str) -> Dict[str, str]:
        """
        Get the structure of a dataset.

        Args:
            dataset_id: The ID of the dataset

        Returns:
            Dict[str, str]: The dataset structure

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/datasets/structure?datasetId={dataset_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
            return json_response["data"]
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def does_log_repository_exist(self, logger_id: str) -> bool:
        """
        Check if a log repository exists.

        Args:
            logger_id: The ID of the logger

        Returns:
            bool: True if the repository exists, False otherwise
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v3/log-repositories?loggerId={logger_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                return False
            return True
        except Exception:
            return False

    def push_logs(self, repository_id: str, logs: str) -> None:
        """
        Push logs to a repository.

        Args:
            repository_id: The ID of the repository
            logs: The logs to push

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint=f"/api/sdk/v3/log?id={repository_id}",
                body=logs,
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def fetch_platform_evaluator(self, name: str, in_workspace_id: str) -> Evaluator:
        """
        Fetch a platform evaluator by name.

        Args:
            name: The name of the evaluator
            in_workspace_id: The workspace ID

        Returns:
            Evaluator: The evaluator details

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/evaluators?name={name}&workspaceId={in_workspace_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return Evaluator.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def create_test_run(
        self,
        name: str,
        workspace_id: str,
        workflow_id: Optional[str],
        prompt_version_id: Optional[str],
        prompt_chain_version_id: Optional[str],
        run_type: RunType,
        evaluator_config: list[Evaluator],
        requires_local_run: bool,
        human_evaluation_config: Optional[HumanEvaluationConfig] = None,
    ) -> TestRun:
        """
        Create a new test run.

        Args:
            name: The name of the test run
            workspace_id: The workspace ID
            workflow_id: Optional workflow ID
            prompt_version_id: Optional prompt version ID
            prompt_chain_version_id: Optional prompt chain version ID
            run_type: The type of run
            evaluator_config: List of evaluators to use
            requires_local_run: Whether the test run requires local execution
            human_evaluation_config: Optional human evaluation configuration

        Returns:
            TestRun: The created test run

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v2/test-run/create",
                body=json.dumps(
                    {
                        k: v
                        for k, v in {
                            "name": name,
                            "workspaceId": workspace_id,
                            "runType": run_type.value,
                            "workflowId": (
                                workflow_id if workflow_id is not None else None
                            ),
                            "promptVersionId": (
                                prompt_version_id
                                if prompt_version_id is not None
                                else None
                            ),
                            "promptChainVersionId": (
                                prompt_chain_version_id
                                if prompt_chain_version_id is not None
                                else None
                            ),
                            "evaluatorConfig": [
                                evaluator.to_dict() for evaluator in evaluator_config
                            ],
                            "requiresLocalRun": requires_local_run,
                            "humanEvaluationConfig": (
                                human_evaluation_config.to_dict()
                                if human_evaluation_config
                                else None
                            ),
                        }.items()
                        if v is not None
                    }
                ),
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return TestRun.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def attach_dataset_to_test_run(self, test_run_id: str, dataset_id: str) -> None:
        """
        Attach a dataset to a test run.

        Args:
            test_run_id: The ID of the test run
            dataset_id: The ID of the dataset

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/attach-dataset",
                body=json.dumps({"testRunId": test_run_id, "datasetId": dataset_id}),
                headers={"Content-Type": "application/json"},
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def push_test_run_entry(
        self,
        test_run: Union[TestRun, TestRunWithDatasetEntry],
        entry: TestRunEntry,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Push an entry to a test run.

        Args:
            test_run: The test run
            entry: The test run entry to push
            run_config: Optional run configuration

        Raises:
            Exception: If the request fails
        """
        try:
            # making sure run_config has not null values
            if run_config is not None:
                run_config = {k: v for k, v in run_config.items() if v is not None}
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/push",
                body=json.dumps(
                    {
                        "testRun": test_run.to_dict(),
                        **({"runConfig": run_config} if run_config is not None else {}),
                        "entry": entry.to_dict(),
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def mark_test_run_processed(self, test_run_id: str) -> None:
        """
        Mark a test run as processed.

        Args:
            test_run_id: The ID of the test run

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/mark-processed",
                body=json.dumps({"testRunId": test_run_id}),
                headers={"Content-Type": "application/json"},
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def mark_test_run_failed(self, test_run_id: str) -> None:
        """
        Mark a test run as failed.

        Args:
            test_run_id: The ID of the test run

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/mark-failed",
                body=json.dumps({"testRunId": test_run_id}),
                headers={"Content-Type": "application/json"},
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_test_run_status(self, test_run_id: str) -> TestRunStatus:
        """
        Get the status of a test run.

        Args:
            test_run_id: The ID of the test run

        Returns:
            TestRunStatus: The status of the test run

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/test-run/status?testRunId={test_run_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            status: Dict[str, Any] = {}
            status = json_response["data"]["entryStatus"]
            status["testRunStatus"] = json_response["data"]["testRunStatus"]
            return TestRunStatus.dict_to_class(status)
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_test_run_final_result(self, test_run_id: str) -> TestRunResult:
        """
        Get the final result of a test run.

        Args:
            test_run_id: The ID of the test run

        Returns:
            TestRunResult: The final result of the test run

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/test-run/result?testRunId={test_run_id}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return TestRunResult.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def execute_workflow_for_data(
        self,
        workflow_id: str,
        data_entry: Dict[str, Union[str, List[str], None]],
        context_to_evaluate: Optional[str] = None,
    ) -> ExecuteWorkflowForDataResponse:
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/execute/workflow",
                body=json.dumps(
                    {
                        "workflowId": workflow_id,
                        "dataEntry": data_entry,
                        "contextToEvaluate": context_to_evaluate,
                    }
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return ExecuteWorkflowForDataResponse.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def execute_prompt_for_data(
        self,
        prompt_version_id: str,
        input: str,
        data_entry: Dict[str, Union[str, List[str], None]],
        context_to_evaluate: Optional[str] = None,
    ) -> ExecutePromptForDataResponse:
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/execute/prompt",
                body=json.dumps(
                    {
                        "promptVersionId": prompt_version_id,
                        "input": input,
                        "dataEntry": data_entry,
                        "contextToEvaluate": context_to_evaluate,
                    }
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return ExecutePromptForDataResponse.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def execute_prompt_chain_for_data(
        self,
        prompt_chain_version_id: str,
        input: str,
        data_entry: Dict[str, Union[str, List[str], None]],
        context_to_evaluate: Optional[str] = None,
    ) -> ExecutePromptForDataResponse:
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v1/test-run/execute/prompt-chain",
                body=json.dumps(
                    {
                        "promptChainVersionId": prompt_chain_version_id,
                        "input": input,
                        "dataEntry": data_entry,
                        "contextToEvaluate": context_to_evaluate,
                    }
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return ExecutePromptForDataResponse.dict_to_class(json_response["data"])
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)
        except Exception as e:
            raise Exception(e)

    def get_upload_url(self, key: str, mime_type: str, size: int) -> SignedURLResponse:
        """
        Get a signed URL for uploading a file.

        Args:
            key: The key (filename) for the upload
            mime_type: The MIME type of the file
            size: The size of the file in bytes

        Returns:
            SignedURLResponse: A dictionary containing the signed URL for upload

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="GET",
                endpoint=f"/api/sdk/v1/log-repositories/attachments/upload-url?key={key}&mimeType={mime_type}&size={size}",
            )
            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"])
            return {"url": json_response["data"]["url"]}
        except requests.HTTPError as e:
            if e.response is not None and e.response.json() is not None:
                error = e.response.json()
                raise Exception(error["error"]["message"])
            raise Exception(e)

    def upload_to_signed_url(self, url: str, data: bytes, mime_type: str) -> bool:
        """
        Upload data to a signed URL using multipart form data.

        Args:
            url: The signed URL to upload to
            data: The binary data to upload
            mime_type: The MIME type of the data

        Returns:
            bool: True if upload was successful, False otherwise
        """
        try:
            headers = {"Content-Type": mime_type}
            response = requests.put(url=url, data=data, headers=headers)
            response.raise_for_status()
            return True
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response else "unknown"
            message = str(e)
            raise Exception(f"Client response error: {status_code} {message}")
        except Exception as e:
            raise Exception(e)
