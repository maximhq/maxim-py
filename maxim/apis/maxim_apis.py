import contextlib
import json
import mimetypes
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union

import httpx
from httpx import (
    ConnectError,
    ConnectTimeout,
    HTTPStatusError,
    PoolTimeout,
    ProtocolError,
    ReadTimeout,
    RequestError,
    TimeoutException,
)

from maxim.models.dataset import DatasetEntryWithRowNo
from maxim.models.dataset import FileVariablePayload, VariableFileAttachment
from maxim.logger.components.attachment import (
    Attachment,
    FileDataAttachment,
    UrlAttachment,
    FileAttachment,
)


from ..models import (
    AgentResponse,
    ChatCompletionMessage,
    DatasetRow,
    Evaluator,
    ExecutePromptForDataResponse,
    ExecuteWorkflowForDataResponse,
    Folder,
    HumanEvaluationConfig,
    ImageURL,
    PromptResponse,
    RunType,
    SignedURLResponse,
    TestRun,
    TestRunEntry,
    TestRunResult,
    TestRunStatus,
    TestRunWithDatasetEntry,
    Tool,
    Variable,
    VersionAndRulesWithPromptChainId,
    VersionAndRulesWithPromptId,
    DatasetAttachmentUploadURL,

)
from ..scribe import scribe
from ..version import current_version


class ConnectionPool:
    """
    Manages HTTP client pooling for efficient network requests.

    This class provides a reusable client with retry logic
    for handling transient network errors.
    """

    def __init__(self):
        """
        Initialize a new client with connection pooling configuration.
        """
        limits = httpx.Limits(
            max_keepalive_connections=10,
            max_connections=20,
            keepalive_expiry=60,
        )

        timeout = httpx.Timeout(
            connect=15.0,
            read=30.0,
            write=120.0,
            pool=120.0,
        )

        self.client = httpx.Client(
            limits=limits,
            timeout=timeout,
            headers={"Connection": "keep-alive", "Keep-Alive": "timeout=60, max=100"},
            follow_redirects=True,
        )

    @contextlib.contextmanager
    def get_client(self):
        """
        Context manager that yields the client.

        Yields:
            httpx.Client: The HTTP client object
        """
        yield self.client


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
        self.max_retries = 3
        self.base_delay = 1.0

    def __make_network_call_with_retry(
        self,
        method: str,
        endpoint: str,
        body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 10,
        max_pool_retries: int = 20,
    ) -> bytes:
        """
        Make a network request with comprehensive retry logic for connection errors.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            body: Request body as a string
            headers: Additional HTTP headers
            max_retries: Maximum number of retries for connection errors
            max_pool_retries: Maximum number of retries specifically for pool exhaustion errors

        Returns:
            bytes: Response content

        Raises:
            Exception: If the request fails after all retries
        """
        if headers is None:
            headers = {}
        headers["x-maxim-api-key"] = self.api_key
        headers["x-maxim-sdk-version"] = current_version
        headers["Connection"] = "keep-alive"  # Encourage connection reuse

        url = f"{self.base_url}{endpoint}"
        last_exception = None
        pool_retry_count = 0

        for attempt in range(max_retries + 1):
            try:
                with self.connection_pool.get_client() as client:
                    response = client.request(
                        method=method,
                        url=url,
                        content=body,
                        headers=headers,
                    )
                    response.raise_for_status()

                    # Handle version check
                    if "x-lt-maxim-sdk-version" in response.headers:
                        if (
                            response.headers["x-lt-maxim-sdk-version"]
                            != current_version
                        ):
                            latest_version = response.headers["x-lt-maxim-sdk-version"]
                            latest_version_parts = list(
                                map(int, latest_version.split("."))
                            )
                            current_version_parts = list(
                                map(int, current_version.split("."))
                            )
                            if latest_version_parts > current_version_parts:
                                scribe().warning(
                                    f"\033[33m[MaximSDK] SDK version is out of date. Please update to the latest version. Current version: {current_version}, Latest version: {latest_version}\033[0m",
                                )

                    return response.content

            except PoolTimeout as e:
                # Handle pool exhaustion separately with more aggressive retries
                last_exception = e
                pool_retry_count += 1
                if pool_retry_count <= max_pool_retries:
                    # Shorter delay for pool exhaustion - just waiting for connections to free up
                    delay = 0.1 * (1.5**pool_retry_count)  # Shorter exponential backoff
                    scribe().debug(
                        f"[MaximSDK] Connection pool exhausted on attempt {pool_retry_count}/{max_pool_retries}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(
                        f"[MaximSDK] Connection pool exhausted after {max_pool_retries} pool retries. Last error: {e}"
                    )
                    raise Exception(
                        f"Connection pool exhausted after {max_pool_retries} retries: {e}"
                    ) from e

            except (
                ConnectError,
                ConnectTimeout,
                ReadTimeout,
                ProtocolError,
                TimeoutException,
            ) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(
                        f"[MaximSDK] Failed to establish connection after {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise Exception(
                        f"Connection failed after {max_retries + 1} attempts: {e}"
                    ) from e

            except HTTPStatusError as e:
                # For HTTP errors, don't retry - these are usually permanent
                if e.response is not None:
                    try:
                        error_data = e.response.json()
                        if (
                            error_data
                            and isinstance(error_data, dict)
                            and "error" in error_data
                            and isinstance(error_data["error"], dict)
                            and "message" in error_data["error"]
                        ):
                            raise Exception(error_data["error"]["message"]) from e
                    except (ValueError, KeyError):
                        pass
                raise Exception(f"HTTP Error {e.response.status_code}: {str(e)}") from e

            except RequestError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)
                    scribe().debug(
                        f"[MaximSDK] Request error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(
                        f"[MaximSDK] Request failed after {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise Exception(
                        f"Request failed after {max_retries + 1} attempts: {e}"
                    ) from e

            except Exception as e:
                # Blanket exception handler for any other unexpected errors with retry
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)
                    scribe().debug(
                        f"[MaximSDK] Unexpected error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(f"[MaximSDK] Unexpected error in network call: {e}")
                    raise Exception(f"Unexpected error: {e}") from e

        # This should never be reached, but just in case
        if last_exception:
            raise Exception(f"Network call failed: {last_exception}")
        raise Exception("Network call failed for unknown reasons")

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
        return self.__make_network_call_with_retry(
            method, endpoint, body, headers, self.max_retries
        )

    def get_prompt(self, id: str, prompt_version_number: Optional[int] = None) -> VersionAndRulesWithPromptId:
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
            endpoint = f"/api/sdk/v4/prompts?promptId={id}"
            if prompt_version_number is not None:
                endpoint += f"&promptVersionNumber={prompt_version_number}"
            res = self.__make_network_call(method="GET", endpoint=endpoint)
            data = json.loads(res.decode())["data"]
            return VersionAndRulesWithPromptId.from_dict(data)
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

    def run_prompt_version(
        self,
        prompt_version_id: str,
        input: str,
        image_urls: Optional[List[ImageURL]],
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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

    def upload_dataset_entry_attachments(self, dataset_id: str, entry_id: str, entry: DatasetEntryWithRowNo) -> Optional[FileVariablePayload]:
        """
        Get upload URL for an attachment and upload the file to the server.
        """

        # Process each attachment in the entry
        file_attachments = []
        for attachment in entry.payload:
            if isinstance(attachment, str):
                continue
            try:
                file_data, mime_type, size = self._process_attachment(attachment)

                # Validate MIME type
                if not mime_type or mime_type == "application/octet-stream":
                    # Try to infer MIME type from any available hint
                    hint = (
                        getattr(attachment, "name", None)
                        or getattr(attachment, "path", None)
                        or getattr(attachment, "url", None)
                    )
                    if hint:
                        inferred_type, _ = mimetypes.guess_type(hint)
                        if inferred_type:
                            mime_type = inferred_type
                # For uploads, reject unknown/generic MIME
                if isinstance(attachment, (FileDataAttachment, FileAttachment)) and (
                    not mime_type or mime_type == "application/octet-stream"
                ):
                    raise ValueError(
                        f"Unrecognized/unsafe MIME type for upload: {getattr(attachment, 'name', getattr(attachment, 'path', 'attachment'))}"
                    )

                # Get upload URL for dataset attachment
                if isinstance(attachment, (FileDataAttachment, FileAttachment)):
                    upload_response = self.get_upload_url_for_dataset_attachment(
                        dataset_id=dataset_id,
                        entry_id=entry_id,
                        key=attachment.id,
                        column_id = entry.column_id,
                        mime_type=mime_type,
                        size=size,
                    )
                    upload_success = self.upload_to_signed_url(upload_response["url"], file_data, mime_type)

                    if not upload_success:
                        raise Exception(f"Failed to upload file {attachment.name} to signed URL")
                    # Create VariableFileAttachment with hosted=True
                    file_attachment = VariableFileAttachment(
                        id=attachment.id,
                        url="",
                        hosted=True,
                        prefix=upload_response["key"],
                        props={
                            "size": size,
                            "type": mime_type,
                        },
                    )
                else:
                    file_attachment = VariableFileAttachment(
                        id=attachment.id,
                        url=attachment.url,
                        hosted=False,
                        prefix="",
                        props={
                            "size": size,
                            "type": mime_type,
                        },
                    )

                file_attachments.append(file_attachment)
            except Exception as e:
                # Log the error but continue processing other attachments
                scribe().error(
                    f"Failed to process attachment "
                    f"id={getattr(attachment, 'id', 'unknown')} "
                    f"name={getattr(attachment, 'name', getattr(attachment, 'path', getattr(attachment, 'url', 'unknown')))}: {e}"
                )
                continue

        # Create FileVariablePayload for this entry
        if file_attachments:
            file_payload = FileVariablePayload(
                text=entry.column_name,
                files=file_attachments,
                entry_id=entry_id,
            )
            return file_payload

        return None
    
    def process_attachment(self, attachment: Attachment) -> tuple[bytes, str, int]:
        """
        Process an attachment and return file data, MIME type, and size.
        """
        return self._process_attachment(attachment)

    def _process_attachment(self, attachment: Attachment) -> tuple[bytes, str, int]:
        """
        Process an attachment and return file data, MIME type, and size.

        Args:
            attachment: The attachment to process (UrlAttachment, FileDataAttachment, or FileAttachment)

        Returns:
            tuple: (file_data, mime_type, size)

        Raises:
            TypeError: If attachment type is not supported
            ValueError: If attachment data is invalid
            Exception: For network or file I/O errors
        """
        if isinstance(attachment, UrlAttachment):
            return self._process_url_attachment(attachment)
        if isinstance(attachment, (FileDataAttachment, FileAttachment)):
            return self._process_file_attachment(attachment)
        raise TypeError(f"Invalid attachment type: {type(attachment).__name__}")

    def _process_url_attachment(self, attachment: UrlAttachment) -> tuple[bytes, str, int]:
        """
        Probe a URL attachment (HEAD) to get metadata without downloading content.

        Args:
            attachment: The URL attachment to process

        Returns:
            tuple: (file_data, mime_type, size)
        """
        try:
            with self.connection_pool.get_client() as client:
                # Validate URL
                parsed = urlparse(attachment.url or "")
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError(f"Invalid URL: {attachment.url!s}")

                # Get file info from HEAD request
                head_response = client.head(attachment.url, timeout=30.0)
                head_response.raise_for_status()

                mime_type = head_response.headers.get('content-type', 'application/octet-stream')
                content_length = head_response.headers.get('content-length')
                size = 0
                if content_length:
                    try:
                        size = int(content_length)
                    except ValueError:
                        size = 0
                return b"", mime_type, size
                
        except Exception as e:
            raise Exception(f"Failed to process URL attachment via HEAD {attachment.url!s}: {e}") from e

    def _process_file_attachment(self, attachment: Union[FileDataAttachment, FileAttachment]) -> tuple[bytes, str, int]:
        """
        Process a file attachment (FileDataAttachment or FileAttachment).
        
        Args:
            attachment: The file attachment to process
            
        Returns:
            tuple: (file_data, mime_type, size)
        """
        try:
            MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
            if isinstance(attachment, FileDataAttachment):
                file_data = attachment.data
                mime_type = attachment.mime_type or 'application/octet-stream'
                size = len(file_data)
                if size > MAX_FILE_SIZE_BYTES:
                    raise ValueError(f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes")
            else:  # FileAttachment
                import os
                if not os.path.exists(attachment.path):
                    raise FileNotFoundError(f"File not found: {attachment.path}")
                
                size = os.path.getsize(attachment.path)
                if size > MAX_FILE_SIZE_BYTES:
                    raise ValueError(f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes")
                mime_type = attachment.mime_type or 'application/octet-stream'
                with open(attachment.path, 'rb') as f:
                    file_data = f.read()

            return file_data, mime_type, size

        except Exception as e:
            attachment_name = getattr(attachment, "name", "unknown")
            raise Exception(f"Failed to process file attachment {attachment_name}: {e}") from e

    def get_upload_url_for_dataset_attachment(self, dataset_id: str, entry_id: str, key: str, column_id:Optional[str], mime_type: str = "application/octet-stream", size: int = 0) -> DatasetAttachmentUploadURL:
        """
        Get a signed URL for uploading a file.

        Args:
            dataset_id: The dataset identifier
            entry_id: The entry identifier within the dataset
            key: The key (filename) for the upload
            column_id: The column identifier within the dataset
            mime_type: The MIME type of the file
            size: The size of the file in bytes

        Returns:
            DatasetAttachmentUploadURL: A dict containing the signed PUT URL and storage key

        Raises:
            Exception: If the request fails
        """
        try:
            payload = {
                "datasetId": dataset_id,
                "entryId": entry_id,
                "file": {
                    "id": key,
                    "type": mime_type,
                    "size": size,
                },
            }
            if column_id is not None:
                payload["columnId"] = column_id

            res = self.__make_network_call(
                method="PUT",
                endpoint="/api/sdk/v4/datasets/entries/attachments/",
                body=json.dumps(payload),
            )

            json_response = json.loads(res.decode())
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])

            return {
                "url": json_response["data"]["url"],
                "key": json_response["data"]["key"],
            }
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        tags: Optional[list[str]] = None,
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
                            "tags": tags,
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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
                endpoint="/api/sdk/v2/test-run/push",
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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

    def execute_prompt_for_data(
        self,
        prompt_version_id: str,
        input: str,
        variables: Dict[str, Variable],
        context_to_evaluate: Optional[str] = None,
    ) -> ExecutePromptForDataResponse:
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v2/test-run/execute/prompt",
                body=json.dumps(
                    {
                        "promptVersionId": prompt_version_id,
                        "input": input,
                        "dataEntry": {
                            key: variable.to_json()
                            for key, variable in variables.items()
                        },
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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

    def execute_prompt_chain_for_data(
        self,
        prompt_chain_version_id: str,
        input: str,
        variables: Dict[str, Variable],
        context_to_evaluate: Optional[str] = None,
    ) -> ExecutePromptForDataResponse:
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v2/test-run/execute/prompt-chain",
                body=json.dumps(
                    {
                        "promptChainVersionId": prompt_chain_version_id,
                        "input": input,
                        "dataEntry": {
                            key: variable.to_json()
                            for key, variable in variables.items()
                        },
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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

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
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e

    def upload_to_signed_url(self, url: str, data: bytes, mime_type: str) -> bool:
        """
        Upload data to a signed URL using multipart form data with retry logic.

        Args:
            url: The signed URL to upload to
            data: The binary data to upload
            mime_type: The MIME type of the data

        Returns:
            bool: True if upload was successful, False otherwise
        """
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                headers = {"Content-Type": mime_type}
                timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=60.0)

                response = httpx.put(
                    url=url,
                    content=data,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                return True

            except (
                ConnectError,
                ConnectTimeout,
                ReadTimeout,
                ProtocolError,
                TimeoutException,
            ) as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)
                    scribe().debug(
                        f"[MaximSDK] File upload connection error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(
                        f"[MaximSDK] File upload failed after {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise Exception(
                        f"File upload connection failed after {max_retries + 1} attempts: {e}"
                    ) from e

            except HTTPStatusError as e:
                status_code = e.response.status_code if e.response else "unknown"
                message = str(e)
                raise Exception(
                    f"Client response error: {status_code} {message}"
                ) from e

            except RequestError as e:
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)
                    scribe().debug(
                        f"[MaximSDK] File upload request error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(
                        f"[MaximSDK] File upload request failed after {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise Exception(
                        f"File upload request failed after {max_retries + 1} attempts: {e}"
                    ) from e

            except Exception as e:
                # Blanket exception handler for any other unexpected file upload errors with retry
                last_exception = e
                if attempt < max_retries:
                    delay = self.base_delay * (2**attempt)
                    scribe().warning(
                        f"[MaximSDK] Unexpected file upload error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    scribe().error(f"[MaximSDK] Unexpected error in file upload: {e}")
                    raise Exception(f"Unexpected file upload error: {e}") from e

        # This should never be reached, but just in case
        if last_exception:
            raise Exception(f"File upload failed: {last_exception}")
        raise Exception("File upload failed for unknown reasons")

    def create_dataset_entries(self, dataset_id: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Create dataset entries via API.

        Args:
            dataset_id: The ID of the dataset to add entries to
            entries: List of dataset entries to add

        Returns:
            dict[str, Any]: Response data from the API

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="POST",
                endpoint="/api/sdk/v4/datasets/entries",
                body=json.dumps({
                    "datasetId": dataset_id,
                    "entries": entries,
                }),
            )
            response_data = json.loads(res.decode())
            if "error" in response_data:
                raise Exception(response_data["error"]["message"])
            return response_data
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(e) from e

    def update_dataset_entries(self, dataset_id: str, updates: list[dict[str, Any]]) -> None:
        """
        Update dataset entries with file attachments.

        Args:
            dataset_id: The ID of the dataset
            updates: List of updates to apply

        Raises:
            Exception: If the request fails
        """
        try:
            res = self.__make_network_call(
                method="PUT",
                endpoint="/api/sdk/v4/datasets/entries",
                body=json.dumps({
                    "datasetId": dataset_id,
                    "updates": updates,
                }),
            )
            response_data = json.loads(res.decode())
            if "error" in response_data:
                raise Exception(response_data["error"]["message"])
        except httpx.HTTPStatusError as e:
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    if (
                        error_data
                        and isinstance(error_data, dict)
                        and "error" in error_data
                        and isinstance(error_data["error"], dict)
                        and "message" in error_data["error"]
                    ):
                        raise Exception(error_data["error"]["message"]) from e
                except (ValueError, KeyError):
                    pass
            raise Exception(e) from e
        except Exception as e:
            raise Exception(f"Failed to update dataset entries with attachments: {e!s}") from e
