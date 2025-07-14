"""
Module for handling log writing operations in the Maxim SDK.

This module provides classes for configuring and managing log writing,
including automatic flushing, file-based persistence, and API integration.
"""

import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.client import RemoteDisconnected
from queue import Queue
from typing import Optional

import filetype
import httpx
from ..apis import MaximAPI
from ..scribe import scribe
from .components.types import CommitLog


class LogWriterConfig:
    """
    Configuration class for the LogWriter.

    Attributes:
        base_url: Base URL for the Maxim API.
        api_key: API key for authentication.
        repository_id: ID of the repository to write logs to.
        auto_flush: Whether to automatically flush logs periodically.
        flush_interval: Time interval in seconds between automatic flushes.
        is_debug: Whether to enable debug logging.
        raise_exceptions: Whether to raise exceptions or handle them silently.
    """

    def __init__(
        self,
        base_url,
        api_key,
        repository_id,
        auto_flush=True,
        flush_interval: Optional[int] = 10,
        is_debug=False,
        raise_exceptions=False,
    ):
        """
        Initialize a LogWriterConfig instance.

        Args:
            base_url: Base URL for the Maxim API.
            api_key: API key for authentication.
            repository_id: ID of the repository to write logs to.
            auto_flush: Whether to automatically flush logs periodically.
            flush_interval: Time interval in seconds between automatic flushes.
            is_debug: Whether to enable debug logging.
            raise_exceptions: Whether to raise exceptions or handle them silently.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.repository_id = repository_id
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        self.is_debug = is_debug
        self.raise_exceptions = raise_exceptions


class LogWriter:
    """
    Handles writing logs to the Maxim API and local filesystem.

    This class manages a queue of logs, periodically flushes them to the API,
    and provides fallback to local filesystem storage when API calls fail.
    """

    def __init__(self, config: LogWriterConfig):
        """
        Initialize a LogWriter instance.

        Args:
            config: Configuration for the LogWriter.

        Raises:
            ValueError: If auto_flush is enabled but flush_interval is None.
        """
        self.is_running = True
        self.id = str(uuid.uuid4())
        self.config = config
        self.maxim_api = MaximAPI(config.base_url, config.api_key)
        self.queue = Queue()
        self.upload_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.upload_executor = ThreadPoolExecutor(max_workers=1)
        self.max_in_memory_logs = 100
        self.is_debug = config.is_debug
        self.raise_exceptions = config.raise_exceptions
        self.logs_dir = os.path.join(
            tempfile.gettempdir(), f"maxim-sdk/{self.id}/maxim-logs"
        )
        self.sinks: list[LogWriter] = []
        self.__flush_thread = None
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
        except Exception:
            scribe().info("[MaximSDK] Maxim library does not have FS access")
        if self.config.auto_flush:
            if self.config.flush_interval is not None:
                scribe().info(
                    "[MaximSDK] Starting flush thread with interval {%s} seconds",
                    self.config.flush_interval,
                )
                # Starting log flush thread
                self.__flush_thread = threading.Thread(target=self.__sync_timer)
                self.__flush_thread.daemon = True
                self.__flush_thread.start()
            else:
                raise ValueError(
                    "flush_interval is set to None.flush_interval has to be a number"
                )

    def add_sink(self, config: LogWriterConfig) -> None:
        """
        Adds a sink to the logger.
        """
        self.sinks.append(LogWriter(config))

    @property
    def repository_id(self):
        """
        Get the repository ID.

        Returns:
            str: The repository ID.
        """
        return self.config.repository_id

    def __sync_timer(self):
        """
        Timer function for periodic log flushing.

        This method runs in a separate thread and periodically calls flush().

        Raises:
            ValueError: If flush_interval is None.
        """
        while self.is_running:
            self.flush()
            if self.config.flush_interval is None:
                raise ValueError(
                    "flush_interval is set to None. flush_interval has to be a number"
                )
            time.sleep(self.config.flush_interval)

    def upload_file_data(self, add_attachment_log: CommitLog):
        """
        Upload a file attachment to the Maxim API.
        """
        attachment = add_attachment_log.data
        if attachment is None:
            return
        try:
            if "data" not in attachment:
                scribe().warning(
                    "[MaximSDK] Data is not set for file attachment. Skipping upload"
                )
                return
            file_data = attachment["data"]
            # Create a temporary file-like object in memory to guess mime type
            mime_type = filetype.guess_mime(file_data)
            if mime_type is None:
                mime_type = attachment.get("mime_type", None)
            if mime_type is None:
                mime_type = "application/octet-stream"
            size = len(file_data)
            key = attachment["key"]
            resp = self.maxim_api.get_upload_url(key, mime_type, size)
            # Writing commit log
            del attachment["data"]
            scribe().debug(
                "[MaximSDK] Uploading file attachment. Attachment: %s", attachment
            )
            self.commit(
                CommitLog(
                    action="add-attachment",
                    entity=add_attachment_log.entity,
                    entity_id=add_attachment_log.entity_id,
                    data=attachment,
                )
            )
            # Uploading file to the Maxim API
            self.maxim_api.upload_to_signed_url(resp["url"], file_data, mime_type)
        except Exception as e:
            import traceback

            scribe().error(
                f"[MaximSDK] Failed to upload file. Error: {e} \n {traceback.format_exc()}"
            )
            attachment["retry_count"] = attachment.get("retry_count", 0) + 1
            if attachment["retry_count"] < 3:
                self.upload_queue.put(attachment)

    def upload_file(self, add_attachment_log: CommitLog):
        """
        Upload a file data attachment to the Maxim API.
        """
        scribe().debug(
            "[MaximSDK] Uploading file attachment. Attachment: %s",
            add_attachment_log.data,
        )
        attachment = add_attachment_log.data
        if attachment is None:
            return
        try:
            if "path" not in attachment:
                scribe().warning(
                    "[MaximSDK] Path is not set for file data attachment. Skipping upload"
                )
                return
            path = attachment["path"]
            file_data = None
            # Reading file from this path
            with open(path, "rb") as f:
                file_data = f.read()
            # Inferring mime type and size of the file
            mime_type = filetype.guess_mime(file_data)
            if mime_type is None:
                mime_type = attachment.get("mime_type", None)
            if mime_type is None:
                mime_type = "application/octet-stream"
            scribe().debug(
                "[MaximSDK] Mime type: %s",
                mime_type,
            )
            size = os.path.getsize(path)
            key = attachment["key"]
            scribe().debug(
                "[MaximSDK] Getting upload url for file. Key: %s, Mime type: %s, Size: %s",
                key,
                mime_type,
                size,
            )
            resp = self.maxim_api.get_upload_url(key, mime_type, size)
            # Pushing back the command
            del attachment["path"]
            self.commit(
                CommitLog(
                    action="add-attachment",
                    entity=add_attachment_log.entity,
                    entity_id=add_attachment_log.entity_id,
                    data=attachment,
                )
            )
            # Uploading file to the Maxim API
            scribe().debug(
                "[MaximSDK] Uploading file to the Maxim API. URL: %s, Mime type: %s, Size: %s",
                resp["url"],
                mime_type,
                size,
            )
            self.maxim_api.upload_to_signed_url(resp["url"], file_data, mime_type)
            scribe().debug(
                "[MaximSDK] File uploaded to the Maxim API. URL: %s, Mime type: %s, Size: %s",
                resp["url"],
                mime_type,
                size,
            )
        except Exception as e:
            # here we will retry the upload and add retry count
            scribe().error(
                f"[MaximSDK] Failed to upload file data attachment. Error: {e}"
            )
            attachment["retry_count"] = attachment.get("retry_count", 0) + 1
            if attachment["retry_count"] < 3:
                self.upload_queue.put(attachment)

    def upload_attachments(self, attachment_logs: list[CommitLog]):
        """
        Upload all attachments to the Maxim API.
        """
        for log in attachment_logs:
            self.upload_attachment(log)

    def upload_attachment(self, add_attachment_log: CommitLog):
        """
        Upload an attachment to the Maxim API.

        Args:
            attachment: Attachment object to upload.
        """
        if add_attachment_log.data is None:
            return
        type = add_attachment_log.data["type"]
        if type == "file":
            self.upload_file(add_attachment_log)
        elif type == "file_data":
            self.upload_file_data(add_attachment_log)
        elif type == "url":
            if "url" not in add_attachment_log.data:
                scribe().warning(
                    f"[MaximSDK] URL is not set for attachment. Skipping upload. Attachment: {add_attachment_log.serialize()}"
                )
                return
            if "key" not in add_attachment_log.data:
                del add_attachment_log.data["key"]
            self.commit(
                CommitLog(
                    action="add-attachment",
                    entity=add_attachment_log.entity,
                    entity_id=add_attachment_log.entity_id,
                    data=add_attachment_log.data,
                )
            )
        else:
            raise ValueError(f"Invalid attachment type: {type}")

    def is_running_on_lambda(self):
        """
        Check if the code is running in an AWS Lambda environment.

        Returns:
            bool: True if running in AWS Lambda, False otherwise.
        """
        return "AWS_LAMBDA_FUNCTION_NAME" in os.environ

    def write_to_file(self, logs):
        """
        Write logs to a local file.

        Args:
            logs: List of CommitLog objects to write.

        Returns:
            str or None: Path to the file if successful, None otherwise.

        Raises:
            Exception: If raise_exceptions is True and writing fails.
        """
        try:
            filename = f"logs-{time.strftime('%Y-%m-%dT%H:%M:%SZ')}.log"
            filepath = os.path.join(self.logs_dir, filename)
            scribe().info(f"[MaximSDK] Writing logs to file: {filename}")
            with open(filepath, "w") as file:
                for log in logs:
                    file.write(log.serialize() + "\n")
            return filepath
        except Exception as e:
            scribe().warning(
                f"[MaximSDK] Failed to write logs to file. We will keep it in memory. Error: {e}"
            )
            if self.raise_exceptions:
                raise e
            return None

    def flush_log_files(self):
        """
        Flush logs from files to the Maxim API.

        This method reads log files from the logs directory, sends them to the API,
        and deletes the files if successful.

        Raises:
            Exception: If raise_exceptions is True and an error occurs.
        """
        try:
            if not os.path.exists(self.logs_dir):
                return
            files = os.listdir(self.logs_dir)
            for file in files:
                with open(os.path.join(self.logs_dir, file), "r") as f:
                    logs = f.read()
                try:
                    self.maxim_api.push_logs(self.config.repository_id, logs)
                    os.remove(os.path.join(self.logs_dir, file))
                except Exception as e:
                    scribe().warning(
                        f"[MaximSDK] Failed to access filesystem. Error: {e}"
                    )
                    if self.raise_exceptions:
                        raise Exception(e)
        except Exception as e:
            scribe().warning(f"[MaximSDK] Failed to access filesystem. Error: {e}")

    def can_access_filesystem(self):
        """
        Check if the filesystem is accessible for writing.

        Returns:
            bool: True if filesystem is accessible, False otherwise.
        """
        try:
            return os.access(tempfile.gettempdir(), os.W_OK)
        except Exception:
            return False

    def process_large_log(self, file_id: str, key: str, serialized_log: str) -> str:
        """
        Flush a large commit log to the Maxim API.
        Exceptions are passed onto the caller.
        Returns the final gcs command
        """
        resp = self.maxim_api.get_upload_url(key, "text/plain", len(serialized_log))
        self.maxim_api.upload_to_signed_url(
            resp["url"], serialized_log.encode("utf-8"), "text/plain"
        )
        return (
            f'storage{{id={file_id},action=process-large-log,data={{"key":"{key}"}}}}'
        )

    def flush_logs(self, logs: list[CommitLog]):
        """
        Flush logs to the Maxim API.

        This method attempts to send logs to the API, with fallback mechanisms
        for handling failures based on the environment.

        Args:
            logs: List of CommitLog objects to flush.
        """
        try:
            # Pushing old logs first
            if self.can_access_filesystem():
                self.flush_log_files()
            # Serialize all logs
            # Maximum size for each batch (5MB)
            MAX_BATCH_SIZE = 5 * 1024 * 1024
            # Split logs into batches to ensure each batch is under 5MB
            current_batch = []
            current_size = 0
            for log in logs:
                # Calculate size of this log plus a newline character
                log_str = log.serialize()
                log_size = len(log_str.encode("utf-8")) + 1
                # Here we will check if its above 900 kb - and if so - we will pass it to the process_large_log first
                # And once we receive the key we will add it to the current batch
                if log_size > 900 * 1024:
                    scribe().debug(
                        f"[MaximSDK] Log is too large. Size: {log_size}. Flushing via storage."
                    )
                    repo_id = self.config.repository_id
                    file_id = str(uuid.uuid4())
                    key = f"{repo_id}/large-logs/{file_id}"
                    log_str = self.process_large_log(file_id, key, log_str)
                    log_size = len(log_str.encode("utf-8")) + 1
                    scribe().debug(
                        f"[MaximSDK] Log flushed to storage. Size: {log_size}. Key: {key}. Log: {log_str}"
                    )
                # If adding this log would exceed the limit, push current batch and start a new one
                if current_size + log_size > MAX_BATCH_SIZE and current_batch:
                    batch_content = "\n".join(current_batch)
                    self.maxim_api.push_logs(self.config.repository_id, batch_content)
                    current_batch = []
                    current_size = 0
                # Add log to current batch
                current_batch.append(log_str)
                current_size += log_size
            # Push any remaining logs
            if current_batch:
                batch_content = "\n".join(current_batch)
                self.maxim_api.push_logs(self.config.repository_id, batch_content)
            scribe().debug("[MaximSDK] Flush complete")
        except (
            RemoteDisconnected,
            ConnectionError,
            TimeoutError,
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ProtocolError,
            httpx.TimeoutException,
            httpx.PoolTimeout,
            httpx.RequestError,
        ) as e:
            # Handle specific connection interruption exceptions
            scribe().warning(
                f"[MaximSDK] Connection to server was interrupted. This issue has been improved with better retry logic. Error: {e}. Storing logs locally and will retry."
            )
            self._handle_flush_failure(logs, e)
        except Exception as e:
            # Handle all other exceptions
            scribe().warning(
                f"[MaximSDK] Failed to push logs to server. Error: {e}. We are trying to store logs in a file and push it later."
            )
            self._handle_flush_failure(logs, e)

    def _handle_flush_failure(self, logs: list[CommitLog], error: Exception):
        """
        Handle failure to flush logs by storing them locally or keeping in memory.

        Args:
            logs: List of CommitLog objects that failed to flush.
            error: The exception that caused the failure.
        """
        if self.is_running_on_lambda():
            scribe().debug(
                "[MaximSDK] As we are running on lambda - we will keep logs in memory for next attempt"
            )
            for log in logs:
                self.queue.put(log)
            scribe().debug("[MaximSDK] Logs added back to queue for next attempt")
        else:
            if self.can_access_filesystem():
                self.write_to_file(logs)
                scribe().warning(
                    f"[MaximSDK] Failed to push logs to server. Writing logs to file. Error: {error}"
                )
            else:
                for log in logs:
                    self.queue.put(log)
                scribe().debug("[MaximSDK] Logs added back to queue for next attempt")

    def commit(self, log: CommitLog):
        """
        Add a log to the queue for later flushing.

        Args:
            log: CommitLog object to add to the queue.

        Raises:
            ValueError: If the entity_id is invalid and raise_exceptions is True.
        """
        # Here first we send these to sink
        try:
            for sink in self.sinks:
                sink.commit(log)
        except Exception as e:
            scribe().warning(f"[MaximSDK] Failed to send log to sink. Error: {e}")
            if self.raise_exceptions:
                raise e
        if not re.match(r"^[a-zA-Z0-9_-]+$", log.entity_id):
            if self.raise_exceptions:
                raise ValueError(
                    f"Invalid ID: {log.entity_id}. ID must only contain alphanumeric characters, hyphens, and underscores. Event will not be logged."
                )
            # Silently drop the log as we have already logged the error in the base container
            return
        # Here we can process the log - and find all upload-attachment commands
        # and push them to separate queue
        if log.action == "upload-attachment":
            if log.data is None:
                # We can't upload the attachment
                scribe().warning(
                    f"[MaximSDK] Attachment data is not set for log. Skipping upload. Log: {log.serialize()}"
                )
                return
            # updating key of the attachment log
            repo_id = self.config.repository_id
            entity_id = log.entity_id
            file_id = log.data["id"]
            key = f"{repo_id}/{log.entity.value}/{entity_id}/files/original/{file_id}"
            log.data["key"] = key
            self.upload_queue.put(log)
        else:
            self.queue.put(log)
        if self.queue.qsize() > self.max_in_memory_logs:
            self.flush()

    def flush_upload_attachment_logs(self, is_sync=False):
        """
        Flush all queued attachments to the Maxim API.


        This method empties the queue and sends all logs to the API,
        with special handling for AWS Lambda environments.

        """
        items = []
        while not self.upload_queue.empty():
            items.append(self.upload_queue.get())
        if len(items) == 0:
            scribe().debug("[MaximSDK] No attachments to flush")
            return
        scribe().debug(
            f"[MaximSDK] Flushing attachments to server {time.strftime('%Y-%m-%dT%H:%M:%S')} with {len(items)} items"
        )
        if self.is_running_on_lambda() or is_sync:
            # uploading attachments synchronously
            self.upload_attachments(items)
            # and once attached flushing commands again
            # as upload generates a few more commands
            self.flush_commit_logs(True)
        else:
            try:
                self.upload_executor.submit(self.upload_attachments, items)
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK] Error while flushing attachments from worker. Error: {e}.\nFlushing synchronously"
                )
                self.upload_attachments(items)
        scribe().debug(f"[MaximSDK] Flushed {len(items)} attachments")

    def flush_commit_logs(self, is_sync=False):
        """
        Flush all queued commit logs to the Maxim API.

        This method empties the queue and sends all logs to the API,
        with special handling for AWS Lambda environments.
        """
        items: list[CommitLog] = []
        while not self.queue.empty():
            items.append(self.queue.get())
        if len(items) == 0:
            self.flush_log_files()
            scribe().debug("[MaximSDK] No logs to flush")
            return
        scribe().debug(
            f"[MaximSDK] Flushing logs to server {time.strftime('%Y-%m-%dT%H:%M:%S')} with {len(items)} items"
        )
        for item in items:
            scribe().debug(f"[MaximSDK] {item.serialize()[:1000]}")
        # if we are running on lambda - we will flush without submitting to the executor
        if self.is_running_on_lambda() or is_sync:
            self.flush_logs(items)
        else:
            try:
                self.executor.submit(self.flush_logs, items)
            except Exception as e:
                scribe().warning(
                    f"[MaximSDK] Error while flushing logs from worker. Error: {e}.\nFlushing synchronously"
                )
                self.flush_logs(items)
        scribe().debug(f"[MaximSDK] Flushed {len(items)} logs")

    def flush(self, is_sync=False):
        """
        Flush all queued logs to the Maxim API.
        """
        self.flush_commit_logs(is_sync)
        self.flush_upload_attachment_logs(is_sync)

    def cleanup(self, is_sync=False):
        """
        Clean up resources used by the LogWriter.

        This method stops the flush thread, flushes any remaining logs,
        and shuts down the executor.
        """
        scribe().debug("[MaximSDK] Cleaning up writer")
        self.is_running = False
        self.flush(is_sync)
        scribe().debug("[MaximSDK] Waiting for executor to shutdown")
        self.executor.shutdown(wait=True)
        self.upload_executor.shutdown(wait=True)
        scribe().debug("[MaximSDK] Writer cleanup complete")
