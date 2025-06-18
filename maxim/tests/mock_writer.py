"""
Mock writer for testing purposes.

This module provides a mock implementation of LogWriter that captures
logs in memory for testing purposes.
"""

import threading
from queue import Queue
from unittest.mock import Mock
from maxim.logger import Logger
from maxim.logger.components.types import CommitLog
from maxim.logger.writer import LogWriterConfig


class MockLogWriter:
    """
    Mock implementation of LogWriter for testing.

    This class captures all logs and commands in memory, allowing tests
    to verify what was logged without making actual API calls.
    """

    def __init__(self, config: LogWriterConfig):
        """
        Initialize a MockLogWriter instance.

        Args:
            config: Configuration for the LogWriter.
        """
        self.config = config
        self.is_running = True
        self.queue = Queue()
        self.upload_queue = Queue()

        # Storage for captured logs
        self.committed_logs: list[CommitLog] = []
        self.flushed_logs: list[CommitLog] = []
        self.uploaded_attachments: list[CommitLog] = []

        # Counters for verification
        self.commit_count = 0
        self.flush_count = 0
        self.cleanup_count = 0

        # Mock API for testing
        self.maxim_api = Mock()

        # Thread safety
        self._lock = threading.Lock()

    @property
    def repository_id(self):
        """Get the repository ID."""
        return self.config.repository_id

    def commit(self, log: CommitLog):
        """
        Mock commit that stores the log instead of queuing it.

        Args:
            log: CommitLog object to store.
        """
        with self._lock:
            self.committed_logs.append(log)
            self.commit_count += 1

            # Handle upload attachments separately like the real writer
            if log.action == "upload-attachment":
                if log.data is not None:
                    repo_id = self.config.repository_id
                    entity_id = log.entity_id
                    file_id = log.data["id"]
                    key = f"{repo_id}/{log.entity.value}/{entity_id}/files/original/{file_id}"
                    log.data["key"] = key
                    self.upload_queue.put(log)
            else:
                self.queue.put(log)

    def flush(self, is_sync=False):
        """
        Mock flush that moves queued logs to flushed logs.

        Args:
            is_sync: Whether to flush synchronously (ignored in mock).
        """
        with self._lock:
            self.flush_count += 1

            # Move commit logs from queue to flushed
            while not self.queue.empty():
                log = self.queue.get()
                self.flushed_logs.append(log)

            # Move upload logs from queue to uploaded
            while not self.upload_queue.empty():
                log = self.upload_queue.get()
                self.uploaded_attachments.append(log)

    def flush_commit_logs(self, is_sync=False):
        """Mock flush for commit logs only."""
        with self._lock:
            while not self.queue.empty():
                log = self.queue.get()
                self.flushed_logs.append(log)

    def flush_upload_attachment_logs(self, is_sync=False):
        """Mock flush for upload attachment logs only."""
        with self._lock:
            while not self.upload_queue.empty():
                log = self.upload_queue.get()
                self.uploaded_attachments.append(log)

    def cleanup(self, is_sync=False):
        """
        Mock cleanup that flushes remaining logs.

        Args:
            is_sync: Whether to cleanup synchronously (ignored in mock).
        """
        self.cleanup_count += 1
        self.is_running = False
        self.flush(is_sync)

    # Verification methods for tests

    def get_all_logs(self) -> list[CommitLog]:
        """Get all logs (committed, flushed, and uploaded)."""
        with self._lock:
            return self.committed_logs + self.flushed_logs + self.uploaded_attachments

    def get_committed_logs(self):
        """Get only committed logs."""
        with self._lock:
            return self.committed_logs.copy()

    def get_flushed_logs(self):
        """Get only flushed logs."""
        with self._lock:
            return self.flushed_logs.copy()

    def get_uploaded_attachments(self):
        """Get only uploaded attachment logs."""
        with self._lock:
            return self.uploaded_attachments.copy()

    def get_logs_by_action(self, action):
        """Get logs filtered by action type."""
        with self._lock:
            return [log for log in self.flushed_logs if log.action == action]

    def get_logs_by_entity_action(self, entity, action):
        """Get logs filtered by entity and action type."""
        with self._lock:
            return [
                log
                for log in self.flushed_logs
                if log.entity.value == entity and log.action == action
            ]

    def get_logs_by_entity(self, entity):
        """Get logs filtered by entity type."""
        with self._lock:
            return [log for log in self.flushed_logs if log.entity.value == entity]

    def get_logs_by_entity_and_action(self, entity, action):
        """Get logs filtered by entity and action type."""
        with self._lock:
            return [
                log
                for log in self.flushed_logs
                if log.entity.value == entity and log.action == action
            ]

    def get_logs_by_entity_id(self, entity_id):
        """Get logs filtered by entity ID."""
        with self._lock:
            return [log for log in self.flushed_logs if log.entity_id == entity_id]

    def clear_logs(self):
        """Clear all stored logs."""
        with self._lock:
            self.committed_logs.clear()
            self.flushed_logs.clear()
            self.uploaded_attachments.clear()
            self.commit_count = 0
            self.flush_count = 0
            self.cleanup_count = 0

    def assert_log_count(self, expected_count):
        """Assert the total number of logs."""
        actual_count = len(self.get_all_logs())
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} logs, got {actual_count}"

    def assert_entity_action_count(self, entity, action, expected_count):
        """Assert the number of logs for a specific entity and action."""
        logs = self.get_logs_by_entity_action(entity, action)
        actual_count = len(logs)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} '{entity} {action}' logs, got {actual_count}"

    def assert_action_count(self, action, expected_count):
        """Assert the number of logs for a specific action."""
        logs = self.get_logs_by_action(action)
        actual_count = len(logs)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} '{action}' logs, got {actual_count}"

    def assert_entity_count(self, entity, expected_count):
        """Assert the number of logs for a specific entity."""
        logs = self.get_logs_by_entity(entity)
        actual_count = len(logs)
        assert (
            actual_count == expected_count
        ), f"Expected {expected_count} '{entity}' logs, got {actual_count}"

    def print_logs_summary(self):
        """Print a summary of all captured logs for debugging."""
        print(f"\n=== Mock Writer Log Summary ===")
        print(f"Total logs: {len(self.get_all_logs())}")
        print(f"Committed: {len(self.committed_logs)}")
        print(f"Flushed: {len(self.flushed_logs)}")
        print(f"Uploaded: {len(self.uploaded_attachments)}")
        print(f"Commit calls: {self.commit_count}")
        print(f"Flush calls: {self.flush_count}")
        print(f"Cleanup calls: {self.cleanup_count}")

        actions = {}
        entities = {}
        for log in self.get_all_logs():
            actions[log.action] = actions.get(log.action, 0) + 1
            entities[log.entity.value] = entities.get(log.entity.value, 0) + 1

        print(f"Actions: {actions}")
        print(f"Entities: {entities}")
        print("================================\n")


def inject_mock_writer(logger: Logger):
    if hasattr(logger, "writer"):
        delattr(logger, "writer")
    # Create mock writer config
    mock_writer_config = LogWriterConfig(
        base_url="https://app.getmaxim.ai",
        api_key="test-api-key",
        repository_id="test-repo-id",
        auto_flush=False,  # Disable auto-flush for testing
        flush_interval=None,
        is_debug=True,
        raise_exceptions=True,
    )
    # Create mock writer
    mock_writer = MockLogWriter(mock_writer_config)
    setattr(logger, "writer", mock_writer)
    return mock_writer
