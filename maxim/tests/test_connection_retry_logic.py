#!/usr/bin/env python3
"""
Test case to simulate RemoteDisconnected connection errors and verify retry logic.

This test demonstrates how the improved connection pool and retry logic in maxim_apis.py
handles connection issues that users were experiencing.
"""

import json
import time
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from urllib3.exceptions import ProtocolError, MaxRetryError, PoolError
from requests.exceptions import (
    ConnectionError,
    RequestException,
    ConnectTimeout,
    ReadTimeout,
    HTTPError,
)
from http.client import RemoteDisconnected

# Add the maxim package to the path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from maxim.apis.maxim_apis import MaximAPI, ConnectionPool


class TestConnectionRetryLogic(unittest.TestCase):
    """Test cases for connection retry logic improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "https://app.getmaxim.ai"
        self.api_key = "test-api-key"
        self.maxim_api = MaximAPI(self.base_url, self.api_key)

    def test_connection_pool_configuration(self):
        """Test that connection pool is configured with improved settings."""
        pool = ConnectionPool()

        # Verify pool is created (basic check)
        self.assertIsNotNone(pool.session)

        # Verify session has adapters mounted for retry functionality
        self.assertIn("https://", pool.session.adapters)
        self.assertIn("http://", pool.session.adapters)

        # Verify adapter configuration
        https_adapter = pool.session.adapters["https://"]
        self.assertIsNotNone(https_adapter)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_remote_disconnected_retry_success(self, mock_sleep, mock_scribe):
        """Test that RemoteDisconnected errors are retried and eventually succeed."""

        # Create a mock response that succeeds on the third attempt
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt-id", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        # Create side effects: fail twice with RemoteDisconnected, then succeed
        connection_error = ConnectionError(
            "Connection aborted",
            RemoteDisconnected("Remote end closed connection without response"),
        )

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session

            # First two calls fail, third succeeds
            mock_session.request.side_effect = [
                connection_error,  # First attempt fails
                connection_error,  # Second attempt fails
                mock_response,  # Third attempt succeeds
            ]

            # Test the retry functionality by calling a public method that will trigger retries
            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify the call eventually succeeded
            self.assertIsNotNone(result)
            self.assertEqual(result.promptId, "test-prompt-id")

            # Verify retries happened (3 total calls: 2 failures + 1 success)
            self.assertEqual(mock_session.request.call_count, 3)

            # Verify warning logs were called for retries
            self.assertTrue(mock_scribe.return_value.warning.called)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_pool_error_retry_logic(self, mock_sleep, mock_scribe):
        """Test that PoolError has separate retry logic with different parameters."""
        # Create mock response for eventual success
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        pool_error = PoolError(None, "Connection pool is full")

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session

            # First call fails with PoolError, second succeeds
            mock_session.request.side_effect = [pool_error, mock_response]

            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify eventual success
            self.assertIsNotNone(result)

            # Verify pool error specific warning was logged
            warning_calls = mock_scribe.return_value.warning.call_args_list
            pool_warning_found = any(
                "Connection pool exhausted" in str(call_args)
                for call_args in warning_calls
            )
            self.assertTrue(pool_warning_found)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_http_error_no_retry(self, mock_sleep, mock_scribe):
        """Test that HTTP errors are not retried (permanent failures)."""
        # Create mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not found"}}

        http_error = HTTPError(response=mock_response)

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session.request.side_effect = http_error

            # HTTP errors should not be retried and should raise immediately
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify only one attempt was made (no retries)
            self.assertEqual(mock_session.request.call_count, 1)

            # Verify the error message is properly formatted
            self.assertIn("Not found", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_max_retries_exhausted(self, mock_sleep, mock_scribe):
        """Test behavior when max retries are exhausted."""
        connection_error = ConnectionError("Persistent connection issue")

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session.request.side_effect = connection_error

            # Should fail after max retries
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify max retries + 1 attempts were made (default is 3 retries + initial)
            expected_attempts = self.maxim_api.max_retries + 1
            self.assertEqual(mock_session.request.call_count, expected_attempts)

            # Verify error log was called
            self.assertTrue(mock_scribe.return_value.error.called)
            self.assertIn("Connection failed after", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep, mock_scribe):
        """Test that exponential backoff timing is correct."""
        connection_error = ConnectionError("Connection issue")

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session.request.side_effect = connection_error

            # Attempt the request (will fail after retries)
            with self.assertRaises(Exception):
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify exponential backoff: 1.0, 2.0, 4.0 seconds
            expected_delays = [
                self.maxim_api.base_delay * (2**0),  # 1.0
                self.maxim_api.base_delay * (2**1),  # 2.0
                self.maxim_api.base_delay * (2**2),  # 4.0
            ]

            actual_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            self.assertEqual(actual_delays, expected_delays)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    @patch("requests.put")
    def test_file_upload_remote_disconnected_retry(
        self, mock_put, mock_sleep, mock_scribe
    ):
        """Test that file upload handles RemoteDisconnected errors with retry logic."""

        # Create mock response for successful upload
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        # Create connection error
        connection_error = ConnectionError(
            "Connection aborted",
            RemoteDisconnected("Remote end closed connection without response"),
        )

        # First two uploads fail, third succeeds
        mock_put.side_effect = [
            connection_error,  # First attempt fails
            connection_error,  # Second attempt fails
            mock_response,  # Third attempt succeeds
        ]

        # Test file upload
        url = "https://signed-upload-url.amazonaws.com/test"
        data = b"test file content"
        mime_type = "text/plain"

        result = self.maxim_api.upload_to_signed_url(url, data, mime_type)

        # Verify upload eventually succeeded
        self.assertTrue(result)

        # Verify retries happened
        self.assertEqual(mock_put.call_count, 3)

        # Verify warning logs for retries
        self.assertTrue(mock_scribe.return_value.warning.called)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    @patch("requests.put")
    def test_file_upload_http_error_no_retry(self, mock_put, mock_sleep, mock_scribe):
        """Test that file upload HTTP errors are not retried."""
        # Create mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 403
        http_error = HTTPError(response=mock_response)

        mock_put.side_effect = http_error

        url = "https://signed-upload-url.amazonaws.com/test"
        data = b"test file content"
        mime_type = "text/plain"

        # Should fail immediately without retries
        with self.assertRaises(Exception) as context:
            self.maxim_api.upload_to_signed_url(url, data, mime_type)

        # Verify only one attempt was made
        self.assertEqual(mock_put.call_count, 1)
        self.assertIn("Client response error", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    def test_api_method_with_connection_error(self, mock_scribe):
        """Test that actual API methods handle connection errors properly."""

        # Create connection error that will be retried
        connection_error = ConnectionError(
            "Connection aborted",
            RemoteDisconnected("Remote end closed connection without response"),
        )

        # Mock the session to return connection error first, then success
        success_response = Mock()
        success_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session

            # First call fails, second succeeds
            mock_session.request.side_effect = [connection_error, success_response]

            # Test that get_prompt method works despite connection error
            result = self.maxim_api.get_prompt("test-prompt-id")

            # Should succeed due to retry logic
            self.assertIsNotNone(result)
            self.assertEqual(result.promptId, "test-prompt")

    def test_different_connection_errors_are_caught(self):
        """Test that various connection-related errors are properly caught in our exception handling."""

        # List of connection errors that should be retried
        connection_errors = [
            ConnectionError("Connection failed"),
            ConnectTimeout("Connection timeout"),
            ReadTimeout("Read timeout"),
            ProtocolError("Protocol error"),
            MaxRetryError(None, None, "Max retries exceeded"),
        ]

        for error in connection_errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    self.maxim_api.connection_pool, "get_session"
                ) as mock_session_context:
                    mock_session = Mock()
                    mock_session_context.return_value.__enter__.return_value = (
                        mock_session
                    )
                    mock_session.request.side_effect = error

                    # Test that each error type is caught and handled
                    with self.assertRaises(Exception):
                        self.maxim_api.get_prompt("test-prompt-id")

    @patch("maxim.apis.maxim_apis.scribe")
    def test_unexpected_exception_handling(self, mock_scribe):
        """Test that unexpected exceptions are properly logged and re-raised."""
        unexpected_error = ValueError("Unexpected error")

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session.request.side_effect = unexpected_error

            # Should re-raise unexpected exceptions immediately
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify only one attempt was made (no retries for unexpected errors)
            self.assertEqual(mock_session.request.call_count, 1)

            # Verify error was logged
            self.assertTrue(mock_scribe.return_value.error.called)
            self.assertIn("Unexpected error", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_request_exception_retry_logic(self, mock_sleep, mock_scribe):
        """Test that general RequestException errors are retried."""
        request_error = RequestException("General request error")

        # Success response for eventual success
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session

            # First call fails with RequestException, second succeeds
            mock_session.request.side_effect = [request_error, mock_response]

            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify eventual success
            self.assertIsNotNone(result)

            # Verify retry warning was logged
            self.assertTrue(mock_scribe.return_value.warning.called)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_custom_retry_parameters(self, mock_sleep, mock_scribe):
        """Test retry logic with custom parameters."""
        connection_error = ConnectionError("Connection issue")

        # Create custom API instance with different retry count
        custom_api = MaximAPI(self.base_url, self.api_key)
        custom_api.max_retries = 1

        with patch.object(
            custom_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session
            mock_session.request.side_effect = connection_error

            with self.assertRaises(Exception):
                custom_api.get_prompt("test-prompt-id")

            # Verify only 2 attempts were made (1 retry + initial)
            self.assertEqual(mock_session.request.call_count, 2)

    def test_connection_pool_session_context_manager(self):
        """Test that the connection pool context manager works correctly."""
        pool = ConnectionPool()

        # Test context manager functionality
        with pool.get_session() as session:
            self.assertIsNotNone(session)
            self.assertEqual(session, pool.session)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_version_check_during_retry(self, mock_sleep, mock_scribe):
        """Test that version checking works correctly during retry scenarios."""
        # Create connection error for first attempt
        connection_error = ConnectionError("Connection failed")

        # Success response with version header
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {"x-lt-maxim-sdk-version": "2.0.0"}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_session"
        ) as mock_session_context:
            mock_session = Mock()
            mock_session_context.return_value.__enter__.return_value = mock_session

            # First call fails, second succeeds with version header
            mock_session.request.side_effect = [connection_error, mock_response]

            with patch("maxim.apis.maxim_apis.current_version", "1.0.0"):
                result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify success and version warning
            self.assertIsNotNone(result)

            # Check if version warning was logged
            warning_calls = mock_scribe.return_value.warning.call_args_list
            version_warning_found = any(
                "SDK version is out of date" in str(call_args)
                for call_args in warning_calls
            )
            self.assertTrue(version_warning_found)


class TestFileUploadRetryLogic(unittest.TestCase):
    """Separate test class for file upload specific retry logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.maxim_api = MaximAPI("https://app.getmaxim.ai", "test-api-key")

    @patch("requests.put")
    @patch("time.sleep")
    @patch("maxim.apis.maxim_apis.scribe")
    def test_file_upload_extended_timeout(self, mock_scribe, mock_sleep, mock_put):
        """Test that file uploads use extended timeouts."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        url = "https://signed-upload-url.amazonaws.com/test"
        data = b"test file content"
        mime_type = "text/plain"

        result = self.maxim_api.upload_to_signed_url(url, data, mime_type)

        # Verify upload succeeded
        self.assertTrue(result)

        # Verify extended timeout was used
        mock_put.assert_called_once()
        call_kwargs = mock_put.call_args[1]
        self.assertEqual(call_kwargs["timeout"], (15, 60))

    @patch("requests.put")
    @patch("time.sleep")
    @patch("maxim.apis.maxim_apis.scribe")
    def test_file_upload_retry_exhaustion(self, mock_scribe, mock_sleep, mock_put):
        """Test file upload behavior when all retries are exhausted."""
        connection_error = ConnectionError("Persistent upload failure")
        mock_put.side_effect = connection_error

        url = "https://signed-upload-url.amazonaws.com/test"
        data = b"test file content"
        mime_type = "text/plain"

        with self.assertRaises(Exception) as context:
            self.maxim_api.upload_to_signed_url(url, data, mime_type)

        # Verify all retries were attempted (3 retries + 1 initial = 4 total)
        self.assertEqual(mock_put.call_count, 4)

        # Verify error message includes retry information
        self.assertIn(
            "File upload connection failed after 4 attempts", str(context.exception)
        )

        # Verify error log was called
        self.assertTrue(mock_scribe.return_value.error.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
