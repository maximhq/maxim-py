#!/usr/bin/env python3
"""
Test case to simulate various connection errors and verify retry logic.

This test demonstrates how the improved connection pool and retry logic in maxim_apis.py
handles connection issues that users were experiencing, including:
- ConnectError and connection-related errors
- SSL/TLS errors
- Pool exhaustion errors
- General connection timeout errors
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
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
        self.assertIsNotNone(pool.client)

        # Verify client has basic httpx.Client functionality
        self.assertTrue(hasattr(pool.client, "request"))
        self.assertTrue(hasattr(pool.client, "timeout"))

        # Verify timeout configuration exists (values are set in the constructor)
        timeout = getattr(pool.client, "timeout", None)
        if timeout:
            self.assertEqual(timeout.connect, 15.0)
            self.assertEqual(timeout.read, 30.0)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_connect_error_retry_success(self, mock_sleep, mock_scribe):
        """Test that ConnectError errors are retried and eventually succeed."""

        # Create a mock response that succeeds on the third attempt
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt-id", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        # Create side effects: fail twice with ConnectError, then succeed
        connect_error = ConnectError("Connection failed")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First two calls fail, third succeeds
            mock_client.request.side_effect = [
                connect_error,  # First attempt fails
                connect_error,  # Second attempt fails
                mock_response,  # Third attempt succeeds
            ]

            # Test the retry functionality by calling a public method that will trigger retries
            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify the call eventually succeeded
            self.assertIsNotNone(result)
            self.assertEqual(result.promptId, "test-prompt-id")

            # Verify retries happened (3 total calls: 2 failures + 1 success)
            self.assertEqual(mock_client.request.call_count, 3)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_pool_timeout_retry_logic(self, mock_sleep, mock_scribe):
        """Test that PoolTimeout has separate retry logic with different parameters."""
        # Create mock response for eventual success
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        pool_timeout_error = PoolTimeout("Connection pool is full")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First call fails with PoolTimeout, second succeeds
            mock_client.request.side_effect = [pool_timeout_error, mock_response]

            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify eventual success
            self.assertIsNotNone(result)

            # Verify pool timeout specific debug was logged
            debug_calls = mock_scribe.return_value.debug.call_args_list
            pool_debug_found = any(
                "Connection pool exhausted" in str(call_args)
                for call_args in debug_calls
            )
            self.assertTrue(pool_debug_found)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_http_status_error_no_retry(self, mock_sleep, mock_scribe):
        """Test that HTTP status errors are not retried (permanent failures)."""
        # Create mock HTTP status error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Not found"}}

        http_status_error = HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = http_status_error

            # HTTP errors should not be retried and should raise immediately
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify only one attempt was made (no retries)
            self.assertEqual(mock_client.request.call_count, 1)

            # Verify the error message is properly formatted (JSON error message is extracted)
            self.assertIn("Not found", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_max_retries_exhausted(self, mock_sleep, mock_scribe):
        """Test behavior when max retries are exhausted."""
        connect_error = ConnectError("Persistent connection issue")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = connect_error

            # Should fail after max retries
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify max retries + 1 attempts were made (default is 3 retries + initial)
            expected_attempts = self.maxim_api.max_retries + 1
            self.assertEqual(mock_client.request.call_count, expected_attempts)

            # Verify error log was called
            self.assertTrue(mock_scribe.return_value.error.called)
            self.assertIn("Connection failed after", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep, mock_scribe):
        """Test that exponential backoff timing is correct."""
        connect_error = ConnectError("Connection issue")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = connect_error

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
    @patch("httpx.put")
    def test_file_upload_connect_error_retry(self, mock_put, mock_sleep, mock_scribe):
        """Test that file upload handles ConnectError errors with retry logic."""

        # Create mock response for successful upload
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        # Create connection error
        connect_error = ConnectError("Connection failed")

        # First two uploads fail, third succeeds
        mock_put.side_effect = [
            connect_error,  # First attempt fails
            connect_error,  # Second attempt fails
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

        # Verify debug logs for retries (file upload connection errors log debug messages)
        self.assertTrue(mock_scribe.return_value.debug.called)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    @patch("httpx.put")
    def test_file_upload_http_status_error_no_retry(
        self, mock_put, mock_sleep, mock_scribe
    ):
        """Test that file upload HTTP status errors are not retried."""
        # Create mock HTTP status error
        mock_response = Mock()
        mock_response.status_code = 403
        http_status_error = HTTPStatusError(
            "Forbidden", request=Mock(), response=mock_response
        )

        mock_put.side_effect = http_status_error

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
        connect_error = ConnectError("Connection failed")

        # Mock the client to return connection error first, then success
        success_response = Mock()
        success_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First call fails, second succeeds
            mock_client.request.side_effect = [connect_error, success_response]

            # Test that get_prompt method works despite connection error
            result = self.maxim_api.get_prompt("test-prompt-id")

            # Should succeed due to retry logic
            self.assertIsNotNone(result)
            self.assertEqual(result.promptId, "test-prompt")

    def test_different_connection_errors_are_caught(self):
        """Test that various connection-related errors are properly caught in our exception handling."""

        # List of connection errors that should be retried
        connection_errors = [
            ConnectError("Connection failed"),
            ConnectTimeout("Connection timeout"),
            ReadTimeout("Read timeout"),
            ProtocolError("Protocol error"),
            TimeoutException("Timeout occurred"),
            RequestError("Request error"),
        ]

        for error in connection_errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    self.maxim_api.connection_pool, "get_client"
                ) as mock_client_context:
                    mock_client = Mock()
                    mock_client_context.return_value.__enter__.return_value = (
                        mock_client
                    )
                    mock_client.request.side_effect = error

                    # Test that each error type is caught and handled
                    with self.assertRaises(Exception):
                        self.maxim_api.get_prompt("test-prompt-id")

    @patch("maxim.apis.maxim_apis.scribe")
    def test_unexpected_exception_handling(self, mock_scribe):
        """Test that unexpected exceptions are properly logged and re-raised."""
        unexpected_error = ValueError("Unexpected error")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = unexpected_error

            # Should re-raise unexpected exceptions after retrying
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify retries were attempted (max_retries + 1 attempts)
            expected_attempts = self.maxim_api.max_retries + 1
            self.assertEqual(mock_client.request.call_count, expected_attempts)

            # Verify debug logs during retries and error log at the end
            self.assertTrue(mock_scribe.return_value.debug.called)
            self.assertTrue(mock_scribe.return_value.error.called)
            self.assertIn("Unexpected error", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_request_error_retry_logic(self, mock_sleep, mock_scribe):
        """Test that general RequestError errors are retried."""
        request_error = RequestError("General request error")

        # Success response for eventual success
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First call fails with RequestError, second succeeds
            mock_client.request.side_effect = [request_error, mock_response]

            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify eventual success
            self.assertIsNotNone(result)

            # Verify retry debug was logged
            self.assertTrue(mock_scribe.return_value.debug.called)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_custom_retry_parameters(self, mock_sleep, mock_scribe):
        """Test retry logic with custom parameters."""
        connect_error = ConnectError("Connection issue")

        # Create custom API instance with different retry count
        custom_api = MaximAPI(self.base_url, self.api_key)
        custom_api.max_retries = 1

        with patch.object(
            custom_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = connect_error

            with self.assertRaises(Exception):
                custom_api.get_prompt("test-prompt-id")

            # Verify only 2 attempts were made (1 retry + initial)
            self.assertEqual(mock_client.request.call_count, 2)

    def test_connection_pool_client_context_manager(self):
        """Test that the connection pool context manager works correctly."""
        pool = ConnectionPool()

        # Test context manager functionality
        with pool.get_client() as client:
            self.assertIsNotNone(client)
            self.assertEqual(client, pool.client)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_version_check_during_retry(self, mock_sleep, mock_scribe):
        """Test that version checking works correctly during retry scenarios."""
        # Create connection error for first attempt
        connect_error = ConnectError("Connection failed")

        # Success response with version header
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {"x-lt-maxim-sdk-version": "2.0.0"}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First call fails, second succeeds with version header
            mock_client.request.side_effect = [connect_error, mock_response]

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

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_timeout_error_retry_success(self, mock_sleep, mock_scribe):
        """Test that TimeoutException errors are retried and eventually succeed."""
        # Create a mock response that succeeds on the third attempt
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt-id", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        # Create TimeoutException - this simulates timeout errors
        timeout_error = TimeoutException("Request timed out")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First two calls fail with TimeoutException, third succeeds
            mock_client.request.side_effect = [
                timeout_error,  # First attempt fails
                timeout_error,  # Second attempt fails
                mock_response,  # Third attempt succeeds
            ]

            # Test the retry functionality
            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify the call eventually succeeded
            self.assertIsNotNone(result)
            self.assertEqual(result.promptId, "test-prompt-id")

            # Verify retries happened (3 total calls: 2 failures + 1 success)
            self.assertEqual(mock_client.request.call_count, 3)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_timeout_error_retry_exhaustion(self, mock_sleep, mock_scribe):
        """Test TimeoutException behavior when max retries are exhausted."""
        timeout_error = TimeoutException("Request timed out")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = timeout_error

            # Should fail after max retries
            with self.assertRaises(Exception) as context:
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify max retries + 1 attempts were made
            expected_attempts = self.maxim_api.max_retries + 1
            self.assertEqual(mock_client.request.call_count, expected_attempts)

            # Verify error log was called and contains timeout information
            self.assertTrue(mock_scribe.return_value.error.called)
            self.assertIn("Connection failed after", str(context.exception))

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_read_timeout_retry_logic(self, mock_sleep, mock_scribe):
        """Test that ReadTimeout errors are retried."""
        # Create mock response for eventual success
        mock_response = Mock()
        mock_response.content = (
            b'{"data": {"promptId": "test-prompt", "rules": {}, "versions": []}}'
        )
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        # Test ReadTimeout
        read_timeout_error = ReadTimeout("Read timeout occurred")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client

            # First call fails with ReadTimeout, second succeeds
            mock_client.request.side_effect = [read_timeout_error, mock_response]

            result = self.maxim_api.get_prompt("test-prompt-id")

            # Verify eventual success
            self.assertIsNotNone(result)
            self.assertEqual(mock_client.request.call_count, 2)

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    @patch("httpx.put")
    def test_file_upload_timeout_error_retry(self, mock_put, mock_sleep, mock_scribe):
        """Test that file upload handles TimeoutException with retry logic."""
        # Create mock response for successful upload
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        # Create TimeoutException for upload
        timeout_error = TimeoutException("Request timed out")

        # First two uploads fail with timeout error, third succeeds
        mock_put.side_effect = [
            timeout_error,  # First attempt fails
            timeout_error,  # Second attempt fails
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

        # For file upload timeout errors, the retry logic logs debug messages during retries
        # Verify that debug logs were called for retries
        self.assertTrue(mock_scribe.return_value.debug.called)

    def test_httpx_errors_in_connection_error_list(self):
        """Test that httpx errors are properly caught in our exception handling."""
        # List of httpx errors that should be retried
        httpx_errors = [
            ConnectError("Connection failed"),
            ConnectTimeout("Connection timeout"),
            ReadTimeout("Read timeout"),
            ProtocolError("Protocol error"),
            TimeoutException("Timeout occurred"),
            RequestError("Request error"),
        ]

        for error in httpx_errors:
            with self.subTest(error=type(error).__name__):
                with patch.object(
                    self.maxim_api.connection_pool, "get_client"
                ) as mock_client_context:
                    mock_client = Mock()
                    mock_client_context.return_value.__enter__.return_value = (
                        mock_client
                    )
                    mock_client.request.side_effect = error

                    # Test that each httpx error type is caught and handled
                    with self.assertRaises(Exception):
                        self.maxim_api.get_prompt("test-prompt-id")

    @patch("maxim.apis.maxim_apis.scribe")
    @patch("time.sleep")
    def test_connect_timeout_with_exponential_backoff(self, mock_sleep, mock_scribe):
        """Test that ConnectTimeout errors use the same exponential backoff as other connection errors."""
        connect_timeout_error = ConnectTimeout("Connection timeout")

        with patch.object(
            self.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.side_effect = connect_timeout_error

            # Attempt the request (will fail after retries)
            with self.assertRaises(Exception):
                self.maxim_api.get_prompt("test-prompt-id")

            # Verify exponential backoff is used for timeout errors too
            expected_delays = [
                self.maxim_api.base_delay * (2**0),  # 1.0
                self.maxim_api.base_delay * (2**1),  # 2.0
                self.maxim_api.base_delay * (2**2),  # 4.0
            ]

            actual_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            self.assertEqual(actual_delays, expected_delays)


class TestFileUploadRetryLogic(unittest.TestCase):
    """Separate test class for file upload specific retry logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.maxim_api = MaximAPI("https://app.getmaxim.ai", "test-api-key")

    @patch("httpx.put")
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
        self.assertIsNotNone(call_kwargs["timeout"])
        # Check that timeout values are correctly set
        timeout = call_kwargs["timeout"]
        self.assertEqual(timeout.connect, 15.0)
        self.assertEqual(timeout.read, 60.0)

    @patch("httpx.put")
    @patch("time.sleep")
    @patch("maxim.apis.maxim_apis.scribe")
    def test_file_upload_retry_exhaustion(self, mock_scribe, mock_sleep, mock_put):
        """Test file upload behavior when all retries are exhausted."""
        connect_error = ConnectError("Persistent upload failure")
        mock_put.side_effect = connect_error

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
