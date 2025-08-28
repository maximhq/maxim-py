import os
import unittest
import json
import tempfile
from unittest.mock import Mock, patch

from maxim import Maxim, Config
from maxim.models.dataset import DatasetEntry, Variable, DatasetEntryWithRowNo, FileVariablePayload, VariableFileAttachment
from maxim.logger.components.attachment import FileAttachment, FileDataAttachment, UrlAttachment
from dotenv import load_dotenv
load_dotenv()

MAXIM_BASE_URL = os.getenv("MAXIM_BASE_URL")
MAXIM_API_KEY = os.getenv("MAXIM_API_KEY")

class TestAddDatasetEntriesComprehensive(unittest.TestCase):
    """Comprehensive test suite for the updated add_dataset_entries method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        config = Config(
            api_key=MAXIM_API_KEY,
            base_url=MAXIM_BASE_URL,
            debug=True,
            raise_exceptions=True
        )
        self.maxim = Maxim(config)
        self.dataset_id = "test-dataset-id"

    def tearDown(self) -> None:
        """Clean up after tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        if "MAXIM_API_KEY" in os.environ:
            del os.environ["MAXIM_API_KEY"]
        if "MAXIM_BASE_URL" in os.environ:
            del os.environ["MAXIM_BASE_URL"]

    def _create_mock_response(self, content: bytes, headers: dict = None) -> Mock:
        """Create a mock HTTP response."""
        mock_response = Mock()
        mock_response.content = content
        mock_response.headers = headers or {}
        mock_response.raise_for_status.return_value = None
        return mock_response

    def _setup_mock_network_calls(self, responses: list) -> Mock:
        """Set up mock network calls that return specified responses in order."""
        mock_client = Mock()
        mock_client.request.side_effect = responses
        
        # Create a context manager mock
        mock_client_context = Mock()
        mock_client_context.__enter__ = Mock(return_value=mock_client)
        mock_client_context.__exit__ = Mock(return_value=None)
        
        # Patch the connection pool's get_client method
        patcher = patch.object(
            self.maxim.maxim_api.connection_pool, "get_client", 
            return_value=mock_client_context
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return mock_client

    def test_add_dataset_entries_with_dict_input(self) -> None:
        """Test add_dataset_entries with dictionary input."""
        # Mock responses for: get_dataset_total_rows, add_dataset_entries
        mock_total_rows_response = self._create_mock_response(b'{"data": 5}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry1", "entry2"], "cells": []}}'
        )
        
        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response
        ])

        # Test data with various types
        entries = [
            {
                "input": "Test input text",
                "expected_output": {"result": "success"},
                "context": "Test context"
            }
        ]

        self.maxim.add_dataset_entries(self.dataset_id, entries)

        # Verify correct number of network calls
        self.assertEqual(mock_client.request.call_count, 2)
        
        # Verify get_dataset_total_rows call
        first_call = mock_client.request.call_args_list[0]
        self.assertEqual(first_call[1]["method"], "GET")
        self.assertIn("total-rows", first_call[1]["url"])
        
        # Verify add_dataset_entries call
        second_call = mock_client.request.call_args_list[1]
        self.assertEqual(second_call[1]["method"], "POST")
        self.assertIn("datasets/entries", second_call[1]["url"])
        
        # Verify request body structure
        request_content = second_call[1]["content"]
        parsed_body = json.loads(request_content)
        self.assertIn("datasetId", parsed_body)
        self.assertIn("entries", parsed_body)
        self.assertEqual(len(parsed_body["entries"]), 3)  # One entry for each column
        
        # Verify DatasetEntryWithRowNo structure (updated format)
        for entry in parsed_body["entries"]:
            self.assertIn("rowNo", entry)
            self.assertIn("columnName", entry)
            self.assertIn("type", entry)
            self.assertIn("value", entry)
            self.assertEqual(entry["rowNo"], 6)  # 5 existing rows + 1

    def test_add_dataset_entries_with_dataset_entry_objects(self) -> None:
        """Test add_dataset_entries with DatasetEntry objects."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 0}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry1", "entry2"], "cells": []}}'
        )
        
        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response
        ])

        # Create DatasetEntry objects
        entry = DatasetEntry(entry={
            "input": Variable(type="text", payload="Test input"),
            "output": Variable(type="json", payload={"status": "complete"}),
        })

        self.maxim.add_dataset_entries(self.dataset_id, [entry])

        # Verify DatasetEntry was properly converted
        second_call = mock_client.request.call_args_list[1]
        request_content = second_call[1]["content"]
        parsed_body = json.loads(request_content)
        self.assertIn("entries", parsed_body)
        self.assertEqual(len(parsed_body["entries"]), 2)  # Two columns

    def test_add_dataset_entries_with_file_attachments(self) -> None:
        """Test add_dataset_entries with file attachments."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 0}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry123"], "cells": [{"entryId": "entry123", "columnName": "files", "rowNo": 1}]}}'
        )
        mock_upload_url_response = self._create_mock_response(
            b'{"data": {"url": "https://signed-url.com", "key": "datasets/test-dataset-id/entry123/test-file-key"}}'
        )
        mock_patch_response = self._create_mock_response(b'{"data": {"message": "Entries updated successfully"}}')
        
        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response,
            mock_upload_url_response,
            mock_patch_response
        ])

        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Test file content")
            temp_file_path = temp_file.name

        try:
            # Create entry with file attachment
            file_attachment = FileAttachment(
                path=temp_file_path,
                name="test.txt",
                mime_type="text/plain"
            )

            entry = DatasetEntry(entry={
                "input": Variable(type="text", payload="Input with file"),
                "files": Variable(type="file", payload=[file_attachment])
            })

            with patch.object(self.maxim.maxim_api, 'upload_to_signed_url', return_value=True):
                self.maxim.add_dataset_entries(self.dataset_id, [entry])

            # Verify file upload process was triggered
            self.assertEqual(mock_client.request.call_count, 4)  # total_rows + entries + upload_url + patch

        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    def test_add_dataset_entries_with_url_attachment(self) -> None:
        """Test add_dataset_entries with URL attachment."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 0}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry123"], "cells": [{"entryId": "entry123", "columnName": "images", "rowNo": 1}]}}'
        )
        mock_upload_url_response = self._create_mock_response(
            b'{"data": {"url": "https://signed-url.com", "key": "datasets/test-dataset-id/entry123/test-file-key"}}'
        )
        mock_patch_response = self._create_mock_response(b'{"success": true}')

        self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response,
            mock_upload_url_response,
            mock_patch_response
        ])

        # Create entry with URL attachment
        url_attachment = UrlAttachment(
            url="https://example.com/image.jpg",
            name="image.jpg",
            mime_type="image/jpeg"
        )

        entry = DatasetEntry(entry={
            "input": Variable(type="text", payload="Input with URL"),
            "images": Variable(type="file", payload=[url_attachment])
        })

        # Mock the URL download
        with patch.object(self.maxim.maxim_api, '_process_url_attachment') as mock_process_url:
            mock_process_url.return_value = (b"fake image data", "image/jpeg", 1024)
            with patch.object(self.maxim.maxim_api, 'upload_to_signed_url', return_value=True):
                self.maxim.add_dataset_entries(self.dataset_id, [entry])

        # Verify URL processing was called
        mock_process_url.assert_called_once_with(url_attachment)

    def test_add_dataset_entries_with_file_data_attachment(self) -> None:
        """Test add_dataset_entries with FileDataAttachment."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 0}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry123"], "cells": [{"entryId": "entry123", "columnName": "data", "rowNo": 1}]}}'
        )
        mock_upload_url_response = self._create_mock_response(
            b'{"data": {"url": "https://signed-url.com", "key": "datasets/test-dataset-id/entry123/test-file-key"}}'
        )
        mock_patch_response = self._create_mock_response(b'{"success": true}')

        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response,
            mock_upload_url_response,
            mock_patch_response
        ])

        # Create entry with file data attachment
        file_data_attachment = FileDataAttachment(
            data=b"Binary file content",
            name="data.bin",
            mime_type="application/octet-stream"
        )

        entry = DatasetEntry(entry={
            "input": Variable(type="text", payload="Input with file data"),
            "data": Variable(type="file", payload=[file_data_attachment])
        })

        with patch.object(self.maxim.maxim_api, 'upload_to_signed_url', return_value=True):
            self.maxim.add_dataset_entries(self.dataset_id, [entry])

        # Verify the upload process was initiated
        self.assertEqual(mock_client.request.call_count, 4)

    def test_add_dataset_entries_mixed_input_types(self) -> None:
        """Test add_dataset_entries with mixed DatasetEntry and dict inputs."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 10}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry1", "entry2"], "cells": []}}'
        )

        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response
        ])

        # Mixed input types
        dataset_entry = DatasetEntry(entry={
            "input": Variable(type="text", payload="Object input"),
            "output": Variable(type="json", payload={"result": "object"}),
        })

        dict_entry = {
            "input": "Dict input",
            "output": {"result": "dict"}
        }

        self.maxim.add_dataset_entries(
            self.dataset_id, [dataset_entry, dict_entry]
        )

        # Verify both entries were processed
        second_call = mock_client.request.call_args_list[1]
        request_content = second_call[1]["content"]
        parsed_body = json.loads(request_content)
        
        # Should have entries for both DatasetEntry and dict inputs
        # Each entry has 2 columns, so 4 total DatasetEntryWithRowNo objects
        self.assertIn("entries", parsed_body)
        self.assertEqual(len(parsed_body["entries"]), 4)
        
        # Check row numbers are correctly assigned
        row_numbers = [entry["rowNo"] for entry in parsed_body["entries"]]
        self.assertEqual(set(row_numbers), {11, 12})  # 10 existing + 2 new entries

    def test_add_dataset_entries_invalid_entry_type(self) -> None:
        """Test that invalid entry types raise TypeError."""
        # Mock the initial call to get_dataset_total_rows
        mock_total_rows_response = self._create_mock_response(b'{"data": 0}')
        self._setup_mock_network_calls([mock_total_rows_response])
        
        with self.assertRaises(TypeError) as context:
            self.maxim.add_dataset_entries(
                self.dataset_id,
                ["invalid_entry_type"]  # String instead of dict or DatasetEntry
            )
        
        self.assertIn("Invalid entry type", str(context.exception))

    def test_add_dataset_entries_empty_list(self) -> None:
        """Test add_dataset_entries with empty list."""
        mock_total_rows_response = self._create_mock_response(b'{"data": 5}')
        mock_add_entries_response = self._create_mock_response(
            b'{"data": {"ids": ["entry1", "entry2"], "cells": []}}'
        )
        
        mock_client = self._setup_mock_network_calls([
            mock_total_rows_response,
            mock_add_entries_response
        ])

        self.maxim.add_dataset_entries(self.dataset_id, [])

        # Verify request was made with empty entries
        second_call = mock_client.request.call_args_list[1]
        request_content = second_call[1]["content"]
        parsed_body = json.loads(request_content)
        self.assertIn("entries", parsed_body)
        self.assertEqual(parsed_body["entries"], [])

    def test_upload_dataset_entry_attachments_file_size_validation(self) -> None:
        """Test file size validation in upload_dataset_entry_attachments."""
        # Create large file attachment that exceeds limit
        large_data = b"x" * (101 * 1024 * 1024)  # 101MB
        large_attachment = FileDataAttachment(
            data=large_data,
            name="large_file.bin",
            mime_type="application/octet-stream"
        )
        
        entry_with_row_no = DatasetEntryWithRowNo(
            row_no=1,
            column_name="large_files",
            type="file",
            payload=[large_attachment]
        )

        # The method catches the ValueError and logs it, then returns None
        # instead of raising the exception
        result = self.maxim.maxim_api.upload_dataset_entry_attachments(
            self.dataset_id, "entry123", entry_with_row_no
        )
        
        # Should return None because the file was too large and was skipped
        self.assertIsNone(result)

    def test_upload_dataset_entry_attachments_mime_type_inference(self) -> None:
        """Test MIME type inference for attachments without explicit type."""
        mock_upload_url_response = self._create_mock_response(
            b'{"data": {"url": "https://signed-url.com", "key": "datasets/test-dataset-id/entry123/test-file-key"}}'
        )
        
        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client.request.return_value = mock_upload_url_response
            mock_client_context.return_value.__enter__.return_value = mock_client
            
            # Create attachment without explicit MIME type
            attachment = FileDataAttachment(
                data=b"test content",
                name="test.txt"  # Should infer text/plain
            )
            
            entry_with_row_no = DatasetEntryWithRowNo(
                row_no=1,
                column_name="files",
                type="file",
                payload=[attachment]
            )

            with patch.object(self.maxim.maxim_api, 'upload_to_signed_url', return_value=True):
                file_payload = self.maxim.maxim_api.upload_dataset_entry_attachments(
                    self.dataset_id, "entry123", entry_with_row_no
                )

            # Verify MIME type was inferred or handled properly
            self.assertIsNotNone(file_payload)

    def test_process_attachment_invalid_type(self) -> None:
        """Test _process_attachment with invalid attachment type."""
        invalid_attachment = "not_an_attachment"
        
        with self.assertRaises(TypeError) as context:
            self.maxim.maxim_api._process_attachment(invalid_attachment)
        
        self.assertIn("Invalid attachment type", str(context.exception))

    def test_process_url_attachment_invalid_url(self) -> None:
        """Test _process_url_attachment with invalid URL."""
        invalid_url_attachment = UrlAttachment(url="invalid-url")
        
        with self.assertRaises(Exception) as context:
            self.maxim.maxim_api._process_url_attachment(invalid_url_attachment)
        
        self.assertIn("Invalid URL", str(context.exception))

    def test_process_file_attachment_missing_file(self) -> None:
        """Test _process_file_attachment with missing file."""
        missing_file_attachment = FileAttachment(path="/nonexistent/file.txt")
        
        with self.assertRaises(Exception) as context:
            self.maxim.maxim_api._process_file_attachment(missing_file_attachment)
        
        self.assertIn("File not found", str(context.exception))

    def test_add_dataset_entries_network_error_handling(self) -> None:
        """Test error handling when network calls fail."""
        # Mock a network error for the total rows call
        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client"
        ) as mock_client_context:
            mock_client = Mock()
            mock_client.request.side_effect = Exception("Network error")
            mock_client_context.return_value.__enter__.return_value = mock_client
            
            with self.assertRaises(Exception) as context:
                self.maxim.add_dataset_entries(
                    self.dataset_id, [{"input": "test"}]
                )
            
            self.assertIn("Network error", str(context.exception))

    def test_dataset_entry_with_row_no_to_dict_file_type(self) -> None:
        """Test DatasetEntryWithRowNo.to_dict() for file type returns empty payload."""
        file_attachment = FileAttachment(path="test.txt")
        entry = DatasetEntryWithRowNo(
            row_no=1,
            column_name="files",
            type="file",
            payload=[file_attachment]
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result["type"], "file")  # File type mapped to attachment
        self.assertEqual(result["value"], [])  # Should be empty for file types

    def test_dataset_entry_with_row_no_to_dict_non_file_type(self) -> None:
        """Test DatasetEntryWithRowNo.to_dict() for non-file types preserves payload."""
        entry = DatasetEntryWithRowNo(
            row_no=1,
            column_name="input",
            type="text",
            payload="test content"
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["value"], "test content")

    def test_file_variable_payload_creation(self) -> None:
        """Test FileVariablePayload and VariableFileAttachment creation."""
        file_attachment = VariableFileAttachment(
            id="file123",
            url="https://storage.com/file.txt",
            hosted=True,
            prefix="datasets/entry123/",
            props={"size": 1024}
        )
        
        payload = FileVariablePayload(
            text="files",
            files=[file_attachment],
            entry_id="entry123"
        )
        
        self.assertEqual(payload.entry_id, "entry123")
        self.assertEqual(payload.text, "files")
        self.assertEqual(len(payload.files), 1)
        self.assertEqual(payload.files[0].id, "file123")
        self.assertTrue(payload.files[0].hosted)


if __name__ == "__main__":
    unittest.main()