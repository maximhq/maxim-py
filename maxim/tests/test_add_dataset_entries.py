import os
import unittest
import json
from unittest.mock import Mock, patch

from maxim import Maxim
from maxim.models import DatasetEntry, Variable


class TestAddDatasetEntries(unittest.TestCase):
    """Comprehensive test for add_dataset_entries function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        os.environ["MAXIM_API_KEY"] = "test-api-key"
        os.environ["MAXIM_DATASET_ID"] = "test-dataset-id"
        self.maxim = Maxim()

    def tearDown(self) -> None:
        """Clean up after tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        if "MAXIM_API_KEY" in os.environ:
            del os.environ["MAXIM_API_KEY"]
        if "MAXIM_DATASET_ID" in os.environ:
            del os.environ["MAXIM_DATASET_ID"]

    def test_add_dataset_entries_with_dict_input(self) -> None:
        """Test the add_dataset_entries function using dictionary input."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Call the function with dictionary input
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [
                    {
                        "input": {
                            "type": "text",
                            "payload": "your content here",
                        },
                        # optional
                        "expectedOutput": {
                            "type": "text",
                            "payload": "your content here",
                        },
                        # optional
                        "context": {
                            "type": "text",
                            "payload": "your content here",
                        },
                    },
                ],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify the request body contains properly converted entries
            call_args = mock_client.request.call_args
            self.assertIsNotNone(call_args)
            
            # Extract the content from the request call
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            
            # Parse and verify the request body structure
            parsed_body = json.loads(request_content)
            self.assertIn("datasetId", parsed_body)
            self.assertIn("entries", parsed_body)
            self.assertEqual(parsed_body["datasetId"], os.environ["MAXIM_DATASET_ID"])
            self.assertEqual(len(parsed_body["entries"]), 1)
            
            # Verify entry structure
            entry = parsed_body["entries"][0]
            self.assertIn("input", entry)
            self.assertIn("expectedOutput", entry)
            self.assertIn("context", entry)
            self.assertEqual(entry["input"]["type"], "text")
            self.assertEqual(entry["input"]["payload"], "your content here")

    def test_add_dataset_entries_with_dataset_entry_objects(self) -> None:
        """Test the add_dataset_entries function using DatasetEntry objects directly."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Create DatasetEntry objects
            input_var = Variable("text", {"text": "test input"})
            expected_output_var = Variable("text", {"text": "expected output"})
            context_var = Variable("text", {"text": "test context"})
            
            dataset_entry = DatasetEntry(
                input=input_var,
                expectedOutput=expected_output_var,
                context=context_var,
            )

            # Call the function with DatasetEntry objects
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [dataset_entry],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify the request body structure
            call_args = mock_client.request.call_args
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            parsed_body = json.loads(request_content)
            
            # Verify entry structure matches DatasetEntry object
            entry = parsed_body["entries"][0]
            self.assertEqual(entry["input"]["type"], "text")
            self.assertEqual(entry["input"]["payload"], {"text": "test input"})
            self.assertEqual(entry["expectedOutput"]["type"], "text")
            self.assertEqual(entry["expectedOutput"]["payload"], {"text": "expected output"})
            self.assertEqual(entry["context"]["type"], "text")
            self.assertEqual(entry["context"]["payload"], {"text": "test context"})

    def test_add_dataset_entries_mixed_input_types(self) -> None:
        """Test the add_dataset_entries function with mixed DatasetEntry and dict inputs."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Create mixed input types
            dataset_entry = DatasetEntry(
                input=Variable("text", {"text": "object input"}),
                expectedOutput=Variable("text", {"text": "object output"}),
            )
            
            dict_entry = {
                "input": {
                    "type": "text",
                    "payload": "dict input",
                },
                "expectedOutput": {
                    "type": "text",
                    "payload": "dict output",
                },
            }

            # Call the function with mixed input types
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [dataset_entry, dict_entry],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify both entries are properly converted
            call_args = mock_client.request.call_args
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            parsed_body = json.loads(request_content)
            
            self.assertEqual(len(parsed_body["entries"]), 2)
            
            # Verify first entry (DatasetEntry object)
            first_entry = parsed_body["entries"][0]
            self.assertEqual(first_entry["input"]["payload"], {"text": "object input"})
            self.assertEqual(first_entry["expectedOutput"]["payload"], {"text": "object output"})
            
            # Verify second entry (dict)
            second_entry = parsed_body["entries"][1]
            self.assertEqual(second_entry["input"]["payload"], "dict input")
            self.assertEqual(second_entry["expectedOutput"]["payload"], "dict output")

    def test_add_dataset_entries_invalid_entry_type(self) -> None:
        """Test that add_dataset_entries raises Exception for invalid entry types."""
        with self.assertRaises(Exception) as context:
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                ["invalid_entry_type"],  # String instead of dict or DatasetEntry
            )
        
        self.assertIn("Invalid entry type", str(context.exception))

    def test_add_dataset_entries_invalid_dict_structure(self) -> None:
        """Test that add_dataset_entries handles invalid dictionary structure properly."""
        with self.assertRaises(Exception) as context:
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [{"invalid_key": "value"}],  # Missing required 'input' field
            )
        
        self.assertIn("Required 'input' field is missing", str(context.exception))

    def test_add_dataset_entries_empty_list(self) -> None:
        """Test the add_dataset_entries function with an empty list."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Call the function with empty list
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify the request body contains empty entries array
            call_args = mock_client.request.call_args
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            parsed_body = json.loads(request_content)
            
            self.assertEqual(parsed_body["datasetId"], os.environ["MAXIM_DATASET_ID"])
            self.assertEqual(parsed_body["entries"], [])

    def test_add_dataset_entries_minimal_input(self) -> None:
        """Test the add_dataset_entries function with minimal required input only."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Call the function with minimal input (only required 'input' field)
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [
                    {
                        "input": {
                            "type": "text",
                            "payload": "minimal input",
                        },
                    },
                ],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify the request body structure
            call_args = mock_client.request.call_args
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            parsed_body = json.loads(request_content)
            
            entry = parsed_body["entries"][0]
            self.assertIn("input", entry)
            self.assertNotIn("expectedOutput", entry)
            self.assertNotIn("context", entry)
            self.assertEqual(entry["input"]["payload"], "minimal input")

    def test_add_dataset_entries_json_variable_type(self) -> None:
        """Test the add_dataset_entries function with JSON variable type."""
        # Create a mock response that succeeds
        mock_response = Mock()
        mock_response.content = b'{"success": true, "message": "Entries added successfully"}'
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(
            self.maxim.maxim_api.connection_pool, "get_client",
        ) as mock_client_context:
            mock_client = Mock()
            mock_client_context.return_value.__enter__.return_value = mock_client
            mock_client.request.return_value = mock_response

            # Call the function with JSON variable type
            self.maxim.maxim_api.add_dataset_entries(
                os.environ["MAXIM_DATASET_ID"],
                [
                    {
                        "input": {
                            "type": "json",
                            "payload": {"key": "value", "number": 42},
                        },
                        "expectedOutput": {
                            "type": "json",
                            "payload": {"result": "success"},
                        },
                    },
                ],
            )

            # Verify the network call was made
            mock_client.request.assert_called_once()
            
            # Verify the request body structure
            call_args = mock_client.request.call_args
            request_content = call_args.kwargs.get('content')
            self.assertIsNotNone(request_content)
            parsed_body = json.loads(request_content)
            
            entry = parsed_body["entries"][0]
            self.assertEqual(entry["input"]["type"], "json")
            self.assertEqual(entry["input"]["payload"], {"key": "value", "number": 42})
            self.assertEqual(entry["expectedOutput"]["type"], "json")
            self.assertEqual(entry["expectedOutput"]["payload"], {"result": "success"})


if __name__ == "__main__":
    unittest.main()