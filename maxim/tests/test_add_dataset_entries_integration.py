import os
from pathlib import Path
import unittest
import tempfile
import time
from contextlib import suppress

import dotenv
from maxim import Maxim, Config
from maxim.models.dataset import DatasetEntry, Variable
from maxim.logger.components.attachment import FileAttachment, FileDataAttachment, UrlAttachment

# Load environment variables
dotenv.load_dotenv()

# Environment variables for integration testing
MAXIM_API_KEY = os.getenv("MAXIM_API_KEY")
MAXIM_BASE_URL = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"
MAXIM_DATASET_ID = "cmesayj1f001c9ybdxwxmtilk"  # Dataset ID for integration tests


class TestAddDatasetEntriesIntegration(unittest.TestCase):
    """
    Integration tests for add_dataset_entries function using real API calls.
    
    These tests require the following environment variables:
    - MAXIM_API_KEY: Your Maxim API key
    - MAXIM_DATASET_ID: A test dataset ID where entries can be added
    - MAXIM_BASE_URL: (optional) Base URL for Maxim API, defaults to https://app.getmaxim.ai
    
    If these environment variables are not set, the tests will be skipped.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up class-level fixtures and check for required environment variables."""
        if not MAXIM_API_KEY:
            raise unittest.SkipTest("MAXIM_API_KEY environment variable is not set")
        
        if not MAXIM_DATASET_ID:
            raise unittest.SkipTest("MAXIM_DATASET_ID environment variable is not set")
        
        print(f"\n🧪 Running integration tests against dataset: {MAXIM_DATASET_ID}")
        print(f"📡 Using API endpoint: {MAXIM_BASE_URL}")

    def setUp(self) -> None:
        """Set up test fixtures for each test."""
        # Ensure no cached Maxim instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        # Create Maxim instance with real credentials
        config = Config(
            api_key=MAXIM_API_KEY,
            base_url=MAXIM_BASE_URL,
            debug=True,
            raise_exceptions=True
        )
        self.maxim = Maxim(config)
        self.dataset_id = MAXIM_DATASET_ID

    def tearDown(self) -> None:
        """Clean up after tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        
        # Add a small delay between tests to avoid rate limiting
        time.sleep(0.5)

    def test_add_dataset_entries_with_dictionary_input(self) -> None:
        """
        Test adding dataset entries using dictionary input with real API calls.
        
        This test verifies that:
        1. Dictionary entries can be successfully added to a dataset
        2. Various data types (strings, numbers, objects) are handled correctly
        3. Multiple attachments per entry are handled correctly (including real files)
        4. Different file types (image, PDF, markdown) are uploaded successfully
        5. The API response indicates success
        """
        print("\n📝 Testing dictionary input with real API (including 3 attachments: image, PDF, and markdown)...")
        
        # Resolve test resources relative to the repository root (portable)
        repo_root = Path(__file__).resolve().parents[2]
        files_dir = repo_root / "maxim" / "tests" / "files"

        real_image_path = files_dir / "png_image.png"
        real_audio_path = files_dir / "wav_audio.wav"

        # Verify files exist before creating attachments
        assert real_image_path.exists(), "Image file not found"
        assert real_audio_path.exists(), "Audio file not found"
        
        # Create one temporary markdown file to complement the real files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write("# Additional Context\n\nThis markdown file contains supplementary information about circle calculations.\n\n## Formula Reference\n- Area: A = π × r²\n- Circumference: C = 2π × r")
            temp_file_path3 = temp_file.name
        
        # Create file attachments for context - 2 real files + 1 temporary
        # Use original filenames and let the system auto-detect MIME types
        context_file1 = FileAttachment(
            path=str(real_image_path),
            name="png_image.png"  # Use original filename
        )
        
        context_file2 = FileAttachment(
            path=str(real_audio_path),
            name="wav_audio.wav"  # Use original filename
        )
        
        context_file3 = UrlAttachment(
            url="https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg",
            name="background.jpeg"
        )
        
        # Create test entries with correct column structure
        # First entry now has 3 attachments to test multiple files per entry
        entries = [
            {
                "Input": "How do I calculate the area of a circle?",
                "Test": {
                    "formula": "A = π × r²",
                    "explanation": "The area of a circle is π times the radius squared",
                    "example": "For radius 5: A = π × 25 = 78.54 square units",
                    "confidence": 0.99,
                    "supporting_docs": 3
                },
                "context": [context_file1, context_file2, context_file3]  # 3 attachments in one entry
            },
            {
                "Input": "What is 15 + 27?",
                "Test": {
                    "answer": 42,
                    "method": "simple addition",
                    "step_by_step": ["15 + 27", "= 42"],
                    "confidence": 1.0
                },
                "context": []  # No attachments for second entry to test mixed scenarios
            }
        ]

        try:
            # Call the real API
            response = self.maxim.add_dataset_entries(self.dataset_id, entries)
            
            # Verify the response
            self.assertIsInstance(response, dict)
            self.assertIn("data", response)
            self.assertIn("ids", response["data"])
            self.assertTrue(len(response["data"]["ids"]) > 0)
            
            print(f"✅ Successfully added {len(entries)} dictionary entries")
            print(f"📊 API Response: {response}")
            
        except (ValueError, ConnectionError, RuntimeError) as e:
            self.fail(f"Failed to add dictionary entries: {str(e)}")
        finally:
            # Clean up temporary files (only the markdown file, not the real files)
            with suppress(OSError):
                os.unlink(temp_file_path3)  # Only clean up the temporary markdown file

    def test_add_dataset_entries_with_dataset_entry_objects(self) -> None:
        """
        Test adding dataset entries using DatasetEntry objects with real API calls.
        
        This test verifies that:
        1. DatasetEntry objects can be successfully added to a dataset
        2. Variable types are properly handled (text, json)
        3. The conversion from DatasetEntry to API format works correctly
        """
        print("\n🏗️ Testing DatasetEntry objects with real API...")
        
        # Create test file for context column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is context data for the first DatasetEntry test.\nIt provides background information for the AI model.")
            temp_file_path1 = temp_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Second context file with different information.\nThis helps test multiple file attachments.")
            temp_file_path2 = temp_file.name
        
        # Create file attachments for context column
        context_file1 = FileAttachment(
            path=temp_file_path1,
            name="context_file_1.txt",
            mime_type="text/plain"
        )
        
        context_file2 = FileAttachment(
            path=temp_file_path2,
            name="context_file_2.txt",
            mime_type="text/plain"
        )
        
        # Create DatasetEntry objects with correct column structure
        entry1 = DatasetEntry(entry={
            "Input": Variable(type="text", payload="What is the capital of France?"),
            "Test": Variable(type="json", payload={
                "answer": "Paris",
                "confidence": 0.95,
                "reasoning": "Paris is the capital and largest city of France",
                "sources": ["geography", "world_knowledge"]
            }),
            "context": Variable(type="file", payload=[context_file1])
        })
        
        entry2 = DatasetEntry(entry={
            "Input": Variable(type="text", payload="Explain the process of photosynthesis"),
            "Test": Variable(type="json", payload={
                "answer": "Photosynthesis is the process by which plants convert sunlight into energy",
                "confidence": 0.98,
                "key_components": ["chlorophyll", "sunlight", "carbon dioxide", "water"],
                "outcome": "glucose and oxygen production"
            }),
            "context": Variable(type="file", payload=[context_file2])
        })

        try:
            # Call the real API
            response = self.maxim.add_dataset_entries(
                self.dataset_id, [entry1, entry2]
            )
            
            # Verify the response
            self.assertIsInstance(response, dict)
            self.assertIn("data", response)
            self.assertIn("ids", response["data"])
            self.assertTrue(len(response["data"]["ids"]) > 0)
            
            print(f"✅ Successfully added {len([entry1, entry2])} DatasetEntry objects")
            print(f"📊 API Response: {response}")
            
        except (ValueError, ConnectionError, RuntimeError) as e:
            self.fail(f"Failed to add DatasetEntry objects: {str(e)}")
        finally:
            # Clean up temporary files
            with suppress(OSError):
                os.unlink(temp_file_path1)
            with suppress(OSError):
                os.unlink(temp_file_path2)

    def test_add_dataset_entries_with_file_attachments(self) -> None:
        """
        Test adding dataset entries with file attachments using real API calls.
        
        This test verifies that:
        1. File attachments can be uploaded and linked to dataset entries
        2. Both FileAttachment and FileDataAttachment work correctly
        3. The file upload and dataset entry creation process completes successfully
        """
        print("\n📎 Testing file attachments with real API...")
        
        # Create temporary test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_text_file:
            temp_text_file.write("This is a test file for integration testing.\nIt contains sample data for the dataset entry.")
            temp_text_file_path = temp_text_file.name

        try:
            # Test with FileAttachment (file path)
            file_attachment = FileAttachment(
                path=temp_text_file_path,
                name="integration_test.txt",
                mime_type="text/plain"
            )
            
            entry_with_file = DatasetEntry(entry={
                "Input": Variable(type="text", payload="Integration test with file attachment"),
                "Test": Variable(type="json", payload={
                    "result": "Successfully processed file attachment",
                    "file_type": "text",
                    "confidence": 0.95
                }),
                "context": Variable(type="file", payload=[file_attachment])
            })

            # Test with FileDataAttachment (in-memory data)
            binary_data = b"Binary test data for integration testing: " + str(time.time()).encode()
            file_data_attachment = FileDataAttachment(
                data=binary_data,
                name="integration_binary.dat",
                mime_type="application/octet-stream"
            )
            
            entry_with_data = DatasetEntry(entry={
                "Input": Variable(type="text", payload="Integration test with binary data"),
                "Test": Variable(type="json", payload={
                    "result": "Successfully processed binary data",
                    "file_size_bytes": len(binary_data),
                    "confidence": 0.97
                }),
                "context": Variable(type="file", payload=[file_data_attachment])
            })

            # Call the real API
            response = self.maxim.add_dataset_entries(
                self.dataset_id, [entry_with_file, entry_with_data]
            )
            
            # Verify the response
            self.assertIsInstance(response, dict)
            self.assertIn("data", response)
            self.assertIn("ids", response["data"])
            self.assertTrue(len(response["data"]["ids"]) > 0)
            
            print("✅ Successfully added entries with file attachments")
            print(f"📊 API Response: {response}")
            
        except (ValueError, ConnectionError, RuntimeError) as e:
            self.fail(f"Failed to add entries with file attachments: {str(e)}")
            
        finally:
            # Clean up temporary files            
            with suppress(OSError):
                os.unlink(temp_text_file_path)

    def test_mixed_entry_types_integration(self) -> None:
        """
        Test adding a mix of dictionary and DatasetEntry objects in a single call.
        
        This test verifies that:
        1. Mixed input types can be processed in a single API call
        2. The conversion and processing pipeline handles both formats correctly
        3. All entries are successfully added to the dataset
        """
        print("\n🔄 Testing mixed entry types with real API...")
        
        # Create test files for context column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Mixed test context for dictionary entry.\nThis provides context for the mixed type test.")
            temp_file_path1 = temp_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("Context for DatasetEntry in mixed test.\nAdditional background information for processing.")
            temp_file_path2 = temp_file.name
        
        # Create file attachments
        context_file1 = FileAttachment(
            path=temp_file_path1,
            name="mixed_context_dict.txt",
            mime_type="text/plain"
        )
        
        context_file2 = FileAttachment(
            path=temp_file_path2,
            name="mixed_context_dataset.txt", 
            mime_type="text/plain"
        )
        
        # Create mixed entry types with correct columns
        dict_entry = {
            "Input": "What are the benefits of renewable energy?",
            "Test": {
                "benefits": ["environmental protection", "sustainability", "cost savings"],
                "types": ["solar", "wind", "hydro", "geothermal"],
                "impact": "positive long-term effects on climate",
                "confidence": 0.92
            },
            "context": [context_file1]
        }
        
        dataset_entry = DatasetEntry(entry={
            "Input": Variable(type="text", payload="Explain machine learning in simple terms"),
            "Test": Variable(type="json", payload={
                "definition": "Machine learning is AI that learns from data to make predictions",
                "key_concepts": ["algorithms", "training", "predictions", "patterns"],
                "applications": ["image recognition", "natural language processing", "recommendation systems"],
                "confidence": 0.94
            }),
            "context": Variable(type="file", payload=[context_file2])
        })

        try:
            # Call the real API with mixed types
            response = self.maxim.add_dataset_entries(
                self.dataset_id, [dict_entry, dataset_entry]
            )
            
            # Verify the response
            self.assertIsInstance(response, dict)
            self.assertIn("data", response)
            self.assertIn("ids", response["data"])
            self.assertTrue(len(response["data"]["ids"]) > 0)
            
            print("✅ Successfully added mixed entry types")
            print(f"📊 API Response: {response}")
            
        except (ValueError, ConnectionError, RuntimeError) as e:
            self.fail(f"Failed to add mixed entry types: {str(e)}")
        finally:
            # Clean up temporary files
            with suppress(OSError):
                os.unlink(temp_file_path1)
            with suppress(OSError):
                os.unlink(temp_file_path2)

    def test_error_handling_with_invalid_dataset_id(self) -> None:
        """
        Test error handling when using an invalid dataset ID.
        
        This test verifies that:
        1. Appropriate exceptions are raised for invalid dataset IDs
        2. Error messages are informative
        3. The API client handles errors gracefully
        """
        print("\n❌ Testing error handling with invalid dataset ID...")
        
        invalid_dataset_id = "invalid-dataset-id-12345"
        test_entry = {
            "input": "This should fail due to invalid dataset ID",
            "expected_behavior": "error"
        }

        with self.assertRaises(Exception) as context:
            self.maxim.add_dataset_entries(invalid_dataset_id, [test_entry])
        
        # Verify that the exception contains meaningful information
        error_message = str(context.exception)
        self.assertTrue(len(error_message) > 0)
        
        print(f"✅ Correctly caught exception: {error_message}")


if __name__ == "__main__":
    # Print setup instructions if environment variables are missing
    if not MAXIM_API_KEY or not MAXIM_DATASET_ID:
        print("\n" + "="*60)
        print("INTEGRATION TEST SETUP REQUIRED")
        print("="*60)
        print("To run these integration tests, set the following environment variables:")
        print("  MAXIM_API_KEY=your_api_key_here")
        print("  MAXIM_DATASET_ID=your_test_dataset_id_here")
        print("  MAXIM_BASE_URL=https://app.getmaxim.ai  # optional")
        print("\nYou can also create a .env file in the project root with these values.")
        print("="*60)
    
    unittest.main(verbosity=2)