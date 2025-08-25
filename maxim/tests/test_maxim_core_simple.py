import os
import unittest
from unittest.mock import Mock, patch
from typing import Optional

from maxim import Maxim, Config
from maxim.cache import MaximCache, MaximInMemoryCache

baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"

class FalsyCache(MaximCache):
    """A cache that evaluates to False but is not None - used to test the 'cache is not None' fix."""

    def __init__(self):
        self.data = {}
        self.was_used = False

    def __bool__(self):
        return False

    def get_all_keys(self):
        return list(self.data.keys())

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        self.was_used = True

    def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]


class TestMaximCacheHandling(unittest.TestCase):
    """Test the cache handling changes, specifically the 'if cache is not None:' fix."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        os.environ["MAXIM_API_KEY"] = "test-api-key"

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        if "MAXIM_API_KEY" in os.environ:
            del os.environ["MAXIM_API_KEY"]

    def test_enable_prompt_management_with_none_cache(self):
        """Test enable_prompt_management with cache=None."""
        maxim = Maxim({"api_key": "test-key", "base_url": baseUrl, "prompt_management": False})
        original_cache = getattr(maxim, "_Maxim__cache")

        # This should work fine even with cache=None
        result = maxim.enable_prompt_management(cache=None)

        # Prompt management should be enabled
        self.assertTrue(maxim.prompt_management)
        # Cache should remain the same (not replaced with None)
        self.assertEqual(getattr(maxim, "_Maxim__cache"), original_cache)
        # Should return self for method chaining
        self.assertEqual(result, maxim)

    def test_enable_prompt_management_with_falsy_cache(self):
        """
        Test enable_prompt_management with a cache that evaluates to False but is not None.

        This tests the fix from 'if cache:' to 'if cache is not None:'.
        Previously, a falsy cache would not be used even if it was a valid cache object.
        """
        maxim = Maxim({"api_key": "test-key", "prompt_management": False})

        falsy_cache = FalsyCache()
        result = maxim.enable_prompt_management(cache=falsy_cache)

        # Prompt management should be enabled
        self.assertTrue(maxim.prompt_management)
        # The falsy cache should be used (this is the key test for the fix)
        self.assertEqual(getattr(maxim, "_Maxim__cache"), falsy_cache)
        # Should return self for method chaining
        self.assertEqual(result, maxim)

    def test_enable_prompt_management_with_valid_cache(self):
        """Test enable_prompt_management with a normal valid cache."""
        maxim = Maxim({"api_key": "test-key", "prompt_management": False})

        new_cache = MaximInMemoryCache()
        result = maxim.enable_prompt_management(cache=new_cache)

        # Prompt management should be enabled
        self.assertTrue(maxim.prompt_management)
        # The new cache should be used
        self.assertEqual(getattr(maxim, "_Maxim__cache"), new_cache)
        # Should return self for method chaining
        self.assertEqual(result, maxim)

    def test_enable_prompt_management_starts_sync_thread(self):
        """Test that enable_prompt_management starts the sync thread."""
        maxim = Maxim({"api_key": "test-key", "prompt_management": False})

        with patch.object(maxim, "_Maxim__sync_thread", None):
            result = maxim.enable_prompt_management()

            # Should create and start a new sync thread
            sync_thread = getattr(maxim, "_Maxim__sync_thread")
            self.assertIsNotNone(sync_thread)
            self.assertTrue(sync_thread.daemon)

    def test_enable_exceptions_method_chaining(self):
        """Test enable_exceptions method returns self for chaining."""
        maxim = Maxim({"api_key": "test-key"})

        # Test enabling exceptions
        result = maxim.enable_exceptions(True)
        self.assertTrue(maxim.raise_exceptions)
        self.assertEqual(result, maxim)

        # Test disabling exceptions
        result = maxim.enable_exceptions(False)
        self.assertFalse(maxim.raise_exceptions)
        self.assertEqual(result, maxim)


class TestMaximInitialization(unittest.TestCase):
    """Test Maxim initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")
        if "MAXIM_API_KEY" in os.environ:
            del os.environ["MAXIM_API_KEY"]

    def test_maxim_requires_api_key(self):
        """Test that Maxim requires an API key."""
        with self.assertRaises(ValueError) as context:
            Maxim({"base_url": baseUrl})

        self.assertIn("API key is required", str(context.exception))

    def test_maxim_uses_env_api_key(self):
        """Test that Maxim uses environment variable for API key."""
        os.environ["MAXIM_API_KEY"] = "test-env-key"

        maxim = Maxim({"base_url": baseUrl})

        self.assertEqual(maxim.api_key, "test-env-key")

    def test_maxim_singleton_pattern(self):
        """Test that Maxim follows singleton pattern."""
        os.environ["MAXIM_API_KEY"] = "test-key"

        maxim1 = Maxim({"base_url": baseUrl})

        with self.assertRaises(RuntimeError) as context:
            Maxim({"base_url": baseUrl})

        self.assertIn("already initialized", str(context.exception))

    def test_maxim_default_cache_creation(self):
        """Test that Maxim creates default cache when none provided."""
        maxim = Maxim({"api_key": "test-key", "base_url": baseUrl})

        cache = getattr(maxim, "_Maxim__cache")
        self.assertIsInstance(cache, MaximInMemoryCache)

    def test_cleanup_method_stops_running(self):
        """Test that cleanup method properly sets is_running to False."""
        maxim = Maxim({"api_key": "test-key", "base_url": baseUrl})

        self.assertTrue(maxim.is_running)
        # Give a moment for initialization to complete
        import time

        time.sleep(0.1)
        maxim.cleanup()
        self.assertFalse(maxim.is_running)

    def test_cleanup_prevents_double_execution(self):
        """Test that cleanup method can be called multiple times safely."""
        maxim = Maxim({"api_key": "test-key", "base_url": baseUrl})

        # Give a moment for initialization to complete
        import time

        time.sleep(0.1)

        # First cleanup
        maxim.cleanup()
        self.assertFalse(maxim.is_running)

        # Second cleanup should not cause issues
        maxim.cleanup()
        self.assertFalse(maxim.is_running)


class TestGetConfigDict(unittest.TestCase):
    """Test the get_config_dict function."""

    def test_get_config_dict_with_config_object(self):
        """Test get_config_dict function with Config object."""
        from maxim.maxim import get_config_dict

        config = Config(
            api_key="test-key",
            base_url="https://test.com",
            debug=True,
            raise_exceptions=True,
            prompt_management=True,
        )

        result = get_config_dict(config)

        self.assertEqual(result["api_key"], "test-key")
        self.assertEqual(result["base_url"], "https://test.com")
        self.assertTrue(result["debug"])
        self.assertTrue(result["raise_exceptions"])
        self.assertTrue(result["prompt_management"])
        self.assertIsInstance(result["cache"], MaximInMemoryCache)

    def test_get_config_dict_with_dict(self):
        """Test get_config_dict function with dictionary."""
        from maxim.maxim import get_config_dict

        config_dict = {
            "api_key": "test-key",
            "base_url": "https://test.com",
            "debug": True,
        }

        result = get_config_dict(config_dict)

        self.assertEqual(result["api_key"], "test-key")
        self.assertEqual(result["base_url"], "https://test.com")
        self.assertTrue(result["debug"])


if __name__ == "__main__":
    unittest.main()
