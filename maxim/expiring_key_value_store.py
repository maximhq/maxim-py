import time
from collections import OrderedDict
from typing import Any


class ExpiringKeyValueStore:
    """Expiring key value store.

    This class represents an expiring key value store.
    """

    def __init__(self):
        """Initialize an expiring key value store."""
        self.store = OrderedDict()

    def set(self, key: str, value: Any, expiry_seconds: int):
        """Set a value in the expiring key value store.

        Args:
            key: The key to set.
            value: The value to set.
            expiry_seconds: The expiry time in seconds.
        """
        expiry_time = time.time() + expiry_seconds
        self.store[key] = (value, expiry_time)
        self._evict_expired()

    def get(self, key: str):
        """Get a value from the expiring key value store.

        Args:
            key: The key to get.

        Returns:
            Any: The value.
        """
        if key in self.store:
            value, expiry_time = self.store[key]
            if time.time() < expiry_time:
                return value
            else:
                del self.store[key]
        return None

    def delete(self, key: str):
        """Delete a value from the expiring key value store.

        Args:
            key: The key to delete.
        """
        if key in self.store:
            del self.store[key]

    def _evict_expired(self):
        """Evict expired values from the expiring key value store.

        This method evicts expired values from the expiring key value store.
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, expiry_time) in self.store.items()
            if current_time >= expiry_time
        ]
        for key in expired_keys:
            del self.store[key]
