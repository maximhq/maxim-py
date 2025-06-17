from typing import List, Optional


class MaximInMemoryCache():
    """
    In-memory cache implementation for Maxim.

    This class provides a simple in-memory cache implementation
    that stores key-value pairs in a dictionary.
    """

    def __init__(self):
        """
        Initialize the in-memory cache.
        """
        self.cache = {}

    def get_all_keys(self) -> List[str]:
        """
        Get all keys currently stored in the cache.
        """
        return list(self.cache.keys())

    def get(self, key: str) -> Optional[str]:
        """
        Get a value from the cache by its key.
        """
        return self.cache.get(key)

    def set(self, key: str, value: str) -> None:
        """
        Store a key-value pair in the cache.
        """
        self.cache[key] = value

    def delete(self, key: str) -> None:
        """
        Remove a key-value pair from the cache.
        """
        if key in self.cache:
            del self.cache[key]
