from typing import List, Optional


class MaximCache():
    """
    Abstract base class for caching implementations in Maxim.

    This class defines the interface for cache operations including
    getting, setting, deleting cache entries, and retrieving all keys.
    Concrete implementations should inherit from this class and provide
    actual storage mechanisms.
    """

    def get_all_keys(self) -> List[str]:
        """
        Retrieve all keys currently stored in the cache.

        Returns:
            List[str]: A list of all cache keys. Returns empty list if no keys exist.
        """
        return []

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a value from the cache by its key.

        Args:
            key (str): The cache key to look up.

        Returns:
            Optional[str]: The cached value if the key exists, None otherwise.
        """
        pass

    def set(self, key: str, value: str) -> None:
        """
        Store a key-value pair in the cache.

        Args:
            key (str): The cache key to store the value under.
            value (str): The value to cache.
        """
        pass

    def delete(self, key: str) -> None:
        """
        Remove a key-value pair from the cache.

        Args:
            key (str): The cache key to remove.
        """
        pass
