from typing import List, Optional


class MaximInMemoryCache():
    def __init__(self):
        self.cache = {}

    def get_all_keys(self) -> List[str]:
        return list(self.cache.keys())

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str) -> None:
        self.cache[key] = value

    def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
