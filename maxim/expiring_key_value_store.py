import time
from collections import OrderedDict
from typing import Any


class ExpiringKeyValueStore:
    def __init__(self):
        self.store = OrderedDict()

    def set(self, key: str, value: Any, expiry_seconds: int):
        expiry_time = time.time() + expiry_seconds
        self.store[key] = (value, expiry_time)
        self._evict_expired()

    def get(self, key: str):
        if key in self.store:
            value, expiry_time = self.store[key]
            if time.time() < expiry_time:
                return value
            else:
                del self.store[key]
        return None

    def delete(self, key: str):
        if key in self.store:
            del self.store[key]

    def _evict_expired(self):
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, expiry_time) in self.store.items()
            if current_time >= expiry_time
        ]
        for key in expired_keys:
            del self.store[key]
