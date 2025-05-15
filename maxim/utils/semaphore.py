import hashlib
import threading
from typing import Dict, List


class Semaphore:
    _semaphores: Dict[str, "Semaphore"] = {}

    def __init__(self, key: str, max_locks: int):
        self.key = self.hash(key)
        self.max_locks = max_locks
        self.current_locks = 0
        self.queue: List[threading.Event] = []
        Semaphore._semaphores[self.key] = self

    @staticmethod
    def hash(key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def acquire(self):
        if self.current_locks < self.max_locks:
            self.current_locks += 1
            return
        event = threading.Event()
        self.queue.append(event)
        event.wait()

    def release(self):
        if self.current_locks > 0:
            self.current_locks -= 1
            if self.queue:
                next_event = self.queue.pop(0)
                self.current_locks += 1
                next_event.set()

    @classmethod
    def get(cls, key: str, max_locks: int) -> "Semaphore":
        hashed_key = cls.hash(key)
        if hashed_key not in cls._semaphores:
            return Semaphore(key, max_locks)
        return cls._semaphores[hashed_key]
