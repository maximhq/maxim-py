from abc import ABC, abstractmethod
from typing import List, Optional


class MaximCache(ABC):
    @abstractmethod
    def get_all_keys(self) -> List[str]: ...

    @abstractmethod
    def get(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set(self, key: str, value: str) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class AsyncMaximCache(ABC):
    @abstractmethod
    async def a_get_all_keys(self) -> List[str]: ...

    @abstractmethod
    async def a_get(self, key: str) -> Optional[str]: ...

    @abstractmethod
    async def a_set(self, key: str, value: str) -> None: ...

    @abstractmethod
    async def a_delete(self, key: str) -> None: ...
