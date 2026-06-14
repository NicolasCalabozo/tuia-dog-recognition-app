from __future__ import annotations

from typing import Protocol

from lib.schemas import EmbeddingRecord


class EmbeddingStoreProtocol(Protocol):
    def all(self) -> list[EmbeddingRecord]:
        ...

    def append(self, record: EmbeddingRecord) -> None:
        ...
