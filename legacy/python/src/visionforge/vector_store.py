"""In-memory cosine index for local development and contract tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from visionforge.domain import SearchResult


@dataclass(frozen=True, slots=True)
class _VectorItem:
    vector: tuple[float, ...]
    metadata: dict[str, Any]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, _VectorItem] = {}
        self._dimension: int | None = None

    def upsert(self, item_id: str, vector: Sequence[float], metadata: dict[str, Any]) -> None:
        normalized = _normalize(vector)
        if self._dimension is None:
            self._dimension = len(normalized)
        if len(normalized) != self._dimension:
            raise ValueError(f"Expected vector dimension {self._dimension}, got {len(normalized)}")
        self._items[item_id] = _VectorItem(normalized, dict(metadata))

    def search(self, vector: Sequence[float], limit: int = 10) -> tuple[SearchResult, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query = _normalize(vector)
        if self._dimension is not None and len(query) != self._dimension:
            raise ValueError(f"Expected vector dimension {self._dimension}, got {len(query)}")
        ranked = sorted(
            (
                SearchResult(
                    item_id=item_id,
                    score=sum(a * b for a, b in zip(query, item.vector, strict=True)),
                    metadata=dict(item.metadata),
                )
                for item_id, item in self._items.items()
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return tuple(ranked[:limit])


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values:
        raise ValueError("Vector must not be empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("Zero vectors cannot be cosine-normalized")
    return tuple(value / norm for value in values)
