"""Dependency-free domain types shared by every adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box coordinates must satisfy x2 >= x1 and y2 >= y1")

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: BoundingBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    label: str
    score: float
    embedding: tuple[float, ...] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox: BoundingBox
    label: str
    score: float
    age: int
    missed_frames: int = 0


@dataclass(frozen=True, slots=True)
class Frame:
    frame_id: str
    source_id: str
    timestamp_ms: int
    payload: Any = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    item_id: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    frame_id: str
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    extracted_text: tuple[str, ...]
    indexed_items: tuple[str, ...]

