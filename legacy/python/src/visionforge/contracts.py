"""Ports that isolate ML frameworks and infrastructure from business logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from visionforge.domain import Detection, Frame, SearchResult, Track


class Detector(Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> Sequence[Track]: ...


class Embedder(Protocol):
    def embed_image(self, image: Any) -> Sequence[float]: ...

    def embed_text(self, text: str) -> Sequence[float]: ...


class TextExtractor(Protocol):
    def extract(self, image: Any) -> Sequence[str]: ...


class VectorStore(Protocol):
    def upsert(self, item_id: str, vector: Sequence[float], metadata: dict[str, Any]) -> None: ...

    def search(self, vector: Sequence[float], limit: int = 10) -> Sequence[SearchResult]: ...


class Cropper(Protocol):
    def crop(self, frame: Frame, detection: Detection) -> Any: ...
