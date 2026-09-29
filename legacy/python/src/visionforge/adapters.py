"""Deterministic demo adapters; production integrations implement the same ports."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from visionforge.domain import BoundingBox, Detection, Frame


class PayloadDetector:
    """Reads detections from a frame payload, useful for demos and contract tests."""

    def detect(self, frame: Frame) -> Sequence[Detection]:
        if not isinstance(frame.payload, dict):
            return ()
        return tuple(
            Detection(
                bbox=BoundingBox(*candidate["bbox"]),
                label=str(candidate["label"]),
                score=float(candidate["score"]),
            )
            for candidate in frame.payload.get("detections", [])
        )


class IdentityCropper:
    def crop(self, frame: Frame, detection: Detection) -> dict[str, Any]:
        return {"frame": frame.frame_id, "label": detection.label, "bbox": detection.bbox}


class HashEmbedder:
    """Stable demo embedding. It is not an ML model and is intentionally explicit."""

    def __init__(self, dimension: int = 4) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed_image(self, image: Any) -> tuple[float, ...]:
        return self._hash(repr(image))

    def embed_text(self, text: str) -> tuple[float, ...]:
        return self._hash(text.casefold())

    def _hash(self, value: str) -> tuple[float, ...]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return tuple((digest[index] + 1) / 256 for index in range(self.dimension))
