"""Typed TOML configuration loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    iou_threshold: float = 0.3
    max_missed_frames: int = 20


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    enable_tracking: bool = True
    enable_embeddings: bool = True
    enable_ocr: bool = False
    index_embeddings: bool = True
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        pipeline = raw.get("pipeline", {})
        tracking = raw.get("tracking", {})
        return cls(
            enable_tracking=bool(pipeline.get("enable_tracking", True)),
            enable_embeddings=bool(pipeline.get("enable_embeddings", True)),
            enable_ocr=bool(pipeline.get("enable_ocr", False)),
            index_embeddings=bool(pipeline.get("index_embeddings", True)),
            tracking=TrackingConfig(
                iou_threshold=float(tracking.get("iou_threshold", 0.3)),
                max_missed_frames=int(tracking.get("max_missed_frames", 20)),
            ),
        )

