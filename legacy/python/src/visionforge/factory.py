"""Composition root for the local reference application."""

from __future__ import annotations

from visionforge.adapters import HashEmbedder, IdentityCropper, PayloadDetector
from visionforge.config import PipelineConfig
from visionforge.pipeline import AnalysisPipeline
from visionforge.tracking import IoUTracker
from visionforge.vector_store import InMemoryVectorStore


def build_demo_pipeline(config: PipelineConfig | None = None) -> AnalysisPipeline:
    config = config or PipelineConfig()
    store = InMemoryVectorStore()
    return AnalysisPipeline(
        config=config,
        detector=PayloadDetector(),
        cropper=IdentityCropper(),
        tracker=IoUTracker(
            iou_threshold=config.tracking.iou_threshold,
            max_missed_frames=config.tracking.max_missed_frames,
        ),
        embedder=HashEmbedder(),
        vector_store=store,
    )

