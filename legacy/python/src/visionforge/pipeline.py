"""Application service that composes CV components."""

from __future__ import annotations

from dataclasses import replace

from visionforge.config import PipelineConfig
from visionforge.contracts import Cropper, Detector, Embedder, TextExtractor, Tracker, VectorStore
from visionforge.domain import AnalysisResult, Frame


class AnalysisPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        detector: Detector,
        cropper: Cropper,
        tracker: Tracker | None = None,
        embedder: Embedder | None = None,
        text_extractor: TextExtractor | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.cropper = cropper
        self.tracker = tracker
        self.embedder = embedder
        self.text_extractor = text_extractor
        self.vector_store = vector_store
        self._validate_components()

    def analyze(self, frame: Frame) -> AnalysisResult:
        detections = tuple(self.detector.detect(frame))
        indexed_items: list[str] = []
        enriched = []

        for index, detection in enumerate(detections):
            crop = self.cropper.crop(frame, detection)
            if self.config.enable_embeddings and self.embedder:
                vector = tuple(float(value) for value in self.embedder.embed_image(crop))
                detection = replace(detection, embedding=vector)
                if self.config.index_embeddings and self.vector_store:
                    item_id = f"{frame.source_id}:{frame.frame_id}:{index}"
                    self.vector_store.upsert(
                        item_id,
                        vector,
                        {
                            "source_id": frame.source_id,
                            "frame_id": frame.frame_id,
                            "label": detection.label,
                            "timestamp_ms": frame.timestamp_ms,
                        },
                    )
                    indexed_items.append(item_id)
            enriched.append(detection)

        tracks = (
            tuple(self.tracker.update(enriched))
            if self.config.enable_tracking and self.tracker
            else ()
        )
        texts: list[str] = []
        if self.config.enable_ocr and self.text_extractor:
            for detection in enriched:
                texts.extend(self.text_extractor.extract(self.cropper.crop(frame, detection)))

        return AnalysisResult(
            frame_id=frame.frame_id,
            detections=tuple(enriched),
            tracks=tracks,
            extracted_text=tuple(texts),
            indexed_items=tuple(indexed_items),
        )

    def _validate_components(self) -> None:
        if self.config.enable_tracking and self.tracker is None:
            raise ValueError("Tracking is enabled but no tracker was provided")
        if self.config.enable_embeddings and self.embedder is None:
            raise ValueError("Embeddings are enabled but no embedder was provided")
        if self.config.index_embeddings and self.vector_store is None:
            raise ValueError("Embedding indexing is enabled but no vector store was provided")
        if self.config.enable_ocr and self.text_extractor is None:
            raise ValueError("OCR is enabled but no text extractor was provided")

