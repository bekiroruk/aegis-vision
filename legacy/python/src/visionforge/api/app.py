"""FastAPI adapter kept outside the dependency-free core."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - only exercised without API extra
    raise RuntimeError('Install API dependencies with: pip install -e ".[api]"') from exc

from visionforge.domain import Frame
from visionforge.factory import build_demo_pipeline


class DetectionInput(BaseModel):
    bbox: tuple[float, float, float, float]
    label: str
    score: float = Field(ge=0, le=1)


class AnalyzeRequest(BaseModel):
    frame_id: str
    source_id: str
    timestamp_ms: int = Field(ge=0)
    detections: list[DetectionInput] = Field(default_factory=list)


class TextSearchRequest(BaseModel):
    text: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


def create_app() -> Any:
    app = FastAPI(title="VisionForge API", version="0.1.0")
    pipeline = build_demo_pipeline()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        payload = {"detections": [item.model_dump() for item in request.detections]}
        result = pipeline.analyze(
            Frame(
                frame_id=request.frame_id,
                source_id=request.source_id,
                timestamp_ms=request.timestamp_ms,
                payload=payload,
            )
        )
        return asdict(result)

    @app.post("/v1/search/text")
    def search_text(request: TextSearchRequest) -> list[dict[str, Any]]:
        if pipeline.embedder is None or pipeline.vector_store is None:
            return []
        vector = pipeline.embedder.embed_text(request.text)
        return [asdict(result) for result in pipeline.vector_store.search(vector, request.limit)]

    return app
