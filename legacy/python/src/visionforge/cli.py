"""Developer-friendly CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from visionforge.domain import Frame
from visionforge.factory import build_demo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="visionforge")
    parser.add_argument("command", choices=("demo",))
    args = parser.parse_args()
    if args.command == "demo":
        run_demo()


def run_demo() -> None:
    pipeline = build_demo_pipeline()
    frames = [
        Frame(
            frame_id="0001",
            source_id="camera-1",
            timestamp_ms=0,
            payload={
                "detections": [
                    {"bbox": [10, 20, 110, 220], "label": "person", "score": 0.94},
                    {"bbox": [250, 80, 390, 190], "label": "car", "score": 0.88},
                ]
            },
        ),
        Frame(
            frame_id="0002",
            source_id="camera-1",
            timestamp_ms=40,
            payload={
                "detections": [
                    {"bbox": [14, 20, 114, 220], "label": "person", "score": 0.96}
                ]
            },
        ),
    ]
    for frame in frames:
        print(json.dumps(asdict(pipeline.analyze(frame)), ensure_ascii=False))


if __name__ == "__main__":
    main()

