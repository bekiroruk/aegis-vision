"""Small, deterministic tracker used as the reference implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from visionforge.domain import Detection, Track


@dataclass(slots=True)
class _TrackState:
    track: Track


class IoUTracker:
    """Greedy class-aware IoU tracker.

    Production adapters can replace this with ByteTrack/BoT-SORT without changing
    the pipeline. The reference version deliberately favors readability and tests.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 20) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._states: dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> tuple[Track, ...]:
        unmatched_track_ids = set(self._states)
        output: list[Track] = []

        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            candidates = [
                (track_id, detection.bbox.iou(self._states[track_id].track.bbox))
                for track_id in unmatched_track_ids
                if self._states[track_id].track.label == detection.label
            ]
            best_id, best_iou = max(candidates, key=lambda pair: pair[1], default=(-1, 0.0))
            if best_iou >= self.iou_threshold:
                previous = self._states[best_id].track
                track = Track(
                    track_id=best_id,
                    bbox=detection.bbox,
                    label=detection.label,
                    score=detection.score,
                    age=previous.age + 1,
                )
                unmatched_track_ids.remove(best_id)
            else:
                track = Track(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    label=detection.label,
                    score=detection.score,
                    age=1,
                )
                self._next_id += 1
            self._states[track.track_id] = _TrackState(track)
            output.append(track)

        for track_id in unmatched_track_ids:
            previous = self._states[track_id].track
            missed = previous.missed_frames + 1
            if missed > self.max_missed_frames:
                del self._states[track_id]
            else:
                self._states[track_id].track = Track(
                    track_id=previous.track_id,
                    bbox=previous.bbox,
                    label=previous.label,
                    score=previous.score,
                    age=previous.age + 1,
                    missed_frames=missed,
                )

        return tuple(sorted(output, key=lambda item: item.track_id))
