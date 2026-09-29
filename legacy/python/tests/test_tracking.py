import unittest

from visionforge.domain import BoundingBox, Detection
from visionforge.tracking import IoUTracker


class TrackerTests(unittest.TestCase):
    def test_preserves_id_for_overlapping_detection(self) -> None:
        tracker = IoUTracker(iou_threshold=0.3)
        first = tracker.update([Detection(BoundingBox(0, 0, 100, 100), "person", 0.9)])
        second = tracker.update([Detection(BoundingBox(5, 0, 105, 100), "person", 0.8)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].age, 2)

    def test_different_class_gets_new_id(self) -> None:
        tracker = IoUTracker()
        first = tracker.update([Detection(BoundingBox(0, 0, 10, 10), "person", 0.9)])
        second = tracker.update([Detection(BoundingBox(0, 0, 10, 10), "car", 0.9)])
        self.assertNotEqual(first[0].track_id, second[0].track_id)


if __name__ == "__main__":
    unittest.main()

