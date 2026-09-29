import unittest

from visionforge.domain import BoundingBox
from visionforge.geometry import transform_point


class BoundingBoxTests(unittest.TestCase):
    def test_iou(self) -> None:
        first = BoundingBox(0, 0, 10, 10)
        second = BoundingBox(5, 5, 15, 15)
        self.assertAlmostEqual(first.iou(second), 25 / 175)

    def test_invalid_box(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(10, 0, 5, 10)

    def test_homography_translation(self) -> None:
        matrix = ((1, 0, 10), (0, 1, -5), (0, 0, 1))
        self.assertEqual(transform_point((2, 3), matrix), (12.0, -2.0))


if __name__ == "__main__":
    unittest.main()

