import unittest

from visionforge.domain import Frame
from visionforge.factory import build_demo_pipeline


class PipelineTests(unittest.TestCase):
    def test_analyzes_tracks_and_indexes(self) -> None:
        pipeline = build_demo_pipeline()
        frame = Frame(
            frame_id="1",
            source_id="cam",
            timestamp_ms=0,
            payload={
                "detections": [
                    {"bbox": [0, 0, 10, 10], "label": "person", "score": 0.9}
                ]
            },
        )
        result = pipeline.analyze(frame)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.indexed_items, ("cam:1:0",))
        self.assertIsNotNone(result.detections[0].embedding)

        query = pipeline.embedder.embed_text("person")
        matches = pipeline.vector_store.search(query)
        self.assertEqual(matches[0].item_id, "cam:1:0")


if __name__ == "__main__":
    unittest.main()
