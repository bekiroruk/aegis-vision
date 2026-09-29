import unittest

from visionforge.vector_store import InMemoryVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_returns_nearest_item_first(self) -> None:
        store = InMemoryVectorStore()
        store.upsert("red", [1, 0], {"label": "red"})
        store.upsert("blue", [0, 1], {"label": "blue"})
        results = store.search([0.9, 0.1])
        self.assertEqual(results[0].item_id, "red")

    def test_rejects_mixed_dimensions(self) -> None:
        store = InMemoryVectorStore()
        store.upsert("a", [1, 0], {})
        with self.assertRaises(ValueError):
            store.upsert("b", [1, 0, 0], {})


if __name__ == "__main__":
    unittest.main()

