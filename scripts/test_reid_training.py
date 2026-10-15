"""Training protocol/sampler checks; torch tests are optional in lightweight CI."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest
from train_reid_metric import load_json, validate_policy, sample_plan, sample_batch, batch_hard_loss


def policy():
    return load_json(Path(__file__).resolve().parents[1] / "configs/reid-training.json")


def dataset():
    return dict(version=1, kind="video", classes=["person"], width=640, height=480, frames=16,
        archive_gt_entry="MOT15Labels/train/TUD-fixture/gt/gt.txt",
        ground_truth=[dict(frame_index=f, id=i, bbox=[0, 0, 40, 100])
                      for f in (1, 6, 11, 16) for i in range(4)])


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_and_only_two_training_sources(self):
        p = policy(); validate_policy(p)
        self.assertEqual(p["steps"], 200)
        for path in ("wildtrack/manifest.json", "artifacts/datasets/mot15-transfer-pets-v1/quality-manifest.json"):
            bad = copy.deepcopy(p); bad["training_sources"][0]["manifest"] = path
            with self.assertRaises(ValueError): validate_policy(bad)

    def test_invalid_training_parameters(self):
        for key, value in (("steps", True), ("steps", 1001), ("seed", -1), ("frame_stride", 0),
                ("learning_rate", float("nan")), ("identity_regularization", 0),
                ("observations_per_identity", 1), ("production_promotion", True), ("base_model_sha256", "x")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy(dict(policy(), **{key: value}))

    def test_sample_plan_is_fixed_and_clipped(self):
        data = dataset()
        data["ground_truth"][0]["bbox"] = [-5.2, -1, 40.1, 100.2]
        rows = sample_plan(data, policy())
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["bbox"], [0, 0, 41, 101])
        self.assertEqual([r["frame"] for r in rows[::4]], [1, 6, 11, 16])

    def test_duplicate_and_bad_boxes_rejected(self):
        data = dataset(); data["ground_truth"].append(data["ground_truth"][0])
        with self.assertRaises(ValueError): sample_plan(data, policy())
        for box in ([0, 0, float("nan"), 100], [0, 0, 0, 100], [False, 0, 40, 100]):
            data = dataset(); data["ground_truth"][0]["bbox"] = box
            with self.assertRaises(ValueError): sample_plan(data, policy())

    def test_singleton_tracks_not_used_as_positives(self):
        data = dataset(); data["ground_truth"].pop()
        with self.assertRaisesRegex(ValueError, "Insufficient identities"):
            sample_plan(data, policy())

    def test_test_sequences_cannot_be_training_metadata(self):
        data = dataset(); data["archive_gt_entry"] = "MOT15Labels/test/TUD-fixture/gt/gt.txt"
        with self.assertRaises(ValueError): sample_plan(data, policy())

    def test_sampler_is_seeded_distinct_and_sequence_local(self):
        groups = {name: {i: list(range(base+i*4, base+(i+1)*4)) for i in range(4)}
                  for name, base in (("first", 0), ("second", 16))}
        a = sample_batch(groups, policy(), random.Random(17))
        self.assertEqual(a, sample_batch(groups, policy(), random.Random(17)))
        indices, labels, sequence = a
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual([labels.count(i) for i in range(4)], [4]*4)
        self.assertTrue(all(i < 16 for i in indices) if sequence == "first" else all(i >= 16 for i in indices))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch preparation environment unavailable")
class LossTests(unittest.TestCase):
    def test_hard_pairs_and_gradients_match_explicit_reference(self):
        import torch
        import torch.nn.functional as F
        x = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [.6, .8]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        actual = batch_hard_loss(x, labels, .2)
        expected = F.triplet_margin_loss(x, x[[1, 0, 3, 2]], x[[3, 3, 1, 1]], margin=.2)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.norm()), 0)

    def test_each_anchor_needs_positive_and_negative(self):
        import torch
        for labels in ([0, 0, 0, 0], [0, 1, 2, 3], [0, 0, 1, 2]):
            with self.assertRaises(ValueError): batch_hard_loss(torch.eye(4), torch.tensor(labels), .2)

    def test_bias_free_export_normalization_equivalence(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(17)
        head = torch.nn.Linear(512, 512, bias=False)
        raw = torch.randn(4, 512) * 100
        a, b = F.normalize(head(raw), dim=1), F.normalize(head(F.normalize(raw, dim=1)), dim=1)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


if __name__ == "__main__": unittest.main()
