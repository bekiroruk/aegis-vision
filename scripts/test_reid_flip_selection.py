import copy
import unittest
from select_reid_flip import select


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(development_protocol="dev", min_similarity=.85, ambiguity_margin=.05,
            minimum_counterpart_queries=30, minimum_rank1_gain=.02, maximum_extra_wrong_accepts=0)
        self.base = dict(complete=True, feature_variant="single", protocol="dev", min_similarity=.85,
            ambiguity_margin=.05, manifest_sha256="data", model_sha256="model", embedding_space="base",
            metrics=dict(queries=100, queries_with_counterpart=50, rank1_correct=20, correct_accepts=2, wrong_accepts=0))
        self.candidate = copy.deepcopy(self.base)
        self.candidate.update(feature_variant="flip_mean", embedding_space="flip")

    def test_gain_boundary(self):
        self.candidate["metrics"]["rank1_correct"] = 21
        result = select(self.policy, self.base, self.candidate)
        self.assertTrue(result["candidate_selected"])
        self.assertTrue(result["holdout_required"])
        self.assertFalse(result["production_promoted"])

    def test_no_gain(self):
        self.assertFalse(select(self.policy, self.base, self.candidate)["candidate_selected"])

    def test_extra_wrong_accept(self):
        self.candidate["metrics"].update(rank1_correct=40, wrong_accepts=1)
        self.assertFalse(select(self.policy, self.base, self.candidate)["candidate_selected"])

    def test_small_population(self):
        self.base["metrics"]["queries_with_counterpart"] = 29
        self.candidate["metrics"].update(queries_with_counterpart=29, rank1_correct=29)
        self.assertFalse(select(self.policy, self.base, self.candidate)["candidate_selected"])

    def test_holdout_rejected(self):
        self.candidate["protocol"] = "holdout"
        with self.assertRaises(ValueError): select(self.policy, self.base, self.candidate)

    def test_mismatched_data(self):
        self.candidate["manifest_sha256"] = "other"
        with self.assertRaises(ValueError): select(self.policy, self.base, self.candidate)

    def test_mixed_queries(self):
        self.candidate["metrics"]["queries"] = 99
        with self.assertRaises(ValueError): select(self.policy, self.base, self.candidate)

    def test_impossible_count(self):
        self.candidate["metrics"]["rank1_correct"] = 51
        with self.assertRaises(ValueError): select(self.policy, self.base, self.candidate)

    def test_empty_data(self):
        for r in (self.base, self.candidate):
            r["metrics"] = dict(queries=0, queries_with_counterpart=0, rank1_correct=0, correct_accepts=0, wrong_accepts=0)
        result = select(self.policy, self.base, self.candidate)
        self.assertIsNone(result["rank1_gain"])
        self.assertFalse(result["candidate_selected"])

    def test_invalid_policy(self):
        for value in (float("nan"), True, 0, -1):
            self.policy["minimum_rank1_gain"] = value
            with self.assertRaises(ValueError): select(self.policy, self.base, self.candidate)


class ModelSelectionTests(SelectionTests):
    def setUp(self):
        super().setUp()
        self.policy.update(comparison="model", baseline_model_sha256="a" * 64,
            candidate_model_sha256="b" * 64, development_manifest_sha256="c" * 64)
        self.base.update(model_sha256="a" * 64, manifest_sha256="c" * 64)
        self.candidate.update(model_sha256="b" * 64, manifest_sha256="c" * 64, feature_variant="single")

    def test_selected_model_is_recorded_without_promotion(self):
        self.candidate["metrics"]["rank1_correct"] = 21
        result = select(self.policy, self.base, self.candidate)
        self.assertEqual(result["selected_model_sha256"], "b" * 64)
        self.assertEqual(result["selected_variant"], "single")
        self.assertFalse(result["production_promoted"])

    def test_weights_cannot_be_swapped_or_substituted(self):
        for key, value in (("model_sha256", "d" * 64), ("feature_variant", "flip_mean"),
                           ("embedding_space", "base")):
            candidate = copy.deepcopy(self.candidate)
            candidate[key] = value
            with self.assertRaises(ValueError): select(self.policy, self.base, candidate)

    def test_policy_pins_must_be_valid_and_distinct(self):
        for key, value in (("candidate_model_sha256", "a" * 64), ("candidate_model_sha256", "x" * 64),
                           ("development_manifest_sha256", "d" * 64), ("comparison", "unknown")):
            policy = dict(self.policy, **{key: value})
            with self.assertRaises(ValueError): select(policy, self.base, self.candidate)


if __name__ == "__main__": unittest.main()
