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


if __name__ == "__main__": unittest.main()
