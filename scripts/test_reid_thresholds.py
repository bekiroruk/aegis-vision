import copy
import unittest
from select_reid_thresholds import select


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(version=1, purpose="development-threshold-calibration-only",
            development_protocol="wildtrack-flip-development-v1", feature_variant="single",
            production_promotion=False, manifest_sha256="a" * 64, model_sha256="b" * 64,
            thresholds=[.6, .85], margins=[.05], minimum_counterpart_queries=30,
            minimum_unknown_queries=30, minimum_correct_accepts=20, minimum_precision=.95,
            minimum_recall=.2, maximum_wrong_accepts=1, maximum_unknown_accepts=0)
        self.report = dict(version=1, complete=True, development_only=True, production_promoted=False,
            holdout_evaluated=False, protocol=self.policy["development_protocol"], policy_sha256="policy",
            feature_variant="single", manifest_sha256="a" * 64, model_sha256="b" * 64,
            cells=[self.cell(.6, 25, 1), self.cell(.85, 0, 0)])

    def cell(self, threshold, tp, fp):
        return dict(min_similarity=threshold, ambiguity_margin=.05, metrics=dict(queries=100,
            queries_with_counterpart=50, queries_without_counterpart=50, rank1_correct=30,
            correct_accepts=tp, wrong_accepts=fp, missed_counterparts=50-tp, correct_unknown_rejections=50),
            rejection_reasons=dict(matched=tp+fp, below_threshold=100-tp-fp))

    def result(self):
        return select(self.policy, self.report, "policy")

    def test_selection_never_promotes(self):
        result = self.result()
        self.assertEqual(result["selected"]["min_similarity"], .6)
        self.assertTrue(result["fresh_final_test_required"])
        self.assertFalse(result["production_promoted"])

    def test_no_accepts_is_not_perfect_precision(self):
        self.report["cells"][0] = self.cell(.6, 0, 0)
        result = self.result()
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["cells"][0]["precision"])

    def test_unknown_accepts_fail_even_when_precision_passes(self):
        self.report["cells"][0]["metrics"]["correct_unknown_rejections"] = 49
        self.assertFalse(self.result()["candidate_selected"])

    def test_precision_and_wrong_accept_gate(self):
        self.report["cells"][0] = self.cell(.6, 25, 2)
        self.assertFalse(self.result()["candidate_selected"])

    def test_insufficient_true_accepts(self):
        self.report["cells"][0] = self.cell(.6, 19, 0)
        self.assertFalse(self.result()["candidate_selected"])

    def test_stricter_threshold_breaks_tie(self):
        self.report["cells"][1] = self.cell(.85, 25, 1)
        self.assertEqual(self.result()["selected"]["min_similarity"], .85)

    def test_holdout_and_mismatched_provenance_rejected(self):
        for key, value in (("protocol", "wildtrack-flip-holdout-v1"), ("policy_sha256", "changed"),
                ("model_sha256", "c" * 64), ("manifest_sha256", "c" * 64),
                ("complete", False), ("holdout_evaluated", True), ("feature_variant", "flip_mean")):
            report = copy.deepcopy(self.report); report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select(self.policy, report, "policy")

    def test_missing_duplicate_and_unexpected_cells_rejected(self):
        for cells in ([self.report["cells"][0]], [self.report["cells"][0]] * 2,
                      [self.cell(.55, 25, 1), self.report["cells"][1]]):
            report = dict(self.report, cells=cells)
            with self.assertRaises(ValueError): select(self.policy, report, "policy")

    def test_bad_metrics_and_reason_totals(self):
        for key, value in (("correct_accepts", -1), ("correct_accepts", True), ("correct_accepts", 31),
                           ("queries", 101), ("missed_counterparts", 0), ("correct_unknown_rejections", 48)):
            report = copy.deepcopy(self.report); report["cells"][0]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): select(self.policy, report, "policy")
        self.report["cells"][0]["rejection_reasons"]["matched"] = 100
        with self.assertRaises(ValueError): self.result()

    def test_rank1_must_be_constant_over_thresholds(self):
        self.report["cells"][1]["metrics"]["rank1_correct"] = 29
        with self.assertRaises(ValueError): self.result()

    def test_invalid_grid_and_policy(self):
        for key, value in (("version", True), ("thresholds", [.6, .6]), ("margins", [True]), ("margins", [float("nan")]),
                           ("minimum_precision", float("nan")), ("minimum_correct_accepts", 0),
                           ("production_promotion", True), ("maximum_wrong_accepts", False)):
            policy = dict(self.policy, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): select(policy, self.report, "policy")

    def test_empty_population_and_zero_precision_denominator(self):
        for cell in self.report["cells"]:
            cell["metrics"] = {key: 0 for key in cell["metrics"]}
            cell["rejection_reasons"] = {}
        result = self.result()
        self.assertIsNone(result["selected"])
        self.assertIsNone(result["cells"][0]["recall"])

    def test_recall_gate_cannot_be_bypassed_by_high_precision(self):
        self.policy["minimum_recall"] = .6
        self.report["cells"][0] = self.cell(.6, 25, 0)
        self.assertFalse(self.result()["candidate_selected"])


if __name__ == "__main__":
    unittest.main()
