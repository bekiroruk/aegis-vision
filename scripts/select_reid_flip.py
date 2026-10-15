"""Select the fixed augmentation candidate using development reports only.

No inference, training, threshold search, holdout access or runtime promotion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def select(policy, baseline, candidate):
    if (type(policy["minimum_counterpart_queries"]) is not int or policy["minimum_counterpart_queries"] < 1
            or type(policy["maximum_extra_wrong_accepts"]) is not int or policy["maximum_extra_wrong_accepts"] < 0
            or type(policy["minimum_rank1_gain"]) not in (int, float)
            or not math.isfinite(policy["minimum_rank1_gain"]) or not 0 < policy["minimum_rank1_gain"] <= 1):
        raise ValueError("Invalid selection policy bounds")
    for variant, report in (("single", baseline), ("flip_mean", candidate)):
        if (report.get("complete") is not True or report.get("feature_variant") != variant
                or report.get("protocol") != policy["development_protocol"]
                or report.get("min_similarity") != policy["min_similarity"]
                or report.get("ambiguity_margin") != policy["ambiguity_margin"]):
            raise ValueError("Only complete fixed-threshold development reports are selectable")
        metrics = report["metrics"]
        for name in ("queries", "queries_with_counterpart", "rank1_correct", "correct_accepts", "wrong_accepts"):
            if type(metrics[name]) is not int or not 0 <= metrics[name] <= 512 * 39:
                raise ValueError("Invalid metric counts")
        if not (metrics["rank1_correct"] <= metrics["queries_with_counterpart"] <= metrics["queries"]):
            raise ValueError("Inconsistent rank-1 counts")
        if metrics["correct_accepts"] + metrics["wrong_accepts"] > metrics["queries"]:
            raise ValueError("Inconsistent accept counts")
        if metrics["correct_accepts"] > metrics["queries_with_counterpart"]:
            raise ValueError("Inconsistent true accepts")
    for field in ("manifest_sha256", "model_sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError("Reports do not share identical data and weights")
    if baseline["embedding_space"] == candidate["embedding_space"]:
        raise ValueError("Different feature protocols must have different spaces")
    a, b = baseline["metrics"], candidate["metrics"]
    for field in ("queries", "queries_with_counterpart"):
        if a[field] != b[field]:
            raise ValueError("Reports score different query populations")
    shared = a["queries_with_counterpart"]
    gain = (b["rank1_correct"] - a["rank1_correct"]) / shared if shared else None
    enough = shared >= policy["minimum_counterpart_queries"]
    improves = gain is not None and gain + 1e-12 >= policy["minimum_rank1_gain"]
    safe = b["wrong_accepts"] - a["wrong_accepts"] <= policy["maximum_extra_wrong_accepts"]
    selected = enough and improves and safe
    return dict(version=1, selected_variant="flip_mean" if selected else "single",
        candidate_selected=selected, holdout_required=selected, production_promoted=False,
        counterpart_queries=shared, rank1_gain=gain,
        gates=dict(enough_queries=enough, rank1_improvement=improves, no_extra_wrong_accepts=safe),
        baseline_metrics=a, candidate_metrics=b,
        warning="Development selection only; not evidence of holdout gain or production readiness.")


def load(path):
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Missing/oversized JSON input")
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must not exist")
    policy, ph = load(args.policy)
    base, bh = load(args.baseline)
    candidate, ch = load(args.candidate)
    result = select(policy, base, candidate)
    result["input_sha256"] = dict(policy=ph, baseline=bh, candidate=ch)
    result["policy"] = policy
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
