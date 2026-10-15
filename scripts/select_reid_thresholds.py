"""Choose development-only Re-ID thresholds; never promote or read holdout data."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Missing or oversized JSON")
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def axis(values):
    if (not isinstance(values, list) or not 1 <= len(values) <= 8
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in values)
            or values != sorted(set(values))):
        raise ValueError("Invalid calibration grid")
    return values


def select(policy, report, policy_sha256):
    if (type(policy.get("version")) is not int or policy["version"] != 1 or policy.get("purpose") != "development-threshold-calibration-only"
            or policy.get("development_protocol") != "wildtrack-flip-development-v1"
            or policy.get("feature_variant") != "single" or policy.get("production_promotion") is not False):
        raise ValueError("Only the development single-feature protocol is supported")
    if (type(report.get("version")) is not int or report["version"] != 1 or report.get("complete") is not True
            or report.get("development_only") is not True or report.get("production_promoted") is not False
            or report.get("holdout_evaluated") is not False or report.get("feature_variant") != "single"
            or report.get("protocol") != policy["development_protocol"]
            or report.get("policy_sha256") != policy_sha256):
        raise ValueError("Incomplete, mismatched or non-development report")
    for key in ("manifest_sha256", "model_sha256"):
        pin = policy[key]
        if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError("Invalid model/data pin")
        if report.get(key) != pin:
            raise ValueError("Calibration model/data pin mismatch")
    for key in ("minimum_counterpart_queries", "minimum_unknown_queries", "minimum_correct_accepts",
                "maximum_wrong_accepts", "maximum_unknown_accepts"):
        if type(policy[key]) is not int or not 0 <= policy[key] <= 512 * 39:
            raise ValueError("Invalid policy count")
    if min(policy[k] for k in ("minimum_counterpart_queries", "minimum_unknown_queries", "minimum_correct_accepts")) < 1:
        raise ValueError("Policy must require positive evidence")
    for key in ("minimum_precision", "minimum_recall"):
        if type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or not 0 < policy[key] <= 1:
            raise ValueError("Invalid policy rate")
    expected = {(t, m) for t in axis(policy["thresholds"]) for m in axis(policy["margins"])}
    if not isinstance(report.get("cells"), list) or len(report["cells"]) != len(expected):
        raise ValueError("Incomplete calibration grid")
    seen, scored, population = set(), [], None
    count_names = ("queries", "queries_with_counterpart", "queries_without_counterpart", "rank1_correct",
                   "correct_accepts", "wrong_accepts", "missed_counterparts", "correct_unknown_rejections")
    reasons = {"no_gallery", "below_threshold", "ambiguous_query", "not_mutual_best", "ambiguous_gallery", "matched"}
    for cell in report["cells"]:
        pair = (cell["min_similarity"], cell["ambiguity_margin"])
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pair) or pair not in expected or pair in seen:
            raise ValueError("Unexpected or duplicate calibration cell")
        seen.add(pair)
        c = cell["metrics"]
        if any(type(c[k]) is not int or not 0 <= c[k] <= 512 * 39 for k in count_names):
            raise ValueError("Invalid metric count")
        q, shared, unknown, rank1, tp, fp, missed, rejected = (c[k] for k in count_names)
        if (q != shared + unknown or not tp <= rank1 <= shared or tp + fp > q
                or missed != shared - tp or rejected > unknown or unknown - rejected > fp):
            raise ValueError("Inconsistent metric counts")
        identity = (q, shared, unknown, rank1)
        if population is not None and identity != population:
            raise ValueError("Grid changed the query population or rank-1")
        population = identity
        rs = cell["rejection_reasons"]
        if (not isinstance(rs, dict) or not set(rs) <= reasons
                or any(type(v) is not int or v < 0 for v in rs.values())
                or sum(rs.values()) != q or rs.get("matched", 0) != tp + fp):
            raise ValueError("Inconsistent decision reasons")
        precision, recall = (tp / (tp + fp) if tp + fp else None), (tp / shared if shared else None)
        unknown_accepts = unknown - rejected
        gates = dict(enough_counterparts=shared >= policy["minimum_counterpart_queries"],
            enough_unknowns=unknown >= policy["minimum_unknown_queries"],
            enough_correct_accepts=tp >= policy["minimum_correct_accepts"],
            precision=precision is not None and precision + 1e-12 >= policy["minimum_precision"],
            recall=recall is not None and recall + 1e-12 >= policy["minimum_recall"],
            wrong_accepts=fp <= policy["maximum_wrong_accepts"],
            unknown_accepts=unknown_accepts <= policy["maximum_unknown_accepts"])
        scored.append(dict(min_similarity=pair[0], ambiguity_margin=pair[1], correct_accepts=tp,
            wrong_accepts=fp, unknown_accepts=unknown_accepts, precision=precision, recall=recall,
            gates=gates, eligible=all(gates.values())))
    eligible = [row for row in scored if row["eligible"]]
    chosen = max(eligible, key=lambda r: (r["correct_accepts"], -r["wrong_accepts"],
        r["min_similarity"], r["ambiguity_margin"])) if eligible else None
    return dict(version=1, development_only=True, selected=chosen, cells=scored,
        candidate_selected=chosen is not None, fresh_final_test_required=chosen is not None,
        production_promoted=False, policy_sha256=policy_sha256,
        warning="Development fit only, not independent accuracy. Repeated people are correlated. Defaults unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    policy, ph = load(args.policy)
    report, rh = load(args.report)
    result = select(policy, report, ph)
    result["report_sha256"] = rh
    result["policy"] = policy
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "policy")}, indent=2))


if __name__ == "__main__":
    main()
