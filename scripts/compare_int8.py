"""Frozen INT8-vs-FP16 experiment gate; no deployment and no threshold tuning."""
import argparse
import json
import math
from pathlib import Path
from compare_tensorrt import digest

MAX_AP_DROP = .005
MAX_CLASS_AP_DROP = .02
MIN_P50_SPEEDUP = 1.10


def inspect_int8(layers):
    counts = {}
    for layer in layers["Layers"]:
        for output in layer["Outputs"]:
            dtype = output["Format/Datatype"]
            counts[dtype] = counts.get(dtype, 0) + 1
    if not any("int8" in dtype.lower() for dtype in counts):
        raise ValueError("Inspector has no INT8 output definitions; builder flag alone is not evidence")
    return counts


def compare(fp16, int8, calibration):
    for report, precision in ((fp16, "fp16-mixed"), (int8, "int8-mixed")):
        if report["kind"] != "images" or report["opencv_threads"] != 1:
            raise ValueError("Expected single-threaded image benchmark")
        for key in ("manifest_sha256", "data_provenance", "opencv_version"):
            if report[key] != fp16[key]:
                raise ValueError("Different evaluation data/version")
        for key in ("model_sha256", "config_sha256", "confidence_threshold", "nms_iou_threshold", "input_size", "max_detections"):
            if report["provenance"][key] != fp16["provenance"][key]:
                raise ValueError("Model/config protocol mismatch")
        if report["provenance"]["engine"] != "tensorrt" or report["provenance"]["precision"] != precision:
            raise ValueError("Mislabeled engine/precision")
        for key in ("gpu", "compute_major", "compute_minor", "cuda_runtime", "tensorrt"):
            if report["provenance"]["tensorrt"][key] != fp16["provenance"]["tensorrt"][key]:
                raise ValueError("Different hardware/runtime")
        perf = report["performance"]
        if perf["warmup_iterations"] != 5 or perf["detector"]["samples"] != len(fp16["data_provenance"]["images"]):
            raise ValueError("Incomplete timings/warmup")
        for key in ("mean_ms", "p50_ms", "p95_ms"):
            if not math.isfinite(perf["detector"][key]) or perf["detector"][key] <= 0:
                raise ValueError("Invalid latency")
        for value in [report["detection"]["ap50"], report["detection"]["ap50_95"]] + [c["ap50_95"] for c in report["detection"]["per_class"]]:
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid AP")
    proof = int8["provenance"]["tensorrt"]["calibration"]
    if (calibration["version"] != 1 or calibration["kind"] != "int8-calibration" or
            calibration["algorithm"] != "entropy2" or calibration["batch_size"] != 1 or
            calibration["model_sha256"] != fp16["provenance"]["model_sha256"] or
            calibration["evaluation_manifest_sha256"] != fp16["manifest_sha256"] or
            proof["evaluation_manifest_sha256"] != fp16["manifest_sha256"] or
            proof["images"] != 128 or proof["algorithm"] != "entropy2" or proof["batch_size"] != 1):
        raise ValueError("Calibration provenance mismatch")
    images, evaluation = calibration["images"], fp16["data_provenance"]["images"]
    if len(images) != 128 or len(evaluation) != 64:
        raise ValueError("Incomplete calibration/evaluation")
    for key in ("coco_image_id", "sha256"):
        selected = {i[key] for i in images}
        excluded = {i[key] for i in calibration["excluded_images"]}
        actual = {i[key] for i in evaluation}
        if len(selected) != 128 or len(excluded) != 64 or excluded != actual or selected & actual:
            raise ValueError("Calibration/evaluation overlap or duplicate")
    if len({i["decoded_sha256"] for i in images}) != 128:
        raise ValueError("Duplicate decoded calibration image")
    base = {c["label"]: c["ap50_95"] for c in fp16["detection"]["per_class"]}
    scores = {c["label"]: c["ap50_95"] for c in int8["detection"]["per_class"]}
    if (base.keys() != scores.keys() or len(base) != len(fp16["detection"]["per_class"]) or
            len(scores) != len(int8["detection"]["per_class"])):
        raise ValueError("Class set differs/duplicate")
    deltas = {k: scores[k]-v for k, v in base.items()}
    ap_deltas = {k: int8["detection"][k]-fp16["detection"][k] for k in ("ap50", "ap50_95")}
    speedup = fp16["performance"]["detector"]["p50_ms"]/int8["performance"]["detector"]["p50_ms"]
    quality = min(ap_deltas.values()) >= -MAX_AP_DROP and min(deltas.values()) >= -MAX_CLASS_AP_DROP
    speed = speedup >= MIN_P50_SPEEDUP and int8["performance"]["detector"]["p95_ms"] <= fp16["performance"]["detector"]["p95_ms"]
    return {"gates": {"max_ap_drop": MAX_AP_DROP, "max_class_ap_drop": MAX_CLASS_AP_DROP,
            "min_p50_speedup": MIN_P50_SPEEDUP, "p95_must_not_regress": True},
        "ap_deltas_vs_fp16": ap_deltas, "class_ap_deltas_vs_fp16": deltas,
        "p50_speedup_vs_fp16": speedup, "quality_gate_passed": quality, "speed_gate_passed": speed,
        "quality_and_speed_gate_passed": quality and speed, "deployment_approved": False,
        "panels": {name: {"ap50": r["detection"]["ap50"], "ap50_95": r["detection"]["ap50_95"],
            "operating_point": r["detection"]["operating_point"], "detector_latency": r["performance"]["detector"],
            "serial_detector_fps": 1000/r["performance"]["detector"]["mean_ms"]}
            for name, r in (("fp16", fp16), ("int8", int8))},
        "limitations": calibration["limitations"] + ["Single sequential local measurement, not production camera FPS.",
            "Mixed precision with FP32 I/O and CPU preprocessing/NMS. No automatic deployment."]}


def validated_report(directory):
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    validation = json.loads((directory/"reference-validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "passed":
        raise ValueError("Official reference validation missing")
    checked = {Path(p).resolve(): h for p, h in validation["input_sha256"].items()}
    for name in ("report.json", "predictions.json"):
        if checked.get((directory/name).resolve()) != digest(directory/name):
            raise ValueError("Stale reference validation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fp16", "int8", "bundle", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    fp16, int8 = validated_report(args.fp16), validated_report(args.int8)
    metadata = json.loads((args.bundle/"manifest.json").read_text(encoding="utf-8"))
    recorded = int8["provenance"]["tensorrt"]
    for key in ("engine_sha256", "calibration", "precision", "model_sha256"):
        if metadata[key] != recorded[key]:
            raise ValueError("Engine bundle differs from benchmark")
    for name, expected in (("model.engine", recorded["engine_sha256"]),
            ("calibration.cache", recorded["calibration"]["cache_sha256"]),
            ("calibration-manifest.json", recorded["calibration"]["manifest_sha256"])):
        if digest(args.bundle/name) != expected:
            raise ValueError("Engine/calibration hash mismatch")
    calibration = json.loads((args.bundle/"calibration-manifest.json").read_text(encoding="utf-8"))
    result = compare(fp16, int8, calibration)
    result["inspector_output_definitions"] = inspect_int8(json.loads((args.bundle/"layers.json").read_text(encoding="utf-8")))
    result["inspector_sha256"] = digest(args.bundle/"layers.json")
    result.update(model_sha256=metadata["model_sha256"], manifest_sha256=fp16["manifest_sha256"], gpu=recorded)
    result["provenance"] = {name: {file: digest(directory/file) for file in
        ("report.json", "predictions.json", "reference-validation.json")} for name, directory in (("fp16", args.fp16), ("int8", args.int8))}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
