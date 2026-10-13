"""Compare completed native CPU/FP32, TensorRT/FP32 and TensorRT/FP16 runs.

Offline only. Gates are frozen before benchmarking; never promotes a model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

# Absolute AP units (0.005 = 0.5 percentage point), not relative percentages.
MAX_AP_DROP = .005
MAX_CLASS_AP_DROP = .02
MIN_P50_SPEEDUP = 2.


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare(cpu, fp32, fp16):
    panels = {"cpu_fp32": cpu, "trt_fp32": fp32, "trt_fp16": fp16}
    for name, report in panels.items():
        if report["kind"] != "images" or report["opencv_threads"] != 1:
            raise ValueError("Expected single-threaded image comparison")
        if any(report[key] != cpu[key] for key in ("manifest_sha256", "data_provenance", "opencv_version")):
            raise ValueError("Different data or OpenCV version")
        for key in ("model_sha256", "config_sha256", "confidence_threshold", "nms_iou_threshold", "input_size", "max_detections"):
            if report["provenance"][key] != cpu["provenance"][key]:
                raise ValueError("Model/config protocol mismatch")
        if report["performance"]["warmup_iterations"] != 5:
            raise ValueError("Expected five warmup iterations")
        latency = report["performance"]["detector"]
        if latency["samples"] != len(cpu["data_provenance"]["images"]):
            raise ValueError("Incomplete timings")
        for key in ("mean_ms", "p50_ms", "p95_ms"):
            if not math.isfinite(latency[key]) or latency[key] <= 0:
                raise ValueError("Invalid latency")
        for key in ("ap50", "ap50_95"):
            value = report["detection"][key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid AP")
        expected = {"cpu_fp32": ("opencv-dnn", "fp32"), "trt_fp32": ("tensorrt", "fp32"), "trt_fp16": ("tensorrt", "fp16-mixed")}[name]
        if (report["provenance"]["engine"], report["provenance"]["precision"]) != expected:
            raise ValueError("Backend/precision mislabeled")
    baseline_classes = {c["label"]: c["ap50_95"] for c in cpu["detection"]["per_class"]}
    result = {"gates": {"max_ap_drop": MAX_AP_DROP, "max_class_ap_drop": MAX_CLASS_AP_DROP,
                        "min_p50_speedup": MIN_P50_SPEEDUP, "p95_must_not_regress": True}, "panels": {}}
    for name, report in panels.items():
        latency = report["performance"]["detector"]
        scores = {c["label"]: c["ap50_95"] for c in report["detection"]["per_class"]}
        if set(scores) != set(baseline_classes):
            raise ValueError("Class set differs")
        deltas = {}
        for label, value in scores.items():
            base = baseline_classes[label]
            if base is None or value is None:
                if base != value:
                    raise ValueError("Class annotation scope differs")
                continue
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid class AP")
            deltas[label] = value - base
        ap_delta = report["detection"]["ap50_95"] - cpu["detection"]["ap50_95"]
        ap50_delta = report["detection"]["ap50"] - cpu["detection"]["ap50"]
        speedup = cpu["performance"]["detector"]["p50_ms"] / latency["p50_ms"]
        quality_ok = min(ap_delta, ap50_delta) >= -MAX_AP_DROP and all(v >= -MAX_CLASS_AP_DROP for v in deltas.values())
        speed_ok = speedup >= MIN_P50_SPEEDUP and latency["p95_ms"] <= cpu["performance"]["detector"]["p95_ms"]
        result["panels"][name] = {"ap50": report["detection"]["ap50"], "ap50_95": report["detection"]["ap50_95"],
            "ap_delta_vs_cpu": ap_delta, "class_ap_deltas": deltas, "operating_point": report["detection"]["operating_point"],
            "detector_latency": latency, "serial_detector_fps": 1000/latency["mean_ms"], "p50_speedup_vs_cpu": speedup,
            "quality_gate_passed": quality_ok, "speed_gate_passed": speed_ok}
    result["fp16_quality_and_speed_gate_passed"] = all(result["panels"]["trt_fp16"][k] for k in ("quality_gate_passed", "speed_gate_passed"))
    result["deployment_approved"] = False
    result["limitations"] = ["Observed 64-scene, eight-class development subset, not a fresh holdout or 80-class COCO score.",
        "CPU baseline uses one OpenCV thread. Serial detector FPS is not end-to-end camera/stream throughput.",
        "FP16 builder permits mixed precision. Engine I/O, preprocessing and NMS remain FP32/CPU.",
        "A passed local gate does not deploy the model or establish production reliability."]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "fp32", "fp16", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reports, provenance = [], {}
    for name in ("cpu", "fp32", "fp16"):
        directory = getattr(args, name)
        report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
        verification = json.loads((directory/"reference-validation.json").read_text(encoding="utf-8"))
        if verification["status"] != "passed":
            raise ValueError("Official reference validation missing")
        checked = {Path(p).resolve(): h for p, h in verification["input_sha256"].items()}
        for filename in ("report.json", "predictions.json"):
            path = (directory/filename).resolve()
            if checked.get(path) != digest(path):
                raise ValueError("Reference validation no longer matches results")
        reports.append(report)
        provenance[name] = {"report_sha256": digest(directory/"report.json"),
            "predictions_sha256": digest(directory/"predictions.json"),
            "reference_sha256": digest(directory/"reference-validation.json")}
    result = compare(*reports)
    result["provenance"] = provenance
    result["model_sha256"] = reports[0]["provenance"]["model_sha256"]
    result["manifest_sha256"] = reports[0]["manifest_sha256"]
    result["gpu"] = reports[2]["provenance"]["tensorrt"]
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result["panels"], indent=2))


if __name__ == "__main__":
    main()
