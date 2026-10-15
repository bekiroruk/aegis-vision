"""Bounded local research: frozen OSNet + GPU-trained linear triplet head.

Reads only two pinned MOT15 TUD training sequences, never WILDTRACK. Exports
native C++ ONNX with golden references; no automatic deployment or downloads.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import time

from export_reid import (ARCHITECTURE_SHA256, ARCHITECTURE_SIZE, LICENSE_SHA256, LICENSE_SIZE,
    MODEL_PROFILES, architecture_tree, file_sha256, load_reid_weights, preprocessing,
    publish_bundle, restricted_checkpoint, verify_file, write_json)

BASE_HASH = "5bf5321c31d064507acaec3aefecf3381b24673123c61403ceab491f49f9dc6a"
SOURCES = {
    "artifacts/datasets/mot15-tud/quality-manifest.json": "adc32ea13e1338d083968d2278a87487136f764beb13b1374a24ab533c009187",
    "artifacts/datasets/mot15-campus/quality-manifest.json": "d880b78c006938ddcbf13cabd00cba61b6dda3f6d6c9efe6c7e5e6eccbb0ff25",
}


def load_json(path, maximum=16 * 1024 * 1024):
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Missing or oversized JSON input")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_policy(p):
    if (type(p.get("version")) is not int or p["version"] != 1 or p.get("method") != "linear-triplet-v1"
            or p.get("base_model_sha256") != BASE_HASH or p.get("production_promotion") is not False
            or p.get("checkpoint") != "last-step-only-no-early-stopping"):
        raise ValueError("Unsupported training protocol")
    if (not isinstance(p.get("training_sources"), list) or len(p["training_sources"]) != 2
            or {x["manifest"]: x["sha256"] for x in p["training_sources"]} != SOURCES):
        raise ValueError("Only the two pinned MOT15 TUD sources may be trained")
    for key, low, high in (("seed", 0, 2**31-1), ("frame_stride", 1, 20),
            ("minimum_crop_width", 20, 128), ("minimum_crop_height", 64, 256),
            ("identities_per_batch", 2, 8), ("observations_per_identity", 2, 8), ("steps", 1, 1000)):
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError("Invalid bounded training count")
    for key, low, high in (("learning_rate", 0, .01), ("triplet_margin", 0, 1),
                           ("identity_regularization", 0, 1)):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or not low < p[key] <= high:
            raise ValueError("Invalid bounded training rate")


def sample_plan(data, p):
    if (data.get("version") != 1 or data.get("kind") != "video" or data.get("classes") != ["person"]
            or not data.get("archive_gt_entry", "").startswith("MOT15Labels/train/TUD-")
            or data.get("width") != 640 or data.get("height") != 480
            or type(data.get("frames")) is not int or not 1 <= data["frames"] <= 1000
            or not isinstance(data.get("ground_truth"), list) or len(data["ground_truth"]) > 50000):
        raise ValueError("Unsupported bounded TUD training metadata")
    seen, selected = set(), []
    for row in data["ground_truth"]:
        frame, identity, raw = row["frame_index"], row["id"], row["bbox"]
        if (type(frame) is not int or not 1 <= frame <= data["frames"] or type(identity) is not int
                or not 0 <= identity <= 1000000 or (frame, identity) in seen):
            raise ValueError("Invalid or duplicate training observation")
        seen.add((frame, identity))
        if (not isinstance(raw, list) or len(raw) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in raw)
                or raw[2] <= raw[0] or raw[3] <= raw[1]):
            raise ValueError("Invalid training box")
        if (frame - 1) % p["frame_stride"]:
            continue
        box = [max(0, math.floor(raw[0])), max(0, math.floor(raw[1])),
               min(640, math.ceil(raw[2])), min(480, math.ceil(raw[3]))]
        if box[2] - box[0] >= p["minimum_crop_width"] and box[3] - box[1] >= p["minimum_crop_height"]:
            selected.append(dict(frame=frame, identity=identity, bbox=box))
    counts = Counter(r["identity"] for r in selected)
    selected = [r for r in selected if counts[r["identity"]] >= p["observations_per_identity"]]
    if len({r["identity"] for r in selected}) < p["identities_per_batch"] or len(selected) > 2048:
        raise ValueError("Insufficient identities or oversized training subset")
    return sorted(selected, key=lambda r: (r["frame"], r["identity"]))


def sample_batch(groups, p, rng):
    # No assertion that person IDs from different sequences identify different
    # humans. Mine only inside one sequence, where GT supplies that information.
    sequence = rng.choice(sorted(groups))
    identities = rng.sample(sorted(groups[sequence]), p["identities_per_batch"])
    indices, labels = [], []
    for label, identity in enumerate(identities):
        indices.extend(rng.sample(groups[sequence][identity], p["observations_per_identity"]))
        labels.extend([label] * p["observations_per_identity"])
    return indices, labels, sequence


def batch_hard_loss(embeddings, labels, margin):
    import torch
    import torch.nn.functional as F
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Invalid metric batch shape")
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    if not bool(positive.any(1).all() and negative.any(1).all()):
        raise ValueError("Each anchor needs a distinct positive and a negative")
    with torch.no_grad():
        distances = torch.cdist(embeddings, embeddings)
        pos = distances.masked_fill(~positive, -1).argmax(1)
        neg = distances.masked_fill(~negative, float("inf")).argmin(1)
    return F.triplet_margin_loss(embeddings, embeddings[pos], embeddings[neg], margin=margin, p=2)


def train(root, bundle, policy_path, output):
    import cv2
    import hashlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    p = load_json(policy_path, 65536)
    validate_policy(p)
    if output.exists():
        raise ValueError("Use a new training output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no silent CPU training fallback")
    if file_sha256(bundle / "model.onnx") != p["base_model_sha256"]:
        raise ValueError("Base model differs from frozen training policy")
    verify_file(bundle / "osnet-source.py", ARCHITECTURE_SIZE, ARCHITECTURE_SHA256)
    verify_file(bundle / "LICENSE-MIT.txt", LICENSE_SIZE, LICENSE_SHA256)
    filename, checkpoint_hash, checkpoint_size = MODEL_PROFILES["osnet_x1_0"]
    verify_file(bundle / filename, checkpoint_size, checkpoint_hash)
    plans = []
    for source in p["training_sources"]:
        path = (root / source["manifest"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != source["sha256"]:
            raise ValueError("Training manifest pin/path mismatch")
        data = load_json(path)
        video = (path.parent / data["video"]).resolve(strict=True)
        if (video.parent != path.parent or video.stat().st_size > 32 * 1024 * 1024
                or file_sha256(video) != data["video_sha256"]):
            raise ValueError("Training video hash/path/size mismatch")
        plans.append((path, video, data, sample_plan(data, p)))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(p["seed"])
    torch.cuda.manual_seed_all(p["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    namespace = {"__name__": "aegis_pinned_osnet"}
    exec(compile(architecture_tree((bundle / "osnet-source.py").read_text(encoding="utf-8"), "osnet_x1_0"),
                 "osnet-pinned.py", "exec"), namespace)
    backbone = namespace["osnet_x1_0"](num_classes=1, pretrained=False).eval()
    loaded = load_reid_weights(backbone, restricted_checkpoint(bundle / filename, torch), torch)
    backbone.requires_grad_(False).cuda()
    backbone_before = {name: tensor.detach().cpu().clone() for name, tensor in backbone.state_dict().items()}
    feature_rows, observations = [], []
    groups = defaultdict(lambda: defaultdict(list))
    started = time.monotonic()
    with torch.no_grad():
        for path, video, data, plan in plans:
            by_frame = defaultdict(list)
            for row in plan: by_frame[row["frame"]].append(row)
            capture = cv2.VideoCapture(str(video))
            try:
                for frame in range(1, data["frames"] + 1):
                    ok, image = capture.read()
                    if not ok or image.shape != (480, 640, 3):
                        raise ValueError("Training decode shape/frame-count mismatch")
                    for row in by_frame[frame]:
                        x1, y1, x2, y2 = row["bbox"]
                        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                        raw = backbone(torch.from_numpy(preprocessing(crop)).cuda())
                        if raw.shape != (1, 512) or not bool(torch.isfinite(raw).all()) or float(raw.norm()) <= 1e-12:
                            raise ValueError("Invalid training feature")
                        feature_rows.append(F.normalize(raw, dim=1).cpu())
                        groups[data["dataset"]][row["identity"]].append(len(feature_rows)-1)
                        observations.append(dict(sequence=data["dataset"], **row,
                            crop_pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
                if capture.read()[0]: raise ValueError("Training video has extra frames")
            finally:
                capture.release()
            print(f"Prepared {data['dataset']}: {len(plan)} crops, {len(groups[data['dataset']])} track IDs", flush=True)
    dataset = dict(version=1, purpose="training-only", sources=p["training_sources"], observations=observations,
                   warning="Single-camera GT tracks; IDs not globally unique. No WILDTRACK used.")
    write_json(output / "training-data.json", dataset)
    x = torch.cat(feature_rows).cuda()
    head = torch.nn.Linear(512, 512, bias=False).cuda()
    identity = torch.eye(512, device="cuda")
    with torch.no_grad(): head.weight.copy_(identity)
    optimizer = torch.optim.Adam(head.parameters(), lr=p["learning_rate"])
    rng, losses = random.Random(p["seed"]), []
    torch.cuda.reset_peak_memory_stats()
    for step in range(p["steps"]):
        indices, labels, sequence = sample_batch(groups, p, rng)
        features = F.normalize(head(x[indices]), dim=1)
        triplet = batch_hard_loss(features, torch.tensor(labels, device="cuda"), p["triplet_margin"])
        penalty = p["identity_regularization"] * (head.weight - identity).square().sum()
        loss = triplet + penalty
        if not bool(torch.isfinite(loss)): raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not bool(torch.isfinite(head.weight.grad).all()): raise ValueError("Non-finite training gradient")
        optimizer.step()
        losses.append(dict(step=step+1, sequence=sequence, triplet=float(triplet.detach()), loss=float(loss.detach())))
        if (step+1) % 50 == 0: print(f"GPU step {step+1}/{p['steps']}: loss={losses[-1]['loss']:.6f}", flush=True)
    if any(not torch.equal(tensor.detach().cpu(), backbone_before[name]) for name, tensor in backbone.state_dict().items()):
        raise ValueError("Frozen backbone changed")
    delta = float((head.weight.detach() - identity).norm())
    if not math.isfinite(delta) or delta <= 0: raise ValueError("No valid learned weight update")
    weights = head.weight.detach().cpu().numpy().astype("<f4")
    (output / "metric-head.f32").write_bytes(weights.tobytes())
    adaptation = dict(method="linear-triplet-v1", base_model_sha256=BASE_HASH,
        training_policy_sha256=file_sha256(policy_path), training_data_sha256=file_sha256(output / "training-data.json"),
        head_sha256=file_sha256(output / "metric-head.f32"))
    write_json(output / "training-report.json", dict(version=1, complete=True, policy=p, adaptation=adaptation,
        torch=torch.__version__, device=torch.cuda.get_device_name(0), trainable_parameters=512*512,
        backbone_frozen=True, training_crops=len(observations), per_sequence_ids={s: len(v) for s, v in groups.items()},
        steps=p["steps"], seed=p["seed"], checkpoint="last", weight_delta_l2=delta,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started,
        losses=losses, evaluation_data_used=False, production_promoted=False,
        data_terms="https://motchallenge.net/data/MOT15/ - CC BY-NC-SA 3.0 plus source-specific terms; local research only"))
    shutil.copy2(bundle / "LICENSE-MIT.txt", output / "LICENSE-MIT.txt")
    shutil.copy2(bundle / "osnet-source.py", output / "osnet-source.py")
    # Bias-free linear scaling commutes with final L2 normalization: applying
    # the head to raw backbone output exports the same unit feature as training.
    model = torch.nn.Sequential(backbone.cpu(), head.cpu()).eval()
    publish_bundle(model, output, plans[0][1], plans[0][0], "osnet_x1_0", loaded, adaptation)
    print(f"Training/export complete; defaults unchanged: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("base_bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    train(args.repo, args.base_bundle, args.policy, args.output)


if __name__ == "__main__": main()
