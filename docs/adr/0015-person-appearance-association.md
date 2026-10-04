# ADR 0015: Bounded OSNet person appearance for local Kalman association

Date: 2026-10-02. Status: accepted as opt-in experimental adapter.

## Context

Center-only motion gating repaired a Campus size-jitter failure, but also retained
more partial/edge detections in Stadtmitte. Geometric overlap and motion alone do
not represent a person's visual appearance. CLIP is a multimodal search encoder,
not a dedicated person Re-ID substitute. Existing defaults and historical results
must remain reproducible; no quality improvement is assumed from adding a model.

## Decision

Use the author's pinned OSNet x0.25 MSMT17-combineall weights, exported to static
FP32 ONNX. C++ OpenCV DNN extracts L2-normalized 512-component features from
eligible person crops. Python prepares the model and references only. Bundle
size/hash/schema/source, preprocessing and real model outputs are validated.

Append optional appearance settings to the core tracker, disabled by default.
Retain class, IoU and motion hard gates; add cosine-distance .20 and equal-weight
IoU/cosine assignment reward. Keep one normalized EMA prototype per bounded
identity, updated only on high-score matches with old weight .90. Preserve the
active-high / active-low / lost-high lifecycle and observed-only output. Invalid
eligible features reject the transaction before state aging.

Expose config-only `kalman-reid` in local video with required `[appearance]`
bundle. Inject the encoder through the existing pipeline before tracking and
disable indexing. OpenCV-only builds report unsupported appearance explicitly.
No live/session/web changes, global identity, gallery or cross-camera handoff.

The quality runner shares one detector result with five independent trackers.
Only the appearance path receives a copy with embeddings; the four historical
paths and raw predictions are untouched. Export normal CLEAR/Identity predictions
for official TrackEval verification; report embedding cost independently. Freeze
the defaults before running both already-inspected development scenes. Bad or
mixed results are published, not hidden by tuning on those same scenes.

## Consequences

There is additional CPU crop inference and possible false appearance rejection
under occlusion or tiny crops. This custom backend is not full DeepSORT. Golden
PyTorch/OpenCV/C++ agreement verifies implementation, not generalization. Re-ID
model/code licensing does not grant upstream data rights. Artifacts stay local.

Next: unseen validation/test sequences, HOTA, calibration/ablation and only then
time-aware streaming or multi-camera work. See [protocol and sources](../person-appearance.md).
