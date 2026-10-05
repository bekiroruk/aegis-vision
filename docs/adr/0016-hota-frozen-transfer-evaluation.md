# ADR 0016: Native HOTA and frozen sequence-transfer evaluation

Date: 2026-10-05. Status: accepted.

## Context

The two development sequences gave mixed appearance-association results. CLEAR
and Identity alone do not describe detection/association balance across overlap
thresholds. Repeatedly tuning against those scenes is not independent evaluation.

## Decision

Implement bounded offline HOTA in the dependency-free C++ evaluation library.
Use sequence-global soft identity alignment, one assignment per frame, then 19
overlap thresholds. Keep CLEAR/Identity unchanged. Verify all counts and scores
against the pinned official TrackEval implementation; preserve legacy reports.
Document algorithm attribution and MIT terms in THIRD_PARTY_NOTICES.md.

Freeze the existing model hashes, detector config, five trackers and association
defaults before inference on complete ETH-Sunnyday and PETS09-S2L1 sequences.
The data preparer embeds the protocol and its hash. The runner checks selected
trackers, parameter values, model/config hashes and sequence/split before work.
Reject KITTI-17 before inference because preview/annotation geometry cannot be
verified. Do not infer a crop or resize transform from dimensions alone.

## Consequences

Metrics see GT and the whole sequence; online trackers never do. HOTA computation
is outside measured processing latency. Work and allocation limits remain
explicit, and per-frame overlap matrices are not retained for the whole video.

These are project-unseen public training sequences, not the official hidden test
or proof of no pretrained-data overlap. After reporting, they are observed data;
future tuning needs a fresh untouched test. Results cannot silently promote the
experimental appearance tracker or change production defaults. Independent
sequence metrics are reported separately, not averaged as a pooled benchmark.

See [protocol and reproduction](../tracking-transfer.md).
