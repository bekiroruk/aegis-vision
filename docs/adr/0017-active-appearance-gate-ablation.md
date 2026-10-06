# ADR 0017: Isolate the active appearance hard gate

Date: 2026-10-06. Status: accepted for experimental evaluation only.

## Context

Candidate traces identified appearance rejection in 14 of 15 consecutive-frame
OSNet identity breaks across two already-observed development sequences. A
counterfactual experiment is needed before treating that observation as a fix.

## Decision

Add a default-off core option removing the cosine hard gate only from active
high/low association. Preserve lost-high appearance gating, geometry, fusion,
EMA and lifecycle. Clamp nonpositive fused reward to zero (unmatched), without
rescaling cosine. Handle exact antipodal EMA cancellation with the observation
direction; this is possible in the generic API but not with experiment momentum .90.

Expose only a quality CLI ablation flag. Run a sixth independent tracker with
the same detector outputs and already-computed embeddings; keep five historical
results unchanged. Extend official-reference and offline-audit allowlists.
Reject the ablation on frozen transfer manifests. Do not expose it as a normal
video/live configuration or change defaults.

## Consequences

Development identity continuity improves, but false positives increase and HOTA
changes have mixed signs. Retain experimental status. Further selection requires
separate validation and untouched test data, not reinterpretation of observed
development scenes. See [protocol and results](../active-appearance-ablation.md).
