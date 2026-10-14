# ADR 0020 — Bounded offline camera handoff before live global identity

Status: accepted for an experimental offline component, 2026-10-14.

Existing OSNet appearance is used within one video. Camera-local numeric IDs,
independent arrival clocks and appearance similarity cannot safely be treated as
a global identity namespace. No labeled synchronized multi-camera validation
has been completed in this project.

We introduce a pure C++ function accepting closed tracklets, explicit camera
namespaces/epochs, one declared embedding space and directed travel windows.
Only strictly time-separated, mutual-best pairs with margins in both directions
are linked. No second-choice fallback or prototype adaptation is performed.
Bounds are 512 tracklets / 16 cameras; IDs are deterministic for a fixed batch.
The CLI requires explicit synchronized-time provenance, but does not verify
clocks. Rejected hypotheses get separate IDs, not assertions of different people.

This is deliberately not an online gallery, full MTMC pipeline, overlapping-view
fusion, identity registry, maximum-weight assignment or accuracy claim. Real
tracklet extraction, clock alignment and held-out multi-camera evaluation are
required before adoption. Default thresholds remain uncalibrated. Current
single-camera tracking, service endpoints and persisted jobs are unchanged.
