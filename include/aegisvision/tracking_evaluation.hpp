#pragma once

#include "aegisvision/domain.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace aegisvision::evaluation {

struct TrackObject {
    std::uint64_t id{};
    BoundingBox bbox{0, 0, 0, 0};
};

// Every decoded frame is represented, including frames with no objects. IDs
// are unique within each side of a frame, but may recur in later frames.
struct TrackingFrame {
    int frame_index{}; // Contiguous, one-based: 1, 2, ..., N.
    std::vector<TrackObject> ground_truth;
    std::vector<TrackObject> predictions;
};

struct TrackingEvaluationLimits {
    static constexpr std::size_t max_frames = 10'000;
    static constexpr std::size_t max_objects_per_side_per_frame = 500;
    static constexpr std::size_t max_total_detections = 1'000'000;
    static constexpr std::size_t max_identities_per_side = 1'000;
    static constexpr std::size_t max_identity_pair_cells = 1'000'000;
    static constexpr std::size_t max_frame_pair_comparisons = 25'000'000;
};

struct TrackingReport {
    std::size_t frames{};
    double iou_threshold{};
    std::uint64_t ground_truth_detections{};
    std::uint64_t predicted_detections{};
    std::size_t ground_truth_identities{};
    std::size_t predicted_identities{};

    // CLEAR counts use continuity-prioritized, one-to-one frame assignment.
    std::uint64_t true_positives{};
    std::uint64_t false_positives{};
    std::uint64_t false_negatives{};
    std::uint64_t id_switches{};
    double motp_sum{}; // Sum of IoU for CLEAR's selected matches, not distances.

    // Identity counts use a separate, sequence-wide identity assignment.
    std::uint64_t id_true_positives{};
    std::uint64_t id_false_positives{};
    std::uint64_t id_false_negatives{};

    // Fractions (not percentages). Scores without ground truth are undefined;
    // precision is also undefined without predictions, MOTP without any TP.
    std::optional<double> precision;
    std::optional<double> recall;
    std::optional<double> mota;
    std::optional<double> motp;
    std::optional<double> id_precision;
    std::optional<double> id_recall;
    std::optional<double> idf1;
};

// Strict, bounded, single-class/single-sequence evaluator. Bounding boxes must
// have finite coordinates and strictly positive width/height; negative image
// coordinates are allowed. Threshold must be finite and in (0, 1].
// This is CLEAR + Identity evaluation, not HOTA or the complete MOTChallenge
// protocol (dataset-specific ignore-region/class filtering belongs upstream).
[[nodiscard]] TrackingReport evaluate_tracking(const std::vector<TrackingFrame> &frames,
                                               double iou_threshold = 0.5);

struct HotaThreshold {
    double alpha{};
    std::uint64_t true_positives{}, false_positives{}, false_negatives{};
    double hota{}, detection_accuracy{}, association_accuracy{}, localization_accuracy{1.0};
};
struct HotaReport {
    std::array<HotaThreshold, 19> thresholds{};
    // Arithmetic means over alpha=.05:.05:.95; undefined without GT.
    std::optional<double> hota, detection_accuracy, association_accuracy, localization_accuracy;
};
// Independent sequence-wide alignment + per-frame assignment, then threshold
// filtering. Does not reuse CLEAR matches or run a new assignment per alpha.
// Same bounded inputs as evaluate_tracking; <=19 million association counters.
[[nodiscard]] HotaReport evaluate_hota(const std::vector<TrackingFrame>& frames);

} // namespace aegisvision::evaluation
