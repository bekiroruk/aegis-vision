#pragma once

#include "aegisvision/domain.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace aegisvision::evaluation {

struct GroundTruth {
    BoundingBox bbox;
    std::string label;
    bool crowd = false;
};

struct Frame {
    std::string frame_id;
    std::vector<GroundTruth> ground_truth;
    std::vector<Detection> predictions;
};

struct EvaluationConfig {
    // As in category-aware COCOeval, this cap applies to EACH image/category,
    // after stable descending-score sorting, not to the whole image.
    std::size_t max_detections_per_image_per_class = 100;
    double operating_score_threshold = 0.25;
    double operating_iou_threshold = 0.50;
};

struct OperatingPoint {
    std::size_t true_positives{};
    std::size_t false_positives{};
    std::size_t false_negatives{};
    std::size_t ignored_predictions{};
    // Undefined denominators are null, never a fabricated zero accuracy.
    std::optional<double> precision;
    std::optional<double> recall;
};

struct ClassReport {
    std::string label;
    // Only non-crowd annotations contribute to recall and AP denominators.
    std::size_t ground_truth_count{};
    std::size_t crowd_ground_truth_count{};
    std::size_t prediction_count{};
    std::size_t evaluated_prediction_count{};
    std::array<std::optional<double>, 10> ap_by_iou{};
    std::optional<double> ap50;
    std::optional<double> ap50_95;
    OperatingPoint operating_point;
};

struct DetectionReport {
    std::size_t image_count{};
    std::size_t ground_truth_count{};
    std::size_t crowd_ground_truth_count{};
    std::size_t prediction_count{};
    std::size_t evaluated_prediction_count{};
    std::array<double, 10> iou_thresholds{};
    std::optional<double> ap50;
    std::optional<double> ap50_95;
    OperatingPoint operating_point;
    std::vector<ClassReport> classes;
};

// Category-aware, all-area bounding-box AP: IoU .50:.05:.95, 101 recall
// samples, stable score ties in supplied frame/prediction order, greedy
// one-to-one normal-GT matching, reusable crowd matches ignored using
// intersection/detection-area. AP is macro-averaged only over classes with
// non-crowd GT. This is not the full official COCO evaluation suite.
//
// class_labels explicitly defines the evaluated taxonomy. Unknown labels,
// duplicate/empty IDs or labels, non-finite/degenerate boxes, invalid scores,
// and invalid configuration throw std::invalid_argument. Callers evaluating
// a category subset must filter BOTH annotations and predictions first.
// Work limits: 10,000 frames, 10,000 objects per side/frame, one million total
// objects, 25 million retained prediction/GT comparisons per IoU pass.
[[nodiscard]] DetectionReport evaluate_detection(const std::vector<Frame> &frames,
                                                 const std::vector<std::string> &class_labels,
                                                 const EvaluationConfig &config = {});

struct LatencySummary {
    std::size_t sample_count{};
    double mean_ms{};
    double min_ms{};
    double p50_ms{};
    double p95_ms{};
    double max_ms{};
};

// Non-negative finite milliseconds; empty input is null. Percentiles use
// nearest rank: sorted[ceil(p * n) - 1], without interpolation.
[[nodiscard]] std::optional<LatencySummary>
summarize_latencies(const std::vector<double> &milliseconds);

} // namespace aegisvision::evaluation
