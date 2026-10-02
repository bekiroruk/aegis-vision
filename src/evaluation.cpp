#include "aegisvision/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aegisvision::evaluation {
namespace {

struct PreparedFrame {
    std::vector<const GroundTruth *> ground_truth;
    std::vector<const Detection *> predictions;
};

enum class Outcome { true_positive, false_positive, ignored };

struct ScoredOutcome {
    float score;
    Outcome outcome;
};

void validate_text(const std::string &value, const std::size_t limit,
                   const std::string_view field) {
    if (value.empty() || value.size() > limit || value.find('\0') != std::string::npos) {
        throw std::invalid_argument(std::string(field) + " must be non-empty, bounded text");
    }
}

void validate_box(const BoundingBox &box) {
    if (!std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) ||
        !std::isfinite(box.y2) || box.x2 <= box.x1 || box.y2 <= box.y1) {
        throw std::invalid_argument("Evaluation boxes must be finite with positive area");
    }
}

double area(const BoundingBox &box) {
    // Convert before subtracting/multiplying: finite float coordinates can
    // overflow float area, but their differences/products fit in double.
    return (static_cast<double>(box.x2) - box.x1) * (static_cast<double>(box.y2) - box.y1);
}

double overlap(const BoundingBox &detection, const GroundTruth &truth) {
    const double width = std::max(0.0, std::min<double>(detection.x2, truth.bbox.x2) -
                                           std::max<double>(detection.x1, truth.bbox.x1));
    const double height = std::max(0.0, std::min<double>(detection.y2, truth.bbox.y2) -
                                            std::max<double>(detection.y1, truth.bbox.y1));
    const double intersection = width * height;
    const double denominator =
        truth.crowd ? area(detection) : area(detection) + area(truth.bbox) - intersection;
    return intersection / denominator;
}

std::vector<ScoredOutcome> match(const std::vector<PreparedFrame> &frames, const double threshold) {
    std::vector<ScoredOutcome> outcomes;
    for (const auto &frame : frames) {
        std::vector<bool> matched(frame.ground_truth.size(), false);
        for (const auto *detection : frame.predictions) {
            std::size_t best = frame.ground_truth.size();
            double best_overlap = std::min(threshold, 1.0 - 1e-10);
            for (std::size_t i = 0; i < frame.ground_truth.size(); ++i) {
                const auto &truth = *frame.ground_truth[i];
                if (matched[i] && !truth.crowd) {
                    continue;
                }
                // Normal annotations sort before crowds. A matching normal
                // annotation takes precedence even over higher crowd overlap.
                if (best != frame.ground_truth.size() && !frame.ground_truth[best]->crowd &&
                    truth.crowd) {
                    break;
                }
                const double candidate_overlap = overlap(detection->bbox, truth);
                if (candidate_overlap < best_overlap) {
                    continue;
                }
                best_overlap = candidate_overlap;
                best = i;
            }
            Outcome outcome = Outcome::false_positive;
            if (best != frame.ground_truth.size()) {
                matched[best] = true;
                outcome =
                    frame.ground_truth[best]->crowd ? Outcome::ignored : Outcome::true_positive;
            }
            outcomes.push_back({detection->score, outcome});
        }
    }
    std::stable_sort(outcomes.begin(), outcomes.end(),
                     [](const auto &left, const auto &right) { return left.score > right.score; });
    return outcomes;
}

double interpolated_ap(const std::vector<ScoredOutcome> &outcomes,
                       const std::size_t ground_truth_count) {
    std::vector<double> recalls;
    std::vector<double> precisions;
    recalls.reserve(outcomes.size());
    precisions.reserve(outcomes.size());
    std::size_t tp = 0;
    std::size_t fp = 0;
    for (const auto &result : outcomes) {
        if (result.outcome == Outcome::ignored) {
            continue;
        }
        tp += result.outcome == Outcome::true_positive ? 1 : 0;
        fp += result.outcome == Outcome::false_positive ? 1 : 0;
        recalls.push_back(static_cast<double>(tp) / static_cast<double>(ground_truth_count));
        precisions.push_back(static_cast<double>(tp) / static_cast<double>(tp + fp));
    }
    // Monotone precision envelope, followed by the COCO 101-point recall grid.
    for (std::size_t i = precisions.size(); i > 1; --i) {
        precisions[i - 2] = std::max(precisions[i - 2], precisions[i - 1]);
    }
    double sum = 0;
    for (int i = 0; i <= 100; ++i) {
        // Match COCOeval's NumPy linspace grid. Multiplication and division
        // round differently at e.g. .70; searchsorted/lower_bound observes it.
        const double recall_threshold = static_cast<double>(i) * 0.01;
        const auto found = std::lower_bound(recalls.begin(), recalls.end(), recall_threshold);
        if (found != recalls.end()) {
            sum += precisions[static_cast<std::size_t>(found - recalls.begin())];
        }
    }
    return sum / 101.0;
}

void finalize_operating_point(OperatingPoint &point) {
    const auto predicted = point.true_positives + point.false_positives;
    const auto truth = point.true_positives + point.false_negatives;
    if (predicted != 0) {
        point.precision =
            static_cast<double>(point.true_positives) / static_cast<double>(predicted);
    }
    if (truth != 0) {
        point.recall = static_cast<double>(point.true_positives) / static_cast<double>(truth);
    }
}

OperatingPoint operating_point(const std::vector<ScoredOutcome> &outcomes,
                               const std::size_t ground_truth_count, const double score_threshold) {
    OperatingPoint point;
    for (const auto &result : outcomes) {
        if (static_cast<double>(result.score) < score_threshold) {
            continue;
        }
        switch (result.outcome) {
        case Outcome::true_positive:
            ++point.true_positives;
            break;
        case Outcome::false_positive:
            ++point.false_positives;
            break;
        case Outcome::ignored:
            ++point.ignored_predictions;
            break;
        }
    }
    point.false_negatives = ground_truth_count - point.true_positives;
    finalize_operating_point(point);
    return point;
}

} // namespace

DetectionReport evaluate_detection(const std::vector<Frame> &frames,
                                   const std::vector<std::string> &class_labels,
                                   const EvaluationConfig &config) {
    if (frames.size() > 10'000) {
        throw std::invalid_argument("Detection evaluation exceeds the frame limit");
    }
    if (class_labels.empty() || class_labels.size() > 1024) {
        throw std::invalid_argument("Evaluation requires 1 to 1024 distinct classes");
    }
    if (config.max_detections_per_image_per_class == 0 ||
        config.max_detections_per_image_per_class > 100000 ||
        !std::isfinite(config.operating_score_threshold) || config.operating_score_threshold < 0 ||
        config.operating_score_threshold > 1 || !std::isfinite(config.operating_iou_threshold) ||
        config.operating_iou_threshold <= 0 || config.operating_iou_threshold > 1) {
        throw std::invalid_argument("Invalid detection evaluation configuration");
    }
    std::unordered_map<std::string, std::size_t> class_indices;
    DetectionReport report;
    report.image_count = frames.size();
    for (std::size_t i = 0; i < report.iou_thresholds.size(); ++i) {
        report.iou_thresholds[i] =
            i == 9 ? 0.95 : 0.50 + static_cast<double>(i) * ((0.95 - 0.50) / 9.0);
    }
    for (const auto &label : class_labels) {
        validate_text(label, 256, "Class label");
        if (!class_indices.emplace(label, report.classes.size()).second) {
            throw std::invalid_argument("Evaluation class labels must be unique");
        }
        ClassReport class_report;
        class_report.label = label;
        report.classes.push_back(std::move(class_report));
    }
    // Only occupied image/category pairs are allocated. A sparse frame with
    // many selected classes must not create frames*classes empty vectors.
    std::vector<std::vector<PreparedFrame>> prepared(class_labels.size());
    std::unordered_set<std::string> frame_ids;
    std::size_t total_objects = 0;
    std::size_t pair_comparisons = 0;
    for (const auto &frame : frames) {
        if (frame.ground_truth.size() > 10'000 || frame.predictions.size() > 10'000) {
            throw std::invalid_argument("Detection evaluation exceeds the per-frame object limit");
        }
        total_objects += frame.ground_truth.size() + frame.predictions.size();
        if (total_objects > 1'000'000) {
            throw std::invalid_argument("Detection evaluation exceeds the total object limit");
        }
        validate_text(frame.frame_id, 1024, "Frame ID");
        if (!frame_ids.insert(frame.frame_id).second) {
            throw std::invalid_argument("Evaluation frame IDs must be unique");
        }
        std::unordered_map<std::size_t, PreparedFrame> categories;
        for (const auto &truth : frame.ground_truth) {
            validate_box(truth.bbox);
            const auto found = class_indices.find(truth.label);
            if (found == class_indices.end()) {
                throw std::invalid_argument("Ground-truth label is outside evaluation classes");
            }
            const auto index = found->second;
            categories[index].ground_truth.push_back(&truth);
            if (truth.crowd) {
                ++report.classes[index].crowd_ground_truth_count;
                ++report.crowd_ground_truth_count;
            } else {
                ++report.classes[index].ground_truth_count;
                ++report.ground_truth_count;
            }
        }
        for (const auto &detection : frame.predictions) {
            validate_box(detection.bbox);
            if (!std::isfinite(detection.score) || detection.score < 0 || detection.score > 1) {
                throw std::invalid_argument("Prediction scores must be finite probabilities");
            }
            const auto found = class_indices.find(detection.label);
            if (found == class_indices.end()) {
                throw std::invalid_argument("Prediction label is outside evaluation classes");
            }
            const auto index = found->second;
            categories[index].predictions.push_back(&detection);
            ++report.classes[index].prediction_count;
            ++report.prediction_count;
        }
        for (auto &[index, category] : categories) {
            std::stable_sort(
                category.ground_truth.begin(), category.ground_truth.end(),
                [](const auto *left, const auto *right) { return !left->crowd && right->crowd; });
            std::stable_sort(
                category.predictions.begin(), category.predictions.end(),
                [](const auto *left, const auto *right) { return left->score > right->score; });
            if (category.predictions.size() > config.max_detections_per_image_per_class) {
                category.predictions.resize(config.max_detections_per_image_per_class);
            }
            pair_comparisons += category.predictions.size() * category.ground_truth.size();
            if (pair_comparisons > 25'000'000) {
                throw std::invalid_argument("Detection evaluation exceeds the matching work limit");
            }
            report.classes[index].evaluated_prediction_count += category.predictions.size();
            report.evaluated_prediction_count += category.predictions.size();
            prepared[index].push_back(std::move(category));
        }
    }
    double macro_ap50 = 0;
    double macro_ap50_95 = 0;
    std::size_t classes_with_truth = 0;
    for (std::size_t c = 0; c < report.classes.size(); ++c) {
        auto &category = report.classes[c];
        auto op_outcomes = match(prepared[c], config.operating_iou_threshold);
        category.operating_point = operating_point(op_outcomes, category.ground_truth_count,
                                                   config.operating_score_threshold);
        report.operating_point.true_positives += category.operating_point.true_positives;
        report.operating_point.false_positives += category.operating_point.false_positives;
        report.operating_point.false_negatives += category.operating_point.false_negatives;
        report.operating_point.ignored_predictions += category.operating_point.ignored_predictions;
        if (category.ground_truth_count == 0) {
            continue;
        }
        double sum = 0;
        for (std::size_t t = 0; t < report.iou_thresholds.size(); ++t) {
            auto outcomes = report.iou_thresholds[t] == config.operating_iou_threshold
                                ? op_outcomes
                                : match(prepared[c], report.iou_thresholds[t]);
            const double ap = interpolated_ap(outcomes, category.ground_truth_count);
            category.ap_by_iou[t] = ap;
            sum += ap;
        }
        category.ap50 = category.ap_by_iou.front();
        category.ap50_95 = sum / static_cast<double>(report.iou_thresholds.size());
        macro_ap50 += *category.ap50;
        macro_ap50_95 += *category.ap50_95;
        ++classes_with_truth;
    }
    if (classes_with_truth != 0) {
        report.ap50 = macro_ap50 / static_cast<double>(classes_with_truth);
        report.ap50_95 = macro_ap50_95 / static_cast<double>(classes_with_truth);
    }
    finalize_operating_point(report.operating_point);
    return report;
}

std::optional<LatencySummary> summarize_latencies(const std::vector<double> &milliseconds) {
    if (milliseconds.empty()) {
        return std::nullopt;
    }
    std::vector<double> sorted = milliseconds;
    long double sum = 0;
    for (const double sample : sorted) {
        if (!std::isfinite(sample) || sample < 0) {
            throw std::invalid_argument("Latencies must be finite, non-negative milliseconds");
        }
        sum += sample;
    }
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](const double p) {
        const auto rank =
            static_cast<std::size_t>(std::ceil(p * static_cast<double>(sorted.size())));
        return sorted[std::min(sorted.size(), std::max<std::size_t>(1, rank)) - 1];
    };
    return LatencySummary{sorted.size(),    static_cast<double>(sum / sorted.size()),
                          sorted.front(),   percentile(0.50),
                          percentile(0.95), sorted.back()};
}

} // namespace aegisvision::evaluation
