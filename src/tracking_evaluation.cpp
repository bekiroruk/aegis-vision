#include "aegisvision/tracking_evaluation.hpp"

#include "aegisvision/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace aegisvision::evaluation {
namespace {
using IdMap = std::map<std::uint64_t, std::size_t>;
using PreviousMatches = std::map<std::uint64_t, std::uint64_t>;

void validate_objects(const std::vector<TrackObject> &objects, IdMap &identities) {
    if (objects.size() > TrackingEvaluationLimits::max_objects_per_side_per_frame) {
        throw std::invalid_argument("Tracking evaluation exceeds the per-frame object limit");
    }
    std::set<std::uint64_t> frame_ids;
    for (const auto &object : objects) {
        if (!frame_ids.insert(object.id).second) {
            throw std::invalid_argument("Tracking IDs must be unique within each side of a frame");
        }
        const auto &box = object.bbox;
        if (!std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) ||
            !std::isfinite(box.y2) || box.x2 <= box.x1 || box.y2 <= box.y1) {
            throw std::invalid_argument(
                "Tracking evaluation boxes must be finite and have positive area");
        }
        identities.try_emplace(object.id, 0);
        if (identities.size() > TrackingEvaluationLimits::max_identities_per_side) {
            throw std::invalid_argument("Tracking evaluation exceeds the unique identity limit");
        }
    }
}

double box_iou(const BoundingBox &a, const BoundingBox &b) {
    // Widen before subtraction/multiplication: finite float coordinates can
    // still overflow float area arithmetic. Neither box is clipped to an image.
    const double width =
        std::max(0.0, std::min(static_cast<double>(a.x2), static_cast<double>(b.x2)) -
                          std::max(static_cast<double>(a.x1), static_cast<double>(b.x1)));
    const double height =
        std::max(0.0, std::min(static_cast<double>(a.y2), static_cast<double>(b.y2)) -
                          std::max(static_cast<double>(a.y1), static_cast<double>(b.y1)));
    const double intersection = width * height;
    const double area_a = (static_cast<double>(a.x2) - a.x1) * (static_cast<double>(a.y2) - a.y1);
    const double area_b = (static_cast<double>(b.x2) - b.x1) * (static_cast<double>(b.y2) - b.y1);
    return std::clamp(intersection / (area_a + area_b - intersection), 0.0, 1.0);
}

void number_identities(IdMap &identities) {
    std::size_t index = 0;
    for (auto &[id, position] : identities) {
        (void)id;
        position = index++;
    }
}
} // namespace

TrackingReport evaluate_tracking(const std::vector<TrackingFrame> &frames, double iou_threshold) {
    if (!std::isfinite(iou_threshold) || iou_threshold <= 0 || iou_threshold > 1) {
        throw std::invalid_argument("Tracking evaluation IoU threshold must be in (0,1]");
    }
    if (frames.size() > TrackingEvaluationLimits::max_frames) {
        throw std::invalid_argument("Tracking evaluation exceeds the frame limit");
    }
    TrackingReport report;
    report.frames = frames.size();
    report.iou_threshold = iou_threshold;
    IdMap gt_ids, prediction_ids;
    std::size_t pair_comparisons = 0;
    // Validate the entire input before allocating matrices or performing the
    // cubic assignment work. All products/additions below have bounded operands.
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto &frame = frames[index];
        if (frame.frame_index != static_cast<int>(index + 1)) {
            throw std::invalid_argument(
                "Tracking evaluation frames must be contiguous and one-based");
        }
        validate_objects(frame.ground_truth, gt_ids);
        validate_objects(frame.predictions, prediction_ids);
        report.ground_truth_detections += frame.ground_truth.size();
        report.predicted_detections += frame.predictions.size();
        if (report.ground_truth_detections + report.predicted_detections >
            TrackingEvaluationLimits::max_total_detections) {
            throw std::invalid_argument("Tracking evaluation exceeds the total detection limit");
        }
        pair_comparisons += frame.ground_truth.size() * frame.predictions.size();
        if (pair_comparisons > TrackingEvaluationLimits::max_frame_pair_comparisons) {
            throw std::invalid_argument("Tracking evaluation exceeds the pair comparison limit");
        }
    }
    report.ground_truth_identities = gt_ids.size();
    report.predicted_identities = prediction_ids.size();
    if (gt_ids.size() * prediction_ids.size() > TrackingEvaluationLimits::max_identity_pair_cells) {
        throw std::invalid_argument("Tracking evaluation exceeds the global identity matrix limit");
    }
    number_identities(gt_ids);
    number_identities(prediction_ids);
    std::vector<std::vector<std::uint64_t>> potential_matches(
        gt_ids.size(), std::vector<std::uint64_t>(prediction_ids.size(), 0));
    PreviousMatches previous_match, previous_timestep_match;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();

    // Matching follows TrackEval CLEAR's implementation. In particular its
    // early empty-side branches preserve previous_timestep_match; IDSW compares
    // the last match of a GT identity even after arbitrarily long gaps.
    for (const auto &frame : frames) {
        if (frame.ground_truth.empty()) {
            report.false_positives += frame.predictions.size();
            continue;
        }
        if (frame.predictions.empty()) {
            report.false_negatives += frame.ground_truth.size();
            continue;
        }
        std::vector<std::vector<double>> similarity(frame.ground_truth.size(),
                                                    std::vector<double>(frame.predictions.size()));
        auto weights = similarity;
        for (std::size_t row = 0; row < frame.ground_truth.size(); ++row) {
            const auto &gt = frame.ground_truth[row];
            const auto previous = previous_timestep_match.find(gt.id);
            for (std::size_t column = 0; column < frame.predictions.size(); ++column) {
                const auto &prediction = frame.predictions[column];
                const double iou = box_iou(gt.bbox, prediction.bbox);
                similarity[row][column] = iou;
                // Identity uses every potential >= threshold pair, not just
                // the frame assignment chosen by CLEAR. Its threshold does
                // not include CLEAR's machine-epsilon tolerance.
                if (iou >= iou_threshold) {
                    ++potential_matches[gt_ids.at(gt.id)][prediction_ids.at(prediction.id)];
                }
                if (iou < iou_threshold - epsilon)
                    continue;
                const double continuity =
                    previous != previous_timestep_match.end() && previous->second == prediction.id
                        ? 1000.0
                        : 0.0;
                // <=500 objects guarantees one retained identity outweighs
                // every possible secondary IoU gain. Division preserves the
                // objective while satisfying assign_max_weight's [0,1] API.
                const double score = continuity + iou;
                if (score > epsilon)
                    weights[row][column] = score / 1001.0;
            }
        }
        const auto assignment = assign_max_weight(weights);
        previous_timestep_match.clear();
        for (const auto &[row, column] : assignment) {
            const auto gt_id = frame.ground_truth[row].id;
            const auto prediction_id = frame.predictions[column].id;
            const auto previous = previous_match.find(gt_id);
            if (previous != previous_match.end() && previous->second != prediction_id) {
                ++report.id_switches;
            }
            previous_match[gt_id] = prediction_id;
            previous_timestep_match[gt_id] = prediction_id;
            report.motp_sum += similarity[row][column];
        }
        report.true_positives += assignment.size();
        report.false_negatives += frame.ground_truth.size() - assignment.size();
        report.false_positives += frame.predictions.size() - assignment.size();
    }

    // Identity's dummy-expanded minimum FN+FP objective has constant total
    // GT+prediction detections minus twice the chosen potential-match count.
    // Therefore this bounded maximum-weight identity assignment is equivalent,
    // including leaving identities unmatched at zero potential reward.
    std::vector<std::vector<double>> identity_weights(
        gt_ids.size(), std::vector<double>(prediction_ids.size(), 0));
    const double scale = static_cast<double>(std::max<std::size_t>(1, frames.size()));
    for (std::size_t row = 0; row < potential_matches.size(); ++row) {
        for (std::size_t column = 0; column < potential_matches[row].size(); ++column) {
            identity_weights[row][column] =
                static_cast<double>(potential_matches[row][column]) / scale;
        }
    }
    for (const auto &[row, column] : assign_max_weight(identity_weights)) {
        report.id_true_positives += potential_matches[row][column];
    }
    report.id_false_negatives = report.ground_truth_detections - report.id_true_positives;
    report.id_false_positives = report.predicted_detections - report.id_true_positives;
    if (report.true_positives > 0) {
        report.motp = report.motp_sum / static_cast<double>(report.true_positives);
    }
    if (report.ground_truth_detections > 0) {
        const double ground_truth = static_cast<double>(report.ground_truth_detections);
        report.recall = static_cast<double>(report.true_positives) / ground_truth;
        report.mota = (static_cast<double>(report.true_positives) -
                       static_cast<double>(report.false_positives) -
                       static_cast<double>(report.id_switches)) /
                      ground_truth;
        report.id_recall = static_cast<double>(report.id_true_positives) / ground_truth;
        report.idf1 = 2.0 * static_cast<double>(report.id_true_positives) /
                      (ground_truth + static_cast<double>(report.predicted_detections));
        if (report.predicted_detections > 0) {
            const double predictions = static_cast<double>(report.predicted_detections);
            report.precision = static_cast<double>(report.true_positives) / predictions;
            report.id_precision = static_cast<double>(report.id_true_positives) / predictions;
        }
    }
    return report;
}

HotaReport evaluate_hota(const std::vector<TrackingFrame>& frames) {
    if (frames.size() > TrackingEvaluationLimits::max_frames)
        throw std::invalid_argument("HOTA exceeds the frame limit");
    IdMap ground_ids, predicted_ids;
    std::size_t total_gt = 0, total_predictions = 0, pair_work = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const auto& frame = frames[i];
        if (frame.frame_index != static_cast<int>(i + 1))
            throw std::invalid_argument("HOTA frames must be contiguous and one-based");
        validate_objects(frame.ground_truth, ground_ids);
        validate_objects(frame.predictions, predicted_ids);
        total_gt += frame.ground_truth.size();
        total_predictions += frame.predictions.size();
        pair_work += frame.ground_truth.size() * frame.predictions.size();
        if (total_gt + total_predictions > TrackingEvaluationLimits::max_total_detections ||
            pair_work > TrackingEvaluationLimits::max_frame_pair_comparisons)
            throw std::invalid_argument("HOTA exceeds the detection/pair-work limit");
    }
    const auto cells = ground_ids.size() * predicted_ids.size();
    if (cells > TrackingEvaluationLimits::max_identity_pair_cells)
        throw std::invalid_argument("HOTA exceeds the identity matrix limit");
    number_identities(ground_ids);
    number_identities(predicted_ids);
    HotaReport result;
    for (std::size_t a = 0; a < result.thresholds.size(); ++a) {
        auto& row = result.thresholds[a];
        row.alpha = .05 + .05 * static_cast<double>(a);
        row.false_negatives = total_gt;
        row.false_positives = total_predictions;
    }
    std::vector<std::size_t> gt_frequency(ground_ids.size()), pred_frequency(predicted_ids.size());
    std::vector<double> alignment(cells);
    const auto cell = [&](std::uint64_t gt, std::uint64_t pred) {
        return ground_ids.at(gt) * predicted_ids.size() + predicted_ids.at(pred);
    };
    const auto similarities = [](const TrackingFrame& frame) {
        std::vector<std::vector<double>> values(frame.ground_truth.size(),
                                               std::vector<double>(frame.predictions.size()));
        for (std::size_t g = 0; g < frame.ground_truth.size(); ++g)
            for (std::size_t p = 0; p < frame.predictions.size(); ++p)
                values[g][p] = box_iou(frame.ground_truth[g].bbox, frame.predictions[p].bbox);
        return values;
    };
    // Soft overlap evidence is normalized within each frame before accumulating
    // sequence-wide identity alignment. No alpha or CLEAR continuity enters it.
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (const auto& frame : frames) {
        for (const auto& gt : frame.ground_truth) ++gt_frequency[ground_ids.at(gt.id)];
        for (const auto& pred : frame.predictions) ++pred_frequency[predicted_ids.at(pred.id)];
        const auto overlaps = similarities(frame);
        std::vector<double> row_sum(frame.ground_truth.size()), col_sum(frame.predictions.size());
        for (std::size_t g = 0; g < row_sum.size(); ++g)
            for (std::size_t p = 0; p < col_sum.size(); ++p) {
                row_sum[g] += overlaps[g][p];
                col_sum[p] += overlaps[g][p];
            }
        for (std::size_t g = 0; g < row_sum.size(); ++g)
            for (std::size_t p = 0; p < col_sum.size(); ++p) {
                const auto denominator = row_sum[g] + col_sum[p] - overlaps[g][p];
                if (denominator > epsilon)
                    alignment[cell(frame.ground_truth[g].id, frame.predictions[p].id)] += overlaps[g][p] / denominator;
            }
    }
    for (std::size_t g = 0; g < gt_frequency.size(); ++g)
        for (std::size_t p = 0; p < pred_frequency.size(); ++p) {
            auto& value = alignment[g * predicted_ids.size() + p];
            value = std::clamp(value / (gt_frequency[g] + pred_frequency[p] - value), 0.0, 1.0);
        }
    // uint32 is safe: a pair can match at most max_frames=10,000 times.
    std::array<std::vector<std::uint32_t>, 19> matched;
    for (auto& counts : matched) counts.resize(cells);
    std::array<double, 19> localization_sum{};
    for (const auto& frame : frames) {
        if (frame.ground_truth.empty() || frame.predictions.empty()) continue;
        const auto overlaps = similarities(frame);
        auto rewards = overlaps;
        for (std::size_t g = 0; g < frame.ground_truth.size(); ++g)
            for (std::size_t p = 0; p < frame.predictions.size(); ++p)
                rewards[g][p] *= alignment[cell(frame.ground_truth[g].id, frame.predictions[p].id)];
        // Zero-reward pairs have zero overlap and cannot pass any positive alpha.
        // Our bounded solver omits those pairs; positive rewards maximize the
        // same frame objective as a full rectangular assignment.
        for (const auto& [g, p] : assign_max_weight(rewards)) {
            const auto index = cell(frame.ground_truth[g].id, frame.predictions[p].id);
            for (std::size_t a = 0; a < result.thresholds.size(); ++a) {
                auto& row = result.thresholds[a];
                if (overlaps[g][p] < row.alpha - epsilon) continue;
                ++row.true_positives;
                --row.false_positives;
                --row.false_negatives;
                ++matched[a][index];
                localization_sum[a] += overlaps[g][p];
            }
        }
    }
    double hota_sum = 0, det_sum = 0, ass_sum = 0, loc_sum = 0;
    for (std::size_t a = 0; a < result.thresholds.size(); ++a) {
        auto& row = result.thresholds[a];
        double association_sum = 0;
        for (std::size_t g = 0; g < gt_frequency.size(); ++g)
            for (std::size_t p = 0; p < pred_frequency.size(); ++p) {
                const double count = matched[a][g * predicted_ids.size() + p];
                association_sum += count * count / std::max(1.0, gt_frequency[g] + pred_frequency[p] - count);
            }
        row.association_accuracy = association_sum / std::max<std::uint64_t>(1, row.true_positives);
        row.detection_accuracy = static_cast<double>(row.true_positives) /
            std::max<std::uint64_t>(1, row.true_positives + row.false_positives + row.false_negatives);
        row.localization_accuracy = row.true_positives ? localization_sum[a] / row.true_positives : 1.0;
        row.hota = std::sqrt(row.detection_accuracy * row.association_accuracy);
        hota_sum += row.hota; det_sum += row.detection_accuracy;
        ass_sum += row.association_accuracy; loc_sum += row.localization_accuracy;
    }
    if (total_gt) {
        result.hota = hota_sum / 19;
        result.detection_accuracy = det_sum / 19;
        result.association_accuracy = ass_sum / 19;
        result.localization_accuracy = loc_sum / 19;
    }
    return result;
}

} // namespace aegisvision::evaluation
