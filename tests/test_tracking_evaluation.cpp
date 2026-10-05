#include "aegisvision/tracking_evaluation.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using aegisvision::BoundingBox;
using namespace aegisvision::evaluation;

void require(const bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
void near(const std::optional<double> &actual, const double expected, const char *message) {
    require(actual.has_value() && std::abs(*actual - expected) < 1e-10, message);
}
template <class F> void rejects(F f, const char *message) {
    try {
        f();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(message);
}
TrackObject object(const std::uint64_t id, const float x = 0) { return {id, {x, 0, x + 10, 10}}; }
std::vector<TrackObject> identities(const std::size_t count, const std::uint64_t first_id = 0) {
    std::vector<TrackObject> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
        result.push_back(object(first_id + i));
    return result;
}

void perfect_and_counting() {
    const auto perfect = evaluate_tracking(
        {{1, {object(0, -10)}, {object(0, -10)}}, {2, {object(0, -8)}, {object(0, -8)}}});
    require(perfect.frames == 2 && perfect.ground_truth_detections == 2 &&
                perfect.predicted_detections == 2 && perfect.ground_truth_identities == 1 &&
                perfect.predicted_identities == 1 && perfect.true_positives == 2 &&
                perfect.false_positives == 0 && perfect.false_negatives == 0 &&
                perfect.id_switches == 0 && perfect.id_true_positives == 2 &&
                perfect.id_false_positives == 0 && perfect.id_false_negatives == 0,
            "Perfect tracking counts/zero identity support differ");
    near(perfect.precision, 1, "Perfect precision differs");
    near(perfect.recall, 1, "Perfect recall differs");
    near(perfect.mota, 1, "Perfect MOTA differs");
    near(perfect.motp, 1, "Perfect MOTP differs");
    near(perfect.id_precision, 1, "Perfect ID precision differs");
    near(perfect.id_recall, 1, "Perfect ID recall differs");
    near(perfect.idf1, 1, "Perfect IDF1 differs");

    const auto missed =
        evaluate_tracking({{1, {object(1), object(2, 20)}, {object(10), object(20, 100)}},
                           {2, {object(1), object(2, 20)}, {object(10)}}});
    require(missed.true_positives == 2 && missed.false_positives == 1 &&
                missed.false_negatives == 2 && missed.id_switches == 0 &&
                missed.id_true_positives == 2 && missed.id_false_positives == 1 &&
                missed.id_false_negatives == 2,
            "Miss/false-positive counts differ");
    near(missed.precision, 2.0 / 3.0, "Miss/false-positive precision differs");
    near(missed.recall, 0.5, "Miss/false-positive recall differs");
    near(missed.mota, 0.25, "MOTA denominator/formula differs");
    near(missed.idf1, 4.0 / 7.0, "IDF1 denominator/formula differs");

    const auto split = evaluate_tracking({{1, {object(1)}, {object(10)}},
                                          {2, {object(1)}, {object(20)}},
                                          {3, {object(1)}, {object(20)}}});
    require(split.true_positives == 3 && split.id_switches == 1 && split.id_true_positives == 2 &&
                split.id_false_positives == 1 && split.id_false_negatives == 1,
            "Split identity accounting differs");
    near(split.mota, 2.0 / 3.0, "ID switch did not reduce MOTA");
    near(split.idf1, 2.0 / 3.0, "Global identity assignment did not select the longer track");

    const float extent = std::numeric_limits<float>::max();
    const TrackObject large{std::numeric_limits<std::uint64_t>::max(),
                            {-extent, -extent, extent, extent}};
    near(evaluate_tracking({{1, {large}, {large}}}).motp, 1,
         "Finite large boxes overflowed or high IDs were rejected");
}

void continuity_and_global_identity() {
    const TrackingFrame initial{1, {object(1), object(2, 4)}, {object(10), object(20, 4)}};
    const TrackingFrame swapped{2, {object(1), object(2, 4)}, {object(10, 4), object(20)}};
    const auto retained = evaluate_tracking({initial, swapped}, 0.4);
    require(retained.true_positives == 4 && retained.id_switches == 0,
            "Higher IoU displaced a still-valid previous identity");
    near(retained.motp, 5.0 / 7.0, "MOTP did not use continuity-selected box IoUs");
    near(retained.idf1, 1, "Identity matching omitted qualifying non-CLEAR edges");
    const auto forced_switch = evaluate_tracking({initial, swapped}, 0.5);
    require(forced_switch.true_positives == 4 && forced_switch.id_switches == 2,
            "Below-threshold continuity edges were retained");
    near(forced_switch.motp, 1, "Forced-switch MOTP differs");
    near(forced_switch.mota, 0.5, "Two ID switches were not included in MOTA");
    near(forced_switch.idf1, 0.5, "Whole-sequence identity reassignment differs");

    // CLEAR deliberately retains GT1->10 at frame two, preventing two matches.
    // Identity instead counts every qualifying edge and can assign GT2->10 and
    // GT1->20 globally. Its IDTP need not equal CLEAR's selected TP count.
    const auto independent =
        evaluate_tracking({{1, {object(1), object(2, 5)}, {object(10), object(20, 100)}},
                           {2, {object(1), object(2, 5)}, {object(10, 5), object(20, -4)}}},
                          0.3);
    require(independent.true_positives == 2 && independent.false_positives == 2 &&
                independent.false_negatives == 2 && independent.id_switches == 0,
            "CLEAR continuity-prioritized assignment differs");
    require(independent.id_true_positives == 3 && independent.id_false_positives == 1 &&
                independent.id_false_negatives == 1,
            "Identity reused CLEAR assignments rather than all potential pairs");
    near(independent.idf1, 0.75, "Independent identity score differs");

    const TrackObject half{10, {0, 0, 20, 10}};
    const auto inclusive = evaluate_tracking({{1, {object(1)}, {half}}}, 0.5);
    require(inclusive.true_positives == 1 && inclusive.id_true_positives == 1,
            "IoU equal to threshold must match in both metrics");
    const auto above = evaluate_tracking({{1, {object(1)}, {half}}}, 0.5001);
    require(above.true_positives == 0 && above.id_true_positives == 0 &&
                above.false_positives == 1 && above.false_negatives == 1 && !above.motp,
            "Below-threshold pair matched or fabricated MOTP");
}

void gaps_and_switch_history() {
    const TrackingFrame initial{1, {object(1), object(2, 4)}, {object(10), object(20, 4)}};
    const TrackingFrame after_gap{3, {object(1), object(2, 4)}, {object(10, 4), object(20)}};
    const auto empty_gap = evaluate_tracking({initial, {2, {}, {}}, after_gap}, 0.4);
    require(empty_gap.frames == 3 && empty_gap.true_positives == 4 && empty_gap.id_switches == 0,
            "Empty-side early return lost the reference continuity history");
    const auto prediction_gap =
        evaluate_tracking({initial, {2, {object(1), object(2, 4)}, {}}, after_gap}, 0.4);
    require(prediction_gap.true_positives == 4 && prediction_gap.false_negatives == 2 &&
                prediction_gap.id_switches == 0,
            "Prediction-empty gap continuity differs");
    const auto truth_gap = evaluate_tracking({initial, {2, {}, {object(99, 100)}}, after_gap}, 0.4);
    require(truth_gap.true_positives == 4 && truth_gap.false_positives == 1 &&
                truth_gap.id_switches == 0,
            "GT-empty gap continuity differs");

    const auto unmatched_gap = evaluate_tracking(
        {initial, {2, {object(1), object(2, 4)}, {object(99, 100)}}, after_gap}, 0.4);
    require(unmatched_gap.true_positives == 4 && unmatched_gap.false_negatives == 2 &&
                unmatched_gap.false_positives == 1 && unmatched_gap.id_switches == 2,
            "Nonempty unmatched timestep failed to clear continuity or preserve switch history");
    const auto long_gap = evaluate_tracking({{1, {object(1)}, {object(10)}},
                                             {2, {}, {}},
                                             {3, {object(1)}, {}},
                                             {4, {}, {}},
                                             {5, {object(1)}, {object(20)}}});
    require(long_gap.true_positives == 2 && long_gap.false_negatives == 1 &&
                long_gap.id_switches == 1,
            "Last-match ID switch history was reset across a gap");
}

void undefined_scores() {
    const auto empty = evaluate_tracking({});
    require(empty.frames == 0 && !empty.precision && !empty.recall && !empty.mota && !empty.motp &&
                !empty.id_precision && !empty.id_recall && !empty.idf1,
            "Empty sequence fabricated quality scores");
    const auto no_truth = evaluate_tracking({{1, {}, {object(1)}}, {2, {}, {}}});
    require(no_truth.false_positives == 1 && no_truth.id_false_positives == 1 &&
                !no_truth.precision && !no_truth.recall && !no_truth.mota && !no_truth.motp &&
                !no_truth.id_precision && !no_truth.id_recall && !no_truth.idf1,
            "No-GT sequence fabricated quality scores");
    const auto no_predictions = evaluate_tracking({{1, {object(1)}, {}}});
    require(no_predictions.false_negatives == 1 && no_predictions.id_false_negatives == 1 &&
                !no_predictions.precision && !no_predictions.id_precision && !no_predictions.motp,
            "Zero-denominator prediction/MOTP scores must be undefined");
    near(no_predictions.recall, 0, "Missing predictions must have zero recall");
    near(no_predictions.mota, 0, "Missing predictions must have zero MOTA");
    near(no_predictions.id_recall, 0, "Missing predictions must have zero ID recall");
    near(no_predictions.idf1, 0, "Missing predictions must have zero IDF1");
    const auto negative =
        evaluate_tracking({{1, {object(1)}, {object(10, 100), object(20, 200), object(30, 300)}}});
    near(negative.mota, -3, "MOTA was incorrectly clamped to zero");
}

void validation_and_limits() {
    for (const double threshold : {0.0, -0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()})
        rejects([threshold] { (void)evaluate_tracking({}, threshold); },
                "Invalid IoU threshold accepted");
    rejects([] { (void)evaluate_tracking({{0, {}, {}}}); }, "Zero-based frame accepted");
    rejects([] { (void)evaluate_tracking({{2, {}, {}}}); }, "Non-one starting frame accepted");
    rejects([] { (void)evaluate_tracking({{1, {}, {}}, {3, {}, {}}}); }, "Missing frame accepted");
    rejects([] { (void)evaluate_tracking({{1, {}, {}}, {1, {}, {}}}); }, "Repeated frame accepted");
    rejects([] { (void)evaluate_tracking({{1, {object(1), object(1)}, {}}}); },
            "Duplicate GT ID accepted");
    rejects([] { (void)evaluate_tracking({{1, {}, {object(1), object(1)}}}); },
            "Duplicate prediction ID accepted");
    for (const auto box : {BoundingBox{0, 0, 0, 10}, BoundingBox{0, 0, 10, 0},
                           BoundingBox{std::numeric_limits<float>::quiet_NaN(), 0, 10, 10},
                           BoundingBox{0, 0, std::numeric_limits<float>::infinity(), 10}}) {
        rejects([box] { (void)evaluate_tracking({{1, {{1, box}}, {}}}); },
                "Invalid GT box accepted");
        rejects([box] { (void)evaluate_tracking({{1, {}, {{1, box}}}}); },
                "Invalid prediction box accepted");
    }
    auto inverted = object(1);
    inverted.bbox.y2 = -1;
    rejects([&] { (void)evaluate_tracking({{1, {inverted}, {}}}); },
            "Mutated inverted box accepted");

    std::vector<TrackingFrame> too_many_frames(TrackingEvaluationLimits::max_frames + 1);
    rejects([&] { (void)evaluate_tracking(too_many_frames); }, "Frame count limit ignored");
    std::vector<TrackingFrame> at_frame_limit(TrackingEvaluationLimits::max_frames);
    for (std::size_t i = 0; i < at_frame_limit.size(); ++i)
        at_frame_limit[i].frame_index = static_cast<int>(i + 1);
    require(evaluate_tracking(at_frame_limit).frames == at_frame_limit.size(),
            "Inclusive frame-count boundary was rejected");
    rejects([] { (void)evaluate_tracking({{1, identities(501), {}}}); },
            "GT per-frame cap ignored");
    rejects([] { (void)evaluate_tracking({{1, {}, identities(501)}}); },
            "Prediction per-frame cap ignored");
    require(evaluate_tracking({{1, identities(500), {}}}).ground_truth_detections == 500,
            "Inclusive per-frame cap was rejected");
    rejects(
        [] {
            (void)evaluate_tracking({{1, identities(500), {}},
                                     {2, identities(500, 500), {}},
                                     {3, identities(1, 1000), {}}});
        },
        "GT unique-identity cap ignored");
    rejects(
        [] {
            (void)evaluate_tracking({{1, {}, identities(500)},
                                     {2, {}, identities(500, 500)},
                                     {3, {}, identities(1, 1000)}});
        },
        "Prediction unique-identity cap ignored");
    require(evaluate_tracking({{1, identities(500), {}}, {2, identities(500, 500), {}}})
                    .ground_truth_identities == 1000,
            "Inclusive unique-identity cap was rejected");

    // These inputs must fail during validation, before any cubic assignment.
    const auto many = identities(500);
    std::vector<TrackingFrame> too_many_pairs;
    for (int frame = 1; frame <= 101; ++frame)
        too_many_pairs.push_back({frame, many, many});
    rejects([&] { (void)evaluate_tracking(too_many_pairs); },
            "Frame-pair comparison budget ignored");
    std::vector<TrackingFrame> too_many_detections;
    for (int frame = 1; frame <= 2001; ++frame)
        too_many_detections.push_back({frame, many, {}});
    rejects([&] { (void)evaluate_tracking(too_many_detections); },
            "Total detection budget ignored");
}
void switch_audit() {
    const auto report = evaluate_tracking({{1, {object(1)}, {object(10), object(30, 100)}},
        {2, {object(1)}, {object(20)}}, {3, {object(1)}, {}}, {4, {}, {}},
        {5, {object(1)}, {object(30), object(20, 100)}}});
    require(report.id_switches == 2 && report.switch_events.size() == report.id_switches,
        "Switch audit lost or duplicated CLEAR events");
    const auto& first = report.switch_events[0];
    require(first.frame_index == 2 && first.previous_match_frame == 1 && first.unmatched_frames == 0 &&
        first.gt_absent_frames == 0 && first.ground_truth_id == 1 && first.previous_prediction_id == 10 &&
        first.prediction_id == 20 && first.prediction_first_seen && !first.previous_prediction_visible &&
        first.match_iou == 1, "Consecutive switch event differs");
    const auto& gap = report.switch_events[1];
    require(gap.frame_index == 5 && gap.previous_match_frame == 2 && gap.unmatched_frames == 2 &&
        gap.gt_absent_frames == 1 && !gap.prediction_first_seen && gap.previous_prediction_visible,
        "GT absence, unmatched gap or emitted identity history conflated");
    require(evaluate_tracking({{1, {object(1)}, {object(10)}}, {2, {}, {}},
        {3, {object(1)}, {object(10)}}}).switch_events.empty(), "Reappearance without switch emitted event");
}
void hota_protocol() {
    const auto perfect = evaluate_hota({{1, {object(1)}, {object(9)}}, {2, {object(1)}, {object(9)}}});
    near(perfect.hota, 1, "Perfect HOTA differs");
    near(perfect.detection_accuracy, 1, "Perfect DetA differs");
    near(perfect.association_accuracy, 1, "Perfect AssA differs");
    require(perfect.thresholds.size() == 19 && perfect.thresholds[18].true_positives == 2,
            "HOTA threshold grid/counts differ");
    const auto split = evaluate_hota({{1, {object(1)}, {object(9)}},
        {2, {object(1)}, {object(10)}}, {3, {object(1)}, {object(10)}}});
    near(split.hota, std::sqrt(5.0 / 9.0), "HOTA identity fragmentation formula differs");
    near(split.association_accuracy, 5.0 / 9.0, "AssA must weight identity Jaccard by matched detections");
    std::vector<TrackingFrame> global{{1, {object(1), object(2, 4)}, {object(9, 4), object(10)}}};
    for (int i = 2; i <= 11; ++i)
        global.push_back({i, {object(1), object(2, 4)}, {object(9), object(10, 4)}});
    const auto alignment = evaluate_hota(global);
    near(alignment.hota, 103.0 / 114.0, "HOTA did not use sequence-wide soft alignment");
    require(alignment.thresholds[9].true_positives == 20,
        "HOTA reassigned below-alpha pairs instead of filtering the single global-aligned assignment");
    const auto partial = evaluate_hota({{1, {object(1)}, {{9, {0, 0, 20, 10}}}}});
    near(partial.hota, 10.0 / 19.0, "HOTA must mean per-alpha scores and include threshold equality");
    require(partial.thresholds[9].true_positives == 1 && partial.thresholds[10].false_negatives == 1 &&
        partial.thresholds[10].false_positives == 1, "HOTA threshold counts differ");
    const auto misses = evaluate_hota({{1, {object(1)}, {}}, {2, {}, {object(9)}}});
    near(misses.hota, 0, "Unmatched HOTA differs");
    near(misses.localization_accuracy, 1, "No-TP LocA convention differs");
    require(!evaluate_hota({}).hota && !evaluate_hota({{1, {}, {object(9)}}}).hota,
        "No-GT HOTA must be undefined");
    near(evaluate_hota({{1, {object(1)}, {}}}).hota, 0, "Empty predictions must produce zero HOTA");
    rejects([] { (void)evaluate_hota({{2, {object(1)}, {}}}); }, "HOTA accepted noncontiguous frames");
    rejects([] { (void)evaluate_hota({{1, {object(1), object(1)}, {}}}); }, "HOTA accepted duplicate IDs");
    rejects([] { (void)evaluate_hota({{1, identities(501), {}}}); }, "HOTA ignored per-frame cap");
    auto invalid = object(1); invalid.bbox.x1 = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)evaluate_hota({{1, {invalid}, {}}}); }, "HOTA accepted NaN boxes");
    const auto large = identities(500);
    std::vector<TrackingFrame> excess;
    for (int i = 1; i <= 101; ++i) excess.push_back({i, large, large});
    rejects([&] { (void)evaluate_hota(excess); }, "HOTA ignored work budget before allocation/assignment");
}
} // namespace

int main() {
    try {
        perfect_and_counting();
        continuity_and_global_identity();
        gaps_and_switch_history();
        undefined_scores();
        validation_and_limits();
        hota_protocol();
        switch_audit();
        std::cout << "CLEAR continuity, global identity, gap and bounded-input tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
