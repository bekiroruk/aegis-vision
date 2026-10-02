#include "aegisvision/evaluation.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using aegisvision::BoundingBox;
using aegisvision::Detection;
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
Detection prediction(const BoundingBox box, const float score = 0.9F,
                     std::string label = "person") {
    return {box, std::move(label), score, {}, {}};
}
GroundTruth truth(const BoundingBox box, std::string label = "person", const bool crowd = false) {
    return {box, std::move(label), crowd};
}
Frame perfect_frame(std::string id = "one") {
    const BoundingBox box{-10, -5, 10, 15};
    return {std::move(id), {truth(box)}, {prediction(box)}};
}

void perfect_and_iou_grid() {
    const auto report = evaluate_detection({perfect_frame()}, {"person"});
    require(report.image_count == 1 && report.ground_truth_count == 1 &&
                report.prediction_count == 1 && report.evaluated_prediction_count == 1,
            "Perfect detection counts differ");
    near(report.ap50, 1, "Perfect AP50 differs");
    near(report.ap50_95, 1, "Perfect mean AP differs");
    near(report.operating_point.precision, 1, "Perfect precision differs");
    near(report.operating_point.recall, 1, "Perfect recall differs");
    for (std::size_t i = 0; i < report.iou_thresholds.size(); ++i) {
        require(report.iou_thresholds[i] ==
                    (i == 9 ? .95 : .50 + static_cast<double>(i) * ((.95 - .50) / 9.0)),
                "IoU grid differs");
        near(report.classes[0].ap_by_iou[i], 1, "Perfect per-IoU AP differs");
    }

    const auto half_overlap = evaluate_detection(
        {{"half", {truth({0, 0, 10, 10})}, {prediction({0, 0, 20, 10})}}}, {"person"});
    near(half_overlap.ap50, 1, "IoU exactly equal to threshold must match");
    near(half_overlap.ap50_95, 0.1, "Only the first IoU threshold should match");
    for (std::size_t i = 1; i < 10; ++i)
        near(half_overlap.classes[0].ap_by_iou[i], 0, "Excess IoU threshold matched");

    // Finite float coordinates may have areas far above FLT_MAX. Evaluation
    // must widen before arithmetic, and must not clip negative coordinates.
    const float extent = std::numeric_limits<float>::max();
    const BoundingBox huge{-extent, -extent, extent, extent};
    const auto large =
        evaluate_detection({{"huge", {truth(huge)}, {prediction(huge)}}}, {"person"});
    near(large.ap50_95, 1, "Large finite boxes overflowed or were clipped");
}

void categories_and_recall_interpolation() {
    const auto macro = evaluate_detection(
        {{"one",
          {truth({0, 0, 10, 10}), truth({20, 0, 30, 10}, "car")},
          {prediction({0, 0, 10, 10}), prediction({50, 0, 60, 10}, 0.8F, "unused")}}},
        {"person", "car", "unused"});
    near(macro.classes[0].ap50, 1, "Detected category AP differs");
    near(macro.classes[1].ap50, 0, "Missing category AP must be zero");
    require(!macro.classes[2].ap50 && !macro.classes[2].ap50_95,
            "Category with no normal GT must have undefined AP");
    near(macro.ap50_95, 0.5, "Macro AP included a category with no GT");
    require(macro.operating_point.true_positives == 1 &&
                macro.operating_point.false_positives == 1 &&
                macro.operating_point.false_negatives == 1,
            "Aggregate operating counts differ");
    near(macro.operating_point.precision, 0.5, "No-GT category false positive was omitted");

    const auto subset = evaluate_detection(
        {perfect_frame("one"), {"two", {truth({0, 0, 10, 10})}, {}}}, {"person"});
    // COCO samples recall 0.00 through 1.00 inclusively: 51 samples at <= .50.
    near(subset.ap50, 51.0 / 101.0, "101-point recall interpolation differs");
    near(subset.ap50_95, 51.0 / 101.0, "Mean AP recall interpolation differs");

    const auto wrong_class = evaluate_detection(
        {{"one", {truth({0, 0, 10, 10})}, {prediction({0, 0, 10, 10}, 0.9F, "car")}}},
        {"person", "car"});
    require(wrong_class.operating_point.true_positives == 0 &&
                wrong_class.operating_point.false_positives == 1 &&
                wrong_class.operating_point.false_negatives == 1,
            "An overlapping prediction matched another category");
}

void stable_scores_and_per_category_cap() {
    const auto false_first = evaluate_detection(
        {{"one",
          {truth({0, 0, 10, 10})},
          {prediction({20, 0, 30, 10}, 0.5F), prediction({0, 0, 10, 10}, 0.5F)}}},
        {"person"});
    near(false_first.ap50, 0.5, "Equal-score prediction order was not stable");
    const auto true_first = evaluate_detection(
        {{"one",
          {truth({0, 0, 10, 10})},
          {prediction({0, 0, 10, 10}, 0.5F), prediction({20, 0, 30, 10}, 0.5F)}}},
        {"person"});
    near(true_first.ap50, 1, "Trailing duplicate/false positive changed completed recall AP");

    const Frame no_truth{"false", {}, {prediction({20, 0, 30, 10}, 0.5F)}};
    const Frame with_truth{"true", {truth({0, 0, 10, 10})}, {prediction({0, 0, 10, 10}, 0.5F)}};
    near(evaluate_detection({no_truth, with_truth}, {"person"}).ap50, 0.5,
         "Equal-score ordering across images was not stable");
    near(evaluate_detection({with_truth, no_truth}, {"person"}).ap50, 1,
         "Input image order was not used for equal scores");

    EvaluationConfig capped;
    capped.max_detections_per_image_per_class = 1;
    const auto cap =
        evaluate_detection({{"one",
                             {truth({0, 0, 10, 10}), truth({40, 0, 50, 10}, "car")},
                             {prediction({20, 0, 30, 10}, 0.99F), prediction({0, 0, 10, 10}, 0.9F),
                              prediction({40, 0, 50, 10}, 0.8F, "car")}}},
                           {"person", "car"}, capped);
    require(cap.prediction_count == 3 && cap.evaluated_prediction_count == 2,
            "Cap must retain raw counts and apply independently to each category");
    near(cap.classes[0].ap50, 0, "Cap was not applied after score sorting");
    near(cap.classes[1].ap50, 1, "Another category consumed this category's cap");
    near(cap.ap50, 0.5, "Capped macro AP differs");

    const auto duplicate =
        evaluate_detection({{"one",
                             {truth({0, 0, 10, 10})},
                             {prediction({0, 0, 10, 10}, 0.9F), prediction({0, 0, 10, 10}, 0.8F)}}},
                           {"person"});
    require(duplicate.operating_point.true_positives == 1 &&
                duplicate.operating_point.false_positives == 1,
            "Normal ground truth matched more than once");
}

void crowd_and_operating_thresholds() {
    // Crowds precede normal truth in the supplied data, deliberately. Repeated
    // small contained predictions must be ignored by IoA, not by box IoU.
    const auto crowd = evaluate_detection(
        {{"one",
          {truth({0, 0, 100, 100}, "person", true), truth({0, 0, 10, 10})},
          {prediction({20, 20, 25, 25}, 0.99F), prediction({30, 30, 35, 35}, 0.95F),
           prediction({0, 0, 10, 10}, 0.9F), prediction({200, 200, 210, 210}, 0.8F)}}},
        {"person"});
    require(crowd.ground_truth_count == 1 && crowd.crowd_ground_truth_count == 1,
            "Crowd was included in the recall denominator");
    require(crowd.operating_point.true_positives == 1 &&
                crowd.operating_point.false_positives == 1 &&
                crowd.operating_point.ignored_predictions == 2,
            "Crowd IoA matching/reuse differs");
    near(crowd.ap50_95, 1, "Ignored crowd predictions reduced AP");

    const auto precedence =
        evaluate_detection({{"one",
                             {truth({0, 0, 100, 100}, "person", true), truth({0, 0, 10, 10})},
                             {prediction({0, 0, 20, 10})}}},
                           {"person"});
    require(precedence.operating_point.true_positives == 1 &&
                precedence.operating_point.ignored_predictions == 0,
            "Higher crowd IoA displaced a qualifying normal annotation");
    near(precedence.ap50_95, 0.1, "Normal-priority IoU grid differs");

    const auto crowd_only = evaluate_detection(
        {{"one", {truth({0, 0, 100, 100}, "person", true)}, {prediction({10, 10, 20, 20})}}},
        {"person"});
    require(!crowd_only.ap50 && !crowd_only.ap50_95 && !crowd_only.operating_point.precision &&
                !crowd_only.operating_point.recall,
            "Crowd-only scores with zero denominators must be undefined");
    require(crowd_only.operating_point.ignored_predictions == 1,
            "Crowd-only matching was not performed");

    const auto cutoff = evaluate_detection(
        {{"one",
          {truth({0, 0, 10, 10})},
          {prediction({0, 0, 10, 10}, 0.25F), prediction({20, 0, 30, 10}, 0.249F)}}},
        {"person"});
    require(cutoff.operating_point.true_positives == 1 &&
                cutoff.operating_point.false_positives == 0,
            "Operating score cutoff must be inclusive and exclude lower scores");
    const auto low_score = evaluate_detection(
        {{"one", {truth({0, 0, 10, 10})}, {prediction({0, 0, 10, 10}, 0.1F)}}}, {"person"});
    near(low_score.ap50_95, 1, "Operating threshold incorrectly filtered the AP curve");
    require(low_score.operating_point.false_negatives == 1 && !low_score.operating_point.precision,
            "Low-score operating counts differ");
}

void empty_inputs_and_validation() {
    const auto empty = evaluate_detection({}, {"person"});
    require(empty.image_count == 0 && !empty.ap50 && !empty.operating_point.precision &&
                !empty.operating_point.recall,
            "Empty detection scores must be undefined");
    const auto no_prediction =
        evaluate_detection({{"one", {truth({0, 0, 10, 10})}, {}}}, {"person"});
    near(no_prediction.ap50_95, 0, "GT without predictions must have zero AP");
    near(no_prediction.operating_point.recall, 0, "GT without predictions must have zero recall");
    require(!no_prediction.operating_point.precision, "No predictions fabricated precision");
    const auto no_truth =
        evaluate_detection({{"one", {}, {prediction({0, 0, 10, 10})}}}, {"person"});
    require(!no_truth.ap50 && !no_truth.operating_point.recall, "No GT fabricated AP/recall");
    near(no_truth.operating_point.precision, 0, "Unmatched predictions must have zero precision");

    rejects([] { (void)evaluate_detection({}, {}); }, "Empty taxonomy accepted");
    rejects([] { (void)evaluate_detection({}, {"person", "person"}); }, "Duplicate class accepted");
    rejects([] { (void)evaluate_detection({}, std::vector<std::string>(1025, "person")); },
            "Taxonomy size limit ignored");
    rejects([] { (void)evaluate_detection({}, {""}); }, "Empty class accepted");
    rejects([] { (void)evaluate_detection({}, {std::string(257, 'a')}); },
            "Oversized class accepted");
    rejects([] { (void)evaluate_detection({}, {std::string("bad\0class", 9)}); },
            "NUL class accepted");
    rejects([] { (void)evaluate_detection({perfect_frame(), perfect_frame()}, {"person"}); },
            "Duplicate frame IDs accepted");
    rejects([] { (void)evaluate_detection({perfect_frame("")}, {"person"}); },
            "Empty frame ID accepted");
    rejects([] { (void)evaluate_detection({perfect_frame(std::string(1025, 'x'))}, {"person"}); },
            "Oversized frame ID accepted");
    rejects(
        [] { (void)evaluate_detection({perfect_frame(std::string("bad\0id", 6))}, {"person"}); },
        "NUL frame ID accepted");
    rejects(
        [] { (void)evaluate_detection({{"one", {truth({0, 0, 10, 10}, "car")}, {}}}, {"person"}); },
        "Unknown GT category accepted");
    rejects(
        [] {
            (void)evaluate_detection({{"one", {}, {prediction({0, 0, 10, 10}, 0.9F, "car")}}},
                                     {"person"});
        },
        "Unknown prediction category accepted");
    for (const float score : {-0.1F, 1.1F, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()}) {
        rejects(
            [score] {
                (void)evaluate_detection({{"one", {}, {prediction({0, 0, 10, 10}, score)}}},
                                         {"person"});
            },
            "Invalid probability accepted");
    }
    for (const auto box : {BoundingBox{0, 0, 0, 10}, BoundingBox{0, 0, 10, 0},
                           BoundingBox{std::numeric_limits<float>::quiet_NaN(), 0, 10, 10},
                           BoundingBox{0, 0, std::numeric_limits<float>::infinity(), 10}}) {
        rejects([box] { (void)evaluate_detection({{"one", {truth(box)}, {}}}, {"person"}); },
                "Invalid GT box accepted");
        rejects([box] { (void)evaluate_detection({{"one", {}, {prediction(box)}}}, {"person"}); },
                "Invalid prediction box accepted");
    }
    auto inverted = perfect_frame();
    inverted.predictions[0].bbox.x2 = inverted.predictions[0].bbox.x1 - 1;
    rejects([&] { (void)evaluate_detection({inverted}, {"person"}); },
            "Mutated inverted box accepted");
    for (const std::size_t cap : {std::size_t{0}, std::size_t{100001}}) {
        EvaluationConfig config;
        config.max_detections_per_image_per_class = cap;
        rejects([&] { (void)evaluate_detection({}, {"person"}, config); },
                "Invalid detection cap accepted");
    }
    for (const double score : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
        EvaluationConfig config;
        config.operating_score_threshold = score;
        rejects([&] { (void)evaluate_detection({}, {"person"}, config); },
                "Invalid score threshold accepted");
    }
    for (const double iou : {0.0, -0.1, 1.1, std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
        EvaluationConfig config;
        config.operating_iou_threshold = iou;
        rejects([&] { (void)evaluate_detection({}, {"person"}, config); },
                "Invalid IoU threshold accepted");
    }
    rejects([] { (void)evaluate_detection(std::vector<Frame>(10001), {"person"}); },
            "Detection frame limit ignored");
    rejects(
        [] {
            (void)evaluate_detection(
                {{"one", std::vector<GroundTruth>(10001, truth({0, 0, 10, 10})), {}}}, {"person"});
        },
        "Detection ground-truth per-frame limit ignored");
    rejects(
        [] {
            (void)evaluate_detection(
                {{"one", {}, std::vector<Detection>(10001, prediction({0, 0, 10, 10}))}},
                {"person"});
        },
        "Detection prediction per-frame limit ignored");
}

void latency_statistics() {
    require(!summarize_latencies({}), "Empty latency summary fabricated samples");
    const auto one = summarize_latencies({0});
    require(one && one->sample_count == 1 && one->mean_ms == 0 && one->p95_ms == 0,
            "One zero-duration sample differs");
    const auto samples = summarize_latencies({20, 3, 1, 7, 2});
    require(samples && samples->sample_count == 5 && samples->min_ms == 1 && samples->p50_ms == 3 &&
                samples->p95_ms == 20 && samples->max_ms == 20 &&
                std::abs(samples->mean_ms - 6.6) < 1e-12,
            "Nearest-rank latency statistics differ");
    const auto even = summarize_latencies({4, 1, 3, 2});
    require(even && even->p50_ms == 2, "Even-size median was interpolated instead of nearest rank");
    for (const double value :
         {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        rejects([value] { (void)summarize_latencies({1, value}); }, "Invalid latency accepted");
}
void official_recall_grid_rounding() {
    std::vector<Frame> frames;
    for (int i = 0; i < 100; ++i) {
        Frame frame{std::to_string(i), {truth({0, 0, 10, 10})}, {}};
        if (i < 70)
            frame.predictions.push_back(prediction({0, 0, 10, 10}));
        frames.push_back(std::move(frame));
    }
    const auto result = evaluate_detection(frames, {"person"});
    near(result.ap50, 70.0 / 101.0, "Recall grid must match NumPy linspace rounding at .70");
    near(result.ap50_95, 70.0 / 101.0, "Mean AP must use official recall grid rounding");
}
} // namespace

int main() {
    try {
        perfect_and_iou_grid();
        categories_and_recall_interpolation();
        stable_scores_and_per_category_cap();
        crowd_and_operating_thresholds();
        empty_inputs_and_validation();
        latency_statistics();
        official_recall_grid_rounding();
        std::cout << "Detection AP, crowd matching, operating point and latency tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
