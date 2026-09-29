#include "aegisvision/adapters.hpp"
#include "aegisvision/geometry.hpp"
#include "aegisvision/pipeline.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/vector_store.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void check(const bool condition, const std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_bounding_box_iou() {
    const aegisvision::BoundingBox first{0, 0, 10, 10};
    const aegisvision::BoundingBox second{5, 5, 15, 15};
    check(std::abs(first.iou(second) - (25.0F / 175.0F)) < 1e-6F, "bounding box IoU");
}

void test_tracking_keeps_identity() {
    aegisvision::IoUTracker tracker;
    const auto first = tracker.update({
        aegisvision::Detection{aegisvision::BoundingBox{0, 0, 100, 100}, "person", 0.9F}});
    const auto second = tracker.update({
        aegisvision::Detection{aegisvision::BoundingBox{5, 0, 105, 100}, "person", 0.8F}});
    check(first.front().track_id == second.front().track_id, "tracker identity continuity");
    check(second.front().age == 2, "tracker age increment");
}

void test_vector_search() {
    aegisvision::InMemoryVectorStore store;
    store.upsert("red", {1, 0}, {{"label", "red"}});
    store.upsert("blue", {0, 1}, {{"label", "blue"}});
    check(store.search({0.9F, 0.1F}).front().item_id == "red", "nearest vector first");
}

void test_homography() {
    const aegisvision::Homography translation{{
        {{1, 0, 10}}, {{0, 1, -5}}, {{0, 0, 1}},
    }};
    const auto [x, y] = aegisvision::transform_point({2, 3}, translation);
    check(x == 12 && y == -2, "homography translation");
}

void test_pipeline() {
    aegisvision::PayloadDetector detector;
    aegisvision::IoUTracker tracker;
    aegisvision::HashEmbedder embedder;
    aegisvision::InMemoryVectorStore store;
    aegisvision::AnalysisPipeline pipeline{
        aegisvision::PipelineConfig{}, detector, &tracker, &embedder, nullptr, &store};
    const aegisvision::Frame frame{"1", "cam", 0, {
        aegisvision::Detection{aegisvision::BoundingBox{0, 0, 10, 10}, "person", 0.9F},
    }};
    const auto result = pipeline.analyze(frame);
    check(result.detections.size() == 1, "pipeline detection count");
    check(result.tracks.size() == 1, "pipeline track count");
    check(result.indexed_items == std::vector<std::string>{"cam:1:0"}, "pipeline indexing");
}
}  // namespace

int main() {
    try {
        test_bounding_box_iou();
        test_tracking_keeps_identity();
        test_vector_search();
        test_homography();
        test_pipeline();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        return 2;
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All AegisVision tests passed\n";
    return 0;
}

