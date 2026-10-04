#pragma once
#include "aegisvision/contracts.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>

namespace aegisvision::evaluation {
struct QualityRunConfig {
    int warmup_iterations{5};
    nlohmann::json provenance = nlohmann::json::object();
    bool compare_kalman{false}; // Preserve the two-panel baseline by default.
    bool compare_kalman_center{false}; // Independent opt-in, same detections as the old backend.
    IEmbedder *appearance_embedder{nullptr}; // Optional person model; caller retains ownership.
};
// Local immutable manifest, bounded images/video, new/empty output directory.
// Model loading is outside this function. Ground truth NEVER enters a tracker.
// Writes raw predictions, timings, metrics and (video only) a comparison AVI.
[[nodiscard]] nlohmann::json run_quality_benchmark(const std::filesystem::path &manifest,
                                                   const std::filesystem::path &output,
                                                   IDetector &detector,
                                                   const QualityRunConfig &config = {});
[[nodiscard]] std::string quality_file_sha256(const std::filesystem::path &path);
} // namespace aegisvision::evaluation
