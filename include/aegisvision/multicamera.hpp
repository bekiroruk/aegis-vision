#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aegisvision {
// Offline, closed person tracks only. Times MUST share a synchronized clock.
// Camera + epoch + local_id is the key; IDs restarting after reconnect need a new epoch.
struct CameraTracklet {
    std::string camera;
    std::uint64_t epoch{}, local_id{};
    std::int64_t start_ms{}, end_ms{};
    std::string embedding_space;
    std::vector<float> embedding; // One 512-D person prototype, never a CLIP/text vector.
};
struct CameraTransition {
    std::string from, to;
    std::int64_t min_gap_ms{1}, max_gap_ms{30000};
};
struct MultiCameraConfig {
    std::vector<std::string> cameras;
    std::vector<CameraTransition> transitions; // Directed, explicit, non-overlapping cameras.
    std::string embedding_space;
    double min_similarity{.85}; // Uncalibrated initial values, not probabilities.
    double ambiguity_margin{.05};
};
struct CameraIdentity {
    std::uint64_t global_id{}; // Batch-local hypothesis, not a persistent/verified person ID.
    std::optional<std::size_t> predecessor; // Input index of the accepted earlier tracklet.
    std::optional<double> best_similarity;
    std::string reason;
};
// Pure bounded operation (<=512 tracklets, <=16 cameras). Invalid input throws
// before returning results. Output preserves input order; IDs are deterministic
// under input permutation. Mutual-best edges require row AND column margins.
// No online capture, clock synchronization, identity persistence or overlapping views.
[[nodiscard]] std::vector<CameraIdentity> associate_cameras(
    const std::vector<CameraTracklet>& tracklets, const MultiCameraConfig& config);
}
