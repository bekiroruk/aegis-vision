#pragma once

#include "aegisvision/contracts.hpp"

#include <array>
#include <map>

namespace aegisvision {

struct KalmanTrackerConfig {
    float low_threshold{0.10F};
    float high_threshold{0.35F};
    float new_track_threshold{0.50F};
    float match_iou{0.30F};
    std::uint32_t max_missed_frames{20};
    // Squared Mahalanobis gate: chi-square, four measurements, 99% coverage.
    double gating_threshold{13.2767};
};

struct KalmanTrackingStats {
    std::uint64_t low_confidence_matches{};
    std::uint64_t reactivations{};
    std::uint64_t created_tracks{};
    std::uint64_t expired_tracks{};
    std::uint64_t gate_rejections{};
    std::uint64_t numerical_resets{};
    std::uint64_t capacity_rejections{};
};

// Bounded, class-aware constant-velocity Kalman matching. The eight states are
// (cx, cy, width, height, vx, vy, vwidth, vheight), with dt = one decoded frame.
// Association order: active/high, remaining active/low, then lost/remaining high.
// Only observed detections are returned; predictions are never visible objects.
// More than 512 observations reject the whole update; at 256 live identities,
// valid unmatched births are suppressed and counted without consuming an ID.
// This is a project-specific backend, not official ByteTrack, SORT or Re-ID.
class KalmanTracker final : public ITracker {
public:
    static constexpr std::size_t max_tracks = 256;
    static constexpr std::size_t max_detections = 512;

    explicit KalmanTracker(KalmanTrackerConfig config = {});
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& detections) override;
    [[nodiscard]] const KalmanTrackingStats& stats() const noexcept { return stats_; }

private:
    struct State {
        Track track;
        std::array<double, 8> mean{};
        std::array<double, 64> covariance{};
    };

    KalmanTrackerConfig config_;
    std::map<std::uint64_t, State> states_;
    std::uint64_t next_id_{1};
    KalmanTrackingStats stats_;
};

}  // namespace aegisvision
