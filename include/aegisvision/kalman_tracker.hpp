#pragma once

#include "aegisvision/contracts.hpp"

#include <array>
#include <map>

namespace aegisvision {

enum class KalmanGateMode { FullBox, CenterOnly };

inline constexpr double kalman_box_gate99 = 13.2767;
inline constexpr double kalman_center_gate99 = 9.2103;

struct KalmanTrackerConfig {
    float low_threshold{0.10F};
    float high_threshold{0.35F};
    float new_track_threshold{0.50F};
    float match_iou{0.30F};
    std::uint32_t max_missed_frames{20};
    // Squared Mahalanobis gate. The legacy default has four measurements and
    // nominal 99% chi-square coverage under calibrated Gaussian innovations,
    // not empirical accuracy. CenterOnly callers should explicitly select
    // kalman_center_gate99 for two measurements at the same nominal coverage.
    double gating_threshold{kalman_box_gate99};
    KalmanGateMode gate_mode{KalmanGateMode::FullBox};
    // Optional local appearance association. All candidates still have to pass
    // the class, IoU and selected motion gate; no gallery/global Re-ID exists.
    bool use_appearance{false};
    // Dimension [1,1024], distance/weight [0,1], momentum [0,1).
    std::size_t appearance_dimension{512};
    double max_cosine_distance{0.20};
    double appearance_weight{0.50};
    double appearance_momentum{0.90};
};

struct KalmanTrackingStats {
    std::uint64_t low_confidence_matches{};
    std::uint64_t reactivations{};
    std::uint64_t created_tracks{};
    std::uint64_t expired_tracks{};
    std::uint64_t gate_rejections{};
    std::uint64_t numerical_resets{};
    std::uint64_t capacity_rejections{};
    // Geometry-valid candidate pairs rejected by the appearance distance gate.
    std::uint64_t appearance_rejections{};
    // Accepted high/low assignments with appearance enabled, excluding births.
    std::uint64_t appearance_matches{};
    // Renormalized high-confidence matched EMA updates, excluding low/births.
    std::uint64_t appearance_updates{};
};

// Bounded, class-aware constant-velocity Kalman matching. The eight states are
// (cx, cy, width, height, vx, vy, vwidth, vheight), with dt = one decoded frame.
// Association order: active/high, remaining active/low, then lost/remaining high.
// CenterOnly gates marginal center uncertainty, not box-size residuals. Both
// modes still require IoU/class agreement and correct all four measurements.
// Only observed detections are returned; predictions are never visible objects.
// More than 512 observations reject the whole update; at 256 live identities,
// valid unmatched births are suppressed and counted without consuming an ID.
// Optional appearance keeps one normalized EMA prototype per live identity.
// Eligible observations must provide finite nonzero embeddings of the configured
// dimension; only high-confidence matches update the prototype. Disabled mode
// does not require, inspect, copy or score detection embeddings.
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
        std::vector<double> appearance;
    };

    KalmanTrackerConfig config_;
    std::map<std::uint64_t, State> states_;
    std::uint64_t next_id_{1};
    KalmanTrackingStats stats_;
};

}  // namespace aegisvision
