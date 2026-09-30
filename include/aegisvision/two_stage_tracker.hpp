#pragma once
#include "aegisvision/contracts.hpp"
#include <map>

namespace aegisvision {
struct TwoStageConfig {
    float low_threshold{0.10F};
    float high_threshold{0.35F};
    float new_track_threshold{0.50F};
    float match_iou{0.30F};
    std::uint32_t max_missed_frames{20};
    bool predict_motion{true};
};
struct TrackingStats {
    std::uint64_t low_confidence_matches{};
    std::uint64_t reactivations{};
    std::uint64_t created_tracks{};
    std::uint64_t expired_tracks{};
};

// ByteTrack-inspired two-stage matching with linear center motion and Hungarian assignment.
// Not the official ByteTrack algorithm: no Kalman covariance, tentative tracks or Re-ID.
class TwoStageTracker final : public ITracker {
public:
    explicit TwoStageTracker(TwoStageConfig config = {});
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& detections) override;
    [[nodiscard]] const TrackingStats& stats() const noexcept { return stats_; }
private:
    struct State {
        Track track;
        float velocity_x{};
        float velocity_y{};
    };
    TwoStageConfig config_;
    std::map<std::uint64_t, State> states_;
    std::uint64_t next_id_{1};
    TrackingStats stats_;
};
}
