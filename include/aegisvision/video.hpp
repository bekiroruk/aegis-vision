#pragma once

#include "aegisvision/contracts.hpp"
#include "aegisvision/kalman_tracker.hpp"
#include "aegisvision/two_stage_tracker.hpp"
#include <filesystem>

namespace aegisvision::vision {
enum class TrackerMode { IoU, TwoStage, Kalman };
struct VideoConfig {
    TrackerMode tracker_mode{TrackerMode::IoU};
    float tracking_iou{0.3F};
    std::uint32_t max_missed_frames{20};
    int max_frames{0}; // 0 = entire local file, no frame skipping
    double fallback_fps{25.0};
    float low_confidence{0.10F};
    float high_confidence{0.35F};
    float new_track_confidence{0.50F};
    double kalman_gating_threshold{13.2767};
    KalmanGateMode kalman_gate_mode{KalmanGateMode::FullBox};
};
struct VideoSummary {
    int processed_frames{};
    std::size_t unique_track_ids{};
    double source_fps{};
    double processing_fps{};
    double mean_analysis_ms{};
    std::string stop_reason;
    TrackingStats tracking_stats;
    KalmanTrackingStats kalman_stats;
};

// One tracker per invocation/source. The injected detector may own a loaded YOLO model.
// OUTPUT must be new/empty. Partial files remain on error; summary.json is only written on success.
[[nodiscard]] VideoSummary process_video(const std::filesystem::path &input,
                                         const std::filesystem::path &output, IDetector &detector,
                                         const VideoConfig &config = {});
} // namespace aegisvision::vision
