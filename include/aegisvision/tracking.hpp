#pragma once

#include "aegisvision/contracts.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace aegisvision {

class IoUTracker final : public ITracker {
public:
    explicit IoUTracker(float iou_threshold = 0.30F, std::uint32_t max_missed_frames = 20);
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& detections) override;

private:
    float iou_threshold_;
    std::uint32_t max_missed_frames_;
    std::uint64_t next_id_{1};
    std::unordered_map<std::uint64_t, Track> tracks_;
};

}  // namespace aegisvision

