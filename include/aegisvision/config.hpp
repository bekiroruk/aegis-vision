#pragma once

#include <cstdint>

namespace aegisvision {

struct TrackingConfig {
    float iou_threshold{0.30F};
    std::uint32_t max_missed_frames{20};
};

struct PipelineConfig {
    bool enable_tracking{true};
    bool enable_embeddings{true};
    bool enable_ocr{false};
    bool index_embeddings{true};
    TrackingConfig tracking{};
};

}  // namespace aegisvision

