#pragma once

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aegisvision {

struct BoundingBox {
    float x1{};
    float y1{};
    float x2{};
    float y2{};

    BoundingBox(float left, float top, float right, float bottom);
    [[nodiscard]] float area() const noexcept;
    [[nodiscard]] float iou(const BoundingBox& other) const noexcept;
};

struct Detection {
    BoundingBox bbox;
    std::string label;
    float score{};
    std::vector<float> embedding;
    std::map<std::string, std::string> attributes;
};

struct Track {
    std::uint64_t track_id{};
    BoundingBox bbox{0.0F, 0.0F, 0.0F, 0.0F};
    std::string label;
    float score{};
    std::uint32_t age{};
    std::uint32_t missed_frames{};
};

// Owned, interleaved 8-bit BGR pixels. The shared const buffer outlives adapters.
// A frame may omit pixels for metadata-only / synthetic detector workflows.
struct ImageBuffer {
    int width{};
    int height{};
    std::size_t stride{};
    std::vector<std::uint8_t> pixels;
};

struct Frame {
    std::string frame_id;
    std::string source_id;
    std::int64_t timestamp_ms{};
    std::vector<Detection> candidate_detections;
    std::shared_ptr<const ImageBuffer> image;
};

struct SearchResult {
    std::string item_id;
    float score{};
    std::map<std::string, std::string> metadata;
};

struct AnalysisResult {
    std::string frame_id;
    std::vector<Detection> detections;
    std::vector<Track> tracks;
    std::vector<std::string> extracted_text;
    std::vector<std::string> indexed_items;
};

}  // namespace aegisvision

