#pragma once

#include "aegisvision/contracts.hpp"
#include "aegisvision/yolo.hpp"
#include <filesystem>
#include <functional>
#include <stdexcept>

namespace aegisvision {
struct DirectoryIndexConfig {
    bool recursive{false};
    std::size_t max_images{10000};
};
struct VideoIndexConfig {
    int frame_stride{30};
    int max_frames{0}; // 0 = entire local file; counts decoded, not sampled frames.
    double fallback_fps{25.0};
    std::string detector_signature; // Model + preprocessing + detector settings identity.
    // Trusted provenance, not user request fields. Cannot override reserved crop fields.
    std::map<std::string,std::string> source_metadata;
    std::vector<std::map<std::string,std::string>> frame_metadata;
};
struct IndexSummary {
    std::string source_id;
    std::size_t indexed_items{};
    std::size_t skipped_entries{};
    std::uint64_t decoded_frames{};
    std::uint64_t sampled_frames{};
    double source_fps{};
    bool used_fallback_fps{};
    std::string stop_reason;
};
using IndexProgress = std::function<void(const IndexSummary&)>;
using IndexCancellation = std::function<bool()>;
class IndexCancelled : public std::runtime_error {
public:
    IndexCancelled() : std::runtime_error("Indexing cancelled; completed writes are retained") {}
};

// Upserts only: a failure leaves completed writes intact. Retrying unchanged inputs
// uses the same IDs. These functions do not delete stale records or create collections.
[[nodiscard]] IndexSummary index_directory(const std::filesystem::path& directory,
    IEmbedder& embedder, IVectorStore& store, const DirectoryIndexConfig& config = {},
    const IndexProgress& progress = {}, const IndexCancellation& cancelled = {});
[[nodiscard]] IndexSummary index_video(const std::filesystem::path& input,
    IDetector& detector, IEmbedder& embedder, IVectorStore& store,
    const VideoIndexConfig& config, const IndexProgress& progress = {}, const IndexCancellation& cancelled = {});
[[nodiscard]] std::string yolo_index_signature(const std::filesystem::path& model,
    const vision::YoloConfig& config);
}
