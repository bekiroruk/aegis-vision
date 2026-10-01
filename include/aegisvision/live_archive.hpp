#pragma once

#include "aegisvision/live.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <memory>

namespace aegisvision::vision {
struct LiveArchiveConfig {
    bool enabled{false};
    std::filesystem::path root;
    int segment_seconds{10}, max_frames{100};
    int max_segments_per_session{4}, max_total_segments{8};
    std::uintmax_t max_bytes{256ULL * 1024 * 1024};
    std::uintmax_t raw_limit{12ULL * 1024 * 1024};
    double output_fps{10}; // CFR analyzed-frame playback, not camera PTS.
};
struct ArchivedSegment {
    std::filesystem::path directory, raw_path, media_path, manifest_path;
    std::string session_id, source_id;
    int index{}, frames{};
    double fps{};
};
// Used only on the live analysis worker. Storage/encoder/admission errors are
// contained here so archiving never turns a working preview into a failed session.
// Closed paths are immutable and retained, including segments awaiting a retry.
class LiveArchive {
public:
    using Callback = std::function<void(const ArchivedSegment&)>;
    LiveArchive(LiveArchiveConfig config, std::string session_id,
        std::string source_id, Callback finalized = {});
    ~LiveArchive();
    LiveArchive(const LiveArchive&) = delete;
    LiveArchive& operator=(const LiveArchive&) = delete;
    void accept(const LiveFrame& frame, int tracking_epoch) noexcept;
    void finish() noexcept;
    nlohmann::json snapshot() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aegisvision::vision
