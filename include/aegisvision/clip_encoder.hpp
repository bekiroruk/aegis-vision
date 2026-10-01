#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>

namespace aegisvision::vision {

struct ArchiveEncodeConfig {
    // Trusted server-startup setting, never supplied by an HTTP request.
    std::filesystem::path executable{"ffmpeg"};
    std::chrono::milliseconds timeout{30000};
    std::uint64_t max_output_bytes{12ULL * 1024 * 1024};
    std::uint64_t expected_frames{};
    double fps{};
};

struct EncodedArchiveClip {
    std::uint64_t frames{}, bytes{};
    int width{}, height{};
    double fps{};
};

// Converts a sealed, regular MJPEG AVI (<=12 MiB, <=1000 frames) to H.264 MP4.
// No shell, URLs, inherited sockets, or unbounded stderr logs. A single owned
// child is cancelled/reaped on timeout. Existing output and .partial.mp4 files
// are rejected, not overwritten. Only fully decoded, frame-count/FPS-verified
// output is atomically published without clobbering an existing destination.
// Normal failures remove this invocation's temporary file, never the source or
// a final MP4. A process crash can leave the fixed partial for owned-job recovery.
// The caller supplies an exclusively owned, trusted archive directory.
[[nodiscard]] EncodedArchiveClip encode_archive_clip(
    const std::filesystem::path& input_avi,
    const std::filesystem::path& output_mp4,
    const ArchiveEncodeConfig& config,
    const std::atomic_bool& cancel);

// Durable replay only: caller first verifies the sealed source fingerprint and
// its owned encoding ticket. This validates an already published clip without
// starting a process or changing any file. It is not an overwrite/reuse policy.
[[nodiscard]] EncodedArchiveClip validate_archive_clip(
    const std::filesystem::path& input_avi,
    const std::filesystem::path& output_mp4,
    const ArchiveEncodeConfig& config,
    const std::atomic_bool& cancel);

}  // namespace aegisvision::vision
