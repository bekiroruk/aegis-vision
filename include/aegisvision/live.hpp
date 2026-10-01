#pragma once

#include "aegisvision/video.hpp"
#include <chrono>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <opencv2/core.hpp>

namespace aegisvision::vision {

struct LiveConfig {
    int duration_seconds{30}; // A bounded recording, not a durable service job.
    int open_timeout_ms{8000}; // Leave room for FFmpeg's initial stream probing.
    int read_timeout_ms{2000};
    int reconnect_initial_ms{250};
    int reconnect_max_ms{2000};
    int max_outage_ms{15000};
    int max_frame_age_ms{1000}; // Decode-arrival age, not camera-to-result latency.
    int tracking_gap_ms{1000};
    int queue_capacity{1};
    double output_fps{10}; // Output contains analyzed frames only, not real-time playback.
};
void validate_live_config(const LiveConfig& config);
// Credentials/query tokens are deliberately unsupported until a secret-safe adapter exists.
void validate_rtsp_url(const std::string& url);

class ILiveCapture {
public:
    virtual ~ILiveCapture() = default;
    virtual bool open(const std::string& url, const LiveConfig& config) = 0;
    virtual bool read(cv::Mat& image) = 0;
    virtual void close() = 0;
};
using LiveCaptureFactory = std::function<std::unique_ptr<ILiveCapture>()>;

struct LiveFrame {
    cv::Mat image; // Independent owned storage, never a decoder's reused buffer.
    std::uint64_t sequence{}, session{};
    std::int64_t arrival_ms{}; // steady-clock elapsed since this source started; NOT PTS.
    std::chrono::steady_clock::time_point arrived;
};
struct LiveStats {
    std::uint64_t connection_attempts{}, sessions{}, read_failures{}, decoded_frames{};
    std::uint64_t delivered_frames{}, dropped_overflow{}, dropped_stale{}, dropped_disconnect{};
    std::size_t queue_high_watermark{};
    std::string stop_reason;
    std::string connection_state{"connecting"};
};

// Exactly one capture thread; bounded drop-oldest queue. All methods except destruction
// are thread-safe. Injected captures must respect the configured open/read timeouts.
class LiveSource {
public:
    LiveSource(std::string url, LiveConfig config = {}, LiveCaptureFactory factory = {});
    ~LiveSource();
    LiveSource(const LiveSource&) = delete;
    LiveSource& operator=(const LiveSource&) = delete;
    std::optional<LiveFrame> next(std::chrono::milliseconds wait = std::chrono::milliseconds(100));
    LiveStats stats() const;
    bool finished() const;
    void stop(); // Joins capture; bounded by an in-flight backend timeout, not instant.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct LiveSummary {
    LiveStats source;
    int processed_frames{}, tracking_epochs{};
    double mean_analysis_ms{}, max_decode_age_at_analysis_ms{};
    std::string stop_reason;
};
// Synchronous worker engine shared by CLI recording and HTTP preview. Callbacks run
// on this worker, must not retain references, and must keep their own queues bounded.
using LiveFrameSink = std::function<void(const LiveFrame&, const AnalysisResult&, const LiveSummary&, double, double)>;
using LiveProgress = std::function<void(const LiveSummary&)>;
LiveSummary analyze_stream(const std::string& url, IDetector& detector,
    const VideoConfig& tracking, const LiveConfig& config, const std::atomic_bool& cancel,
    LiveFrameSink sink, LiveProgress progress = {}, LiveCaptureFactory factory = {});
LiveSummary process_stream(const std::string& url, const std::filesystem::path& output,
    IDetector& detector, const VideoConfig& tracking, const LiveConfig& config = {},
    LiveCaptureFactory factory = {});

} // namespace aegisvision::vision
