#include "aegisvision/live.hpp"
#include "aegisvision/pipeline.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

namespace aegisvision::vision {
namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;
constexpr std::size_t max_frame_bytes = 32 * 1024 * 1024;
constexpr std::size_t max_queue_bytes = 64 * 1024 * 1024;
std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}
class OpenCvCapture final : public ILiveCapture {
public:
    bool open(const std::string& url, const LiveConfig& config) override {
        // Open-only FFmpeg properties; do not silently fall back to a backend
        // that ignores deadlines. The default FFmpeg RTSP transport prefers TCP.
        return capture_.open(url, cv::CAP_FFMPEG, {
            cv::CAP_PROP_OPEN_TIMEOUT_MSEC, config.open_timeout_ms,
            cv::CAP_PROP_READ_TIMEOUT_MSEC, config.read_timeout_ms});
    }
    bool read(cv::Mat& image) override { return capture_.read(image); }
    void close() override { capture_.release(); }
private:
    cv::VideoCapture capture_;
};
std::unique_ptr<ITracker> make_tracker(const VideoConfig& config) {
    if (config.tracker_mode == TrackerMode::Kalman)
        throw std::invalid_argument("Kalman requires consecutive frames; live timestamp-aware dt is not supported");
    if (config.tracker_mode == TrackerMode::IoU)
        return std::make_unique<IoUTracker>(config.tracking_iou, config.max_missed_frames);
    if (config.tracker_mode == TrackerMode::TwoStage)
        return std::make_unique<TwoStageTracker>(TwoStageConfig{config.low_confidence,
            config.high_confidence, config.new_track_confidence, config.tracking_iou,
            config.max_missed_frames, true});
    throw std::invalid_argument("Unknown tracker mode");
}
}

void validate_live_config(const LiveConfig& c) {
    if (c.duration_seconds < 1 || c.duration_seconds > 86400 ||
        c.open_timeout_ms < 1 || c.open_timeout_ms > 15000 ||
        c.read_timeout_ms < 1 || c.read_timeout_ms > 5000 ||
        c.reconnect_initial_ms < 1 || c.reconnect_max_ms < c.reconnect_initial_ms ||
        c.reconnect_max_ms > 5000 || c.max_outage_ms < 1 || c.max_outage_ms > 600000 ||
        c.max_frame_age_ms < 1 || c.max_frame_age_ms > 10000 ||
        c.tracking_gap_ms < 1 || c.tracking_gap_ms > 10000 ||
        c.queue_capacity < 1 || c.queue_capacity > 16 || !std::isfinite(c.output_fps) ||
        c.output_fps < 1 || c.output_fps > 120)
        throw std::invalid_argument("Invalid live stream configuration");
}
void validate_rtsp_url(const std::string& url) {
    if (!url.starts_with("rtsp://") || url.size() > 2048 || url.find_first_of("@?#\\") != std::string::npos ||
        url.find_first_of(" \t\r\n") != std::string::npos || url.size() <= 7 || url[7] == '/')
        throw std::invalid_argument("Expected rtsp://host[:port]/path without credentials or query tokens");
    for (unsigned char c : url) if (c < 32 || c >= 127)
        throw std::invalid_argument("RTSP URL must use printable ASCII (encode the path if needed)");
}

struct LiveSource::Impl {
    std::string url;
    LiveConfig config;
    LiveCaptureFactory factory;
    mutable std::mutex mutex;
    std::mutex join_mutex;
    std::condition_variable changed;
    std::deque<LiveFrame> queue;
    std::size_t queue_bytes{};
    LiveStats stats;
    bool stopping{}, done{};
    Clock::time_point started{Clock::now()};
    std::thread reader;

    bool stop_requested() const { std::lock_guard lock(mutex); return stopping; }
    void clear_queue() {
        stats.dropped_disconnect += queue.size();
        queue.clear(); queue_bytes = 0;
    }
    void run() noexcept {
        auto last_arrival = started;
        int backoff = config.reconnect_initial_ms;
        std::string reason = "stopped";
        try {
            while (!stop_requested()) {
                if (Clock::now() - last_arrival >= Ms(config.max_outage_ms)) {
                    reason = "outage_limit"; break;
                }
                { std::lock_guard lock(mutex); ++stats.connection_attempts;
                  stats.connection_state = stats.sessions ? "reconnecting" : "connecting"; }
                auto capture = factory();
                if (!capture) throw std::runtime_error("Capture factory returned null");
                bool opened = false;
                try { opened = capture->open(url, config); } catch (const std::exception&) {}
                if (opened) {
                    bool first = true;
                    while (!stop_requested()) {
                        cv::Mat image;
                        bool read = false;
                        try { read = capture->read(image); } catch (const std::exception&) {}
                        if (stop_requested()) break;
                        if (!read || image.empty()) {
                            std::lock_guard lock(mutex);
                            ++stats.read_failures; clear_queue(); stats.connection_state = "reconnecting";
                            break;
                        }
                        const auto arrived = Clock::now();
                        if (arrived - last_arrival >= Ms(config.max_outage_ms)) {
                            reason = "outage_limit"; break;
                        }
                        if (image.type() != CV_8UC3 || image.total() > max_frame_bytes / image.elemSize()) {
                            reason = "unsupported_frame"; break;
                        }
                        // Clone before publication even when an injected decoder reuses its buffer.
                        LiveFrame frame{image.clone(), 0, 0,
                            std::chrono::duration_cast<Ms>(arrived - started).count(), arrived};
                        {
                            std::lock_guard lock(mutex);
                            if (first) { ++stats.sessions; first = false; }
                            stats.connection_state = "live";
                            frame.session = stats.sessions;
                            frame.sequence = ++stats.decoded_frames;
                            const auto bytes = frame.image.total() * frame.image.elemSize();
                            while (!queue.empty() && (queue.size() >= static_cast<std::size_t>(config.queue_capacity) ||
                                queue_bytes + bytes > max_queue_bytes)) {
                                queue_bytes -= queue.front().image.total() * queue.front().image.elemSize();
                                queue.pop_front(); ++stats.dropped_overflow;
                            }
                            queue_bytes += bytes;
                            queue.push_back(std::move(frame));
                            stats.queue_high_watermark = std::max(stats.queue_high_watermark, queue.size());
                        }
                        changed.notify_all();
                        last_arrival = arrived;
                        backoff = config.reconnect_initial_ms;
                    }
                }
                capture->close();
                if (reason != "stopped") break;
                std::unique_lock lock(mutex);
                const auto retry_at = std::min(Clock::now() + Ms(backoff), last_arrival + Ms(config.max_outage_ms));
                changed.wait_until(lock, retry_at, [&] { return stopping; });
                backoff = std::min(config.reconnect_max_ms, backoff * 2);
            }
        } catch (const std::exception&) { reason = "capture_error"; }
        catch (...) { reason = "capture_error"; }
        { std::lock_guard lock(mutex); clear_queue(); stats.stop_reason = reason;
          stats.connection_state = "stopped"; done = true; }
        changed.notify_all();
    }
};

LiveSource::LiveSource(std::string url, LiveConfig config, LiveCaptureFactory factory) {
    validate_rtsp_url(url); validate_live_config(config);
    impl_ = std::make_unique<Impl>();
    impl_->url = std::move(url); impl_->config = config;
    impl_->factory = factory ? std::move(factory) : [] { return std::make_unique<OpenCvCapture>(); };
    impl_->reader = std::thread([this] { impl_->run(); });
}
LiveSource::~LiveSource() { stop(); }
void LiveSource::stop() {
    std::lock_guard join_lock(impl_->join_mutex);
    { std::lock_guard lock(impl_->mutex); impl_->stopping = true; }
    impl_->changed.notify_all();
    // Serialize concurrent stop calls; destruction must not race another method.
    if (impl_->reader.joinable()) impl_->reader.join();
}
bool LiveSource::finished() const { std::lock_guard lock(impl_->mutex); return impl_->done; }
LiveStats LiveSource::stats() const { std::lock_guard lock(impl_->mutex); return impl_->stats; }
std::optional<LiveFrame> LiveSource::next(Ms wait) {
    if (wait.count() < 0 || wait.count() > 5000) throw std::invalid_argument("Invalid live poll timeout");
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait_for(lock, wait, [&] { return !impl_->queue.empty() || impl_->done || impl_->stopping; });
    while (!impl_->queue.empty()) {
        auto frame = std::move(impl_->queue.front()); impl_->queue.pop_front();
        impl_->queue_bytes -= frame.image.total() * frame.image.elemSize();
        if (Clock::now() - frame.arrived > Ms(impl_->config.max_frame_age_ms)) {
            ++impl_->stats.dropped_stale; continue;
        }
        ++impl_->stats.delivered_frames;
        return frame;
    }
    return std::nullopt;
}

LiveSummary analyze_stream(const std::string& url, IDetector& detector, const VideoConfig& tracking,
    const LiveConfig& config, const std::atomic_bool& cancel, LiveFrameSink sink,
    LiveProgress progress, LiveCaptureFactory factory) {
    if (tracking.use_appearance)
        throw std::invalid_argument("Appearance tracking requires local video; live appearance is not supported");
    auto tracker = make_tracker(tracking);
    PipelineConfig pipeline_config;
    pipeline_config.enable_embeddings = false; pipeline_config.index_embeddings = false;
    return analyze_live_frames(url,config,cancel,[&](const LiveFrame& frame,bool reset) {
        if (reset) tracker=make_tracker(tracking);
        AnalysisPipeline pipeline(pipeline_config,detector,tracker.get(),nullptr,nullptr,nullptr);
        return pipeline.analyze(image_frame(frame.image,std::to_string(frame.sequence),
            "rtsp-session-"+std::to_string(frame.session),frame.arrival_ms));
    },std::move(sink),std::move(progress),std::move(factory));
}
LiveSummary analyze_live_frames(const std::string& url,const LiveConfig& config,const std::atomic_bool& cancel,
    LiveAnalyzer analyze,LiveFrameSink sink,LiveProgress progress,LiveCaptureFactory factory) {
    validate_rtsp_url(url);validate_live_config(config);
    if (!analyze) throw std::invalid_argument("Live analyzer required");
    if (!sink) throw std::invalid_argument("Live frame sink required");
    if (cancel) { LiveSummary cancelled; cancelled.stop_reason = "cancelled"; return cancelled; }
    LiveSource source(url, config, std::move(factory));
    LiveSummary summary;
    const auto started = Clock::now();
    std::uint64_t session = 0;
    std::int64_t previous_arrival = 0;
    double total_analysis = 0;
    cv::Size previous_size;
    while (true) {
        if (cancel) { summary.stop_reason = "cancelled"; break; }
        if (Clock::now() - started >= std::chrono::seconds(config.duration_seconds)) {
            summary.stop_reason = "duration_limit"; break;
        }
        auto frame = source.next();
        summary.source = source.stats();
        if (progress) progress(summary);
        if (cancel) { summary.stop_reason = "cancelled"; break; }
        if (!frame) {
            if (source.finished()) { summary.stop_reason = source.stats().stop_reason; break; }
            continue;
        }
        const bool reset=session!=frame->session || frame->arrival_ms-previous_arrival>config.tracking_gap_ms ||
            frame->image.size()!=previous_size;
        if (reset) ++summary.tracking_epochs;
        previous_size=frame->image.size();
        session = frame->session; previous_arrival = frame->arrival_ms;
        const auto analysis_started = Clock::now();
        const double age = std::chrono::duration<double, std::milli>(analysis_started - frame->arrived).count();
        summary.max_decode_age_at_analysis_ms = std::max(summary.max_decode_age_at_analysis_ms, age);
        auto result = analyze(*frame,reset);
        const double analysis_ms = std::chrono::duration<double, std::milli>(Clock::now() - analysis_started).count();
        if (cancel) { summary.stop_reason = "cancelled"; break; }
        total_analysis += analysis_ms;
        ++summary.processed_frames;
        summary.mean_analysis_ms = total_analysis / summary.processed_frames;
        summary.source = source.stats();
        sink(*frame, result, summary, age, analysis_ms);
        if (progress) progress(summary);
    }
    source.stop(); summary.source = source.stats();
    if (progress) progress(summary);
    return summary;
}

LiveSummary process_stream(const std::string& url, const std::filesystem::path& output,
    IDetector& detector, const VideoConfig& tracking, const LiveConfig& config, LiveCaptureFactory factory) {
    namespace fs = std::filesystem;
    if (tracking.use_appearance)
        throw std::invalid_argument("Appearance tracking requires local video; live appearance is not supported");
    validate_rtsp_url(url); validate_live_config(config);
    (void)make_tracker(tracking);
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    fs::create_directories(output);
    std::ofstream rows(output / "tracks.csv"), frames(output / "frames.csv");
    rows << "output_frame,source_sequence,source_session,tracking_epoch,arrival_ms,track_id,label,confidence,x1,y1,x2,y2\n";
    frames << "output_frame,source_sequence,source_session,tracking_epoch,arrival_ms,decode_age_ms,analysis_ms\n";
    if (!rows || !frames) throw std::runtime_error("Cannot open live reports");
    cv::VideoWriter writer;
    cv::Size output_size;
    std::atomic_bool cancel{false};
    int epoch = 0;
    auto summary = analyze_stream(url, detector, tracking, config, cancel,
      [&](const LiveFrame& frame, const AnalysisResult& result, const LiveSummary& current, double age, double analysis_ms) {
        if (!writer.isOpened()) {
            output_size = frame.image.size();
            if (output_size.width % 2 || output_size.height % 2)
                throw std::runtime_error("MJPEG output requires even frame dimensions");
            writer.open(utf8(output / "tracked.avi"), cv::CAP_OPENCV_MJPEG,
                cv::VideoWriter::fourcc('M','J','P','G'), config.output_fps, output_size);
            if (!writer.isOpened()) throw std::runtime_error("Cannot open live video writer");
        }
        if (frame.image.size() != output_size) throw std::runtime_error("Live recording frame format changed");
        if (epoch != current.tracking_epochs) {
            epoch = current.tracking_epochs;
            std::cout << "live session=" << frame.session << " tracking_epoch=" << epoch << std::endl;
        }
        const int index = current.processed_frames - 1;
        std::vector<Detection> display;
        for (const auto& track : result.tracks) {
            display.push_back({track.bbox, "E" + std::to_string(epoch) + "/ID " +
                std::to_string(track.track_id) + " " + track.label, track.score, {}, {}});
            std::string label;
            for (char c : track.label) { if (c == '"') label += '"'; label += c; }
            rows << index << ',' << frame.sequence << ',' << frame.session << ',' <<
                epoch << ',' << frame.arrival_ms << ',' << track.track_id << ",\"" << label << "\"," <<
                track.score << ',' << track.bbox.x1 << ',' << track.bbox.y1 << ',' << track.bbox.x2 << ',' << track.bbox.y2 << '\n';
        }
        frames << index << ',' << frame.sequence << ',' << frame.session << ',' << epoch << ',' <<
            frame.arrival_ms << ',' << age << ',' << analysis_ms << '\n';
        auto drawn = annotate(frame.image, display);
        if (index == 0) save_image(output / "preview.jpg", drawn);
        save_image(output / "latest.jpg", drawn);
        writer.write(drawn);
        if (current.processed_frames % 10 == 0) {
            const auto& stats = current.source;
            std::cout << "live frames=" << current.processed_frames << " decoded=" << stats.decoded_frames <<
                " dropped=" << stats.dropped_overflow + stats.dropped_stale + stats.dropped_disconnect << std::endl;
        }
        if (!rows || !frames) throw std::runtime_error("Cannot write live reports");
      }, {}, std::move(factory));
    writer.release(); rows.flush(); frames.flush();
    if (!rows || !frames) throw std::runtime_error("Cannot finalize live reports");
    cv::FileStorage report("summary.json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    report << "processed_frames" << summary.processed_frames << "tracking_epochs" << summary.tracking_epochs;
    report << "stop_reason" << summary.stop_reason << "timestamp_basis" << "steady-clock decode arrival; not camera PTS";
    report << "mean_analysis_ms" << summary.mean_analysis_ms << "max_decode_age_at_analysis_ms" << summary.max_decode_age_at_analysis_ms;
    report << "output_fps" << config.output_fps << "playback_basis" << "analyzed frames only; not wall-clock duration";
    const auto& s = summary.source;
    report << "connection_attempts" << static_cast<double>(s.connection_attempts) << "sessions" << static_cast<double>(s.sessions);
    report << "read_failures" << static_cast<double>(s.read_failures) << "decoded_frames" << static_cast<double>(s.decoded_frames);
    report << "delivered_frames" << static_cast<double>(s.delivered_frames) << "dropped_overflow" << static_cast<double>(s.dropped_overflow);
    report << "dropped_stale" << static_cast<double>(s.dropped_stale) << "dropped_disconnect_or_stop" << static_cast<double>(s.dropped_disconnect);
    report << "queue_high_watermark" << static_cast<double>(s.queue_high_watermark) << "queue_capacity" << config.queue_capacity;
    report << "source_stop_reason" << s.stop_reason;
    report << "duration_seconds" << config.duration_seconds << "open_timeout_ms" << config.open_timeout_ms;
    report << "read_timeout_ms" << config.read_timeout_ms << "max_frame_age_ms" << config.max_frame_age_ms;
    report << "queue_byte_limit" << static_cast<double>(max_queue_bytes) << "frame_byte_limit" << static_cast<double>(max_frame_bytes);
    std::ofstream json(output / "summary.json"); json << report.releaseAndGetString(); json.flush();
    if (!json) throw std::runtime_error("Cannot write live summary");
    return summary;
}
} // namespace aegisvision::vision
