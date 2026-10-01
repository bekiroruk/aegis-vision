#include "aegisvision/live_session.hpp"
#include "aegisvision/vision.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <set>

namespace aegisvision {
using Json = nlohmann::json;
using namespace std::chrono_literals;
LiveSessions::LiveSessions(LiveServiceConfig config, LiveDetectorFactory detector, vision::LiveCaptureFactory capture,
    vision::LiveArchive::Callback archived)
    : config_(std::move(config)), detector_(std::move(detector)), capture_(std::move(capture)), archived_(std::move(archived)) {
    vision::validate_live_config(config_.stream);
    if (config_.sources.size() > 16 || (!config_.sources.empty() && !detector_))
        throw std::invalid_argument("Live service requires at most 16 presets and an isolated detector factory");
    std::set<std::string> ids;
    for (const auto& source : config_.sources) {
        if (source.id.empty() || source.id.size() > 64 || source.id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos ||
            source.label.empty() || source.label.size() > 128 || !ids.insert(source.id).second)
            throw std::invalid_argument("Invalid or duplicate live preset");
        vision::validate_rtsp_url(source.url);
    }
}
LiveSessions::~LiveSessions() { shutdown(); }
Json LiveSessions::sources() const {
    auto entries = Json::array();
    for (const auto& s : config_.sources) entries.push_back({{"id", s.id}, {"label", s.label}});
    return {{"sources", entries}, {"max_active", 1}, {"duration_seconds", config_.stream.duration_seconds},
        {"preview_max_age_ms", 2000}, {"persistent", false}, {"archive_available", config_.archive.enabled},
        {"archive_config", {{"segment_seconds",config_.archive.segment_seconds},
            {"max_segments_per_session",config_.archive.max_segments_per_session},
            {"max_total_segments",config_.archive.max_total_segments},{"max_bytes",config_.archive.max_bytes}}}};
}
LivePreview LiveSessions::preview_locked() const {
    auto value = preview_;
    value.decode_age_ms = std::chrono::duration<double, std::milli>(Clock::now() - arrived_).count();
    if (!active_ || cancel_ || summary_.source.connection_state != "live" ||
        value.source_session != summary_.source.sessions || value.decode_age_ms > 2000)
        value.jpeg.reset();
    return value;
}
Json LiveSessions::snapshot_locked() const {
    const auto& s = summary_.source;
    const auto picture = preview_locked();
    return {{"id", id_}, {"source_id", source_id_}, {"state", state_}, {"active", active_},
        {"connection_state", s.connection_state}, {"cancel_requested", cancel_.load()}, {"error", error_},
        {"processed_frames", summary_.processed_frames}, {"decoded_frames", s.decoded_frames},
        {"sessions", s.sessions}, {"connection_attempts", s.connection_attempts}, {"read_failures", s.read_failures},
        {"dropped_frames", s.dropped_overflow + s.dropped_stale + s.dropped_disconnect},
        {"queue_high_watermark", s.queue_high_watermark}, {"queue_capacity", config_.stream.queue_capacity},
        {"tracking_epochs", summary_.tracking_epochs}, {"mean_analysis_ms", summary_.mean_analysis_ms},
        {"stop_reason", summary_.stop_reason}, {"duration_seconds", config_.stream.duration_seconds},
        {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>((active_ ? Clock::now() : finished_) - started_).count()},
        {"has_preview", static_cast<bool>(picture.jpeg)}, {"preview_sequence", picture.jpeg ? picture.sequence : 0},
        {"decode_age_ms", picture.jpeg ? Json(picture.decode_age_ms) : Json(nullptr)},
        {"archive",archive_},
        {"timestamp_basis", "steady-clock decode arrival; not camera PTS"}};
}
Json LiveSessions::current() const {
    std::lock_guard lock(state_mutex_);
    return id_.empty() ? Json(nullptr) : snapshot_locked();
}
Json LiveSessions::start(const std::string& source_id, bool archive) {
    if (archive && !config_.archive.enabled) throw std::invalid_argument("Live archiving is not enabled on this server");
    const auto preset = std::find_if(config_.sources.begin(), config_.sources.end(),
        [&](const auto& item) { return item.id == source_id; });
    if (preset == config_.sources.end()) throw std::invalid_argument("Unknown configured live source");
    std::lock_guard operation(operation_mutex_);
    {
        std::lock_guard lock(state_mutex_);
        if (closed_) throw std::runtime_error("Live service is shutting down");
        if (active_) throw LiveSessionBusy();
    }
    if (worker_.joinable()) worker_.join();
    std::lock_guard lock(state_mutex_);
    cancel_ = false; active_ = true; summary_ = {}; preview_ = {}; error_.clear();
    started_ = Clock::now(); arrived_ = started_; finished_ = started_;
    id_ = "live-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" + std::to_string(next_id_++);
    source_id_ = source_id; state_ = "starting";
    archive_ = {{"enabled",archive},{"state",archive ? "starting" : "disabled"},{"error",""},
        {"closed_segments",0},{"index_queue_failures",0}};
    try { worker_ = std::thread([this, source = *preset, archive] { run(source, archive); }); }
    catch (...) { active_ = false; state_ = "failed"; error_ = "Cannot start live worker"; throw; }
    return snapshot_locked();
}
bool LiveSessions::request_stop(const std::string& id) {
    std::lock_guard lock(state_mutex_);
    if (id_.empty() || id != id_) return false;
    if (active_) { cancel_ = true; state_ = "stopping"; preview_.jpeg.reset(); }
    return true;
}
bool LiveSessions::contains(const std::string& id) const {
    std::lock_guard lock(state_mutex_); return !id_.empty() && id == id_;
}
LivePreview LiveSessions::preview(const std::string& id) const {
    std::lock_guard lock(state_mutex_);
    return id != id_ || id_.empty() ? LivePreview{} : preview_locked();
}
void LiveSessions::shutdown() {
    std::lock_guard operation(operation_mutex_);
    { std::lock_guard lock(state_mutex_); closed_ = true; cancel_ = true; preview_.jpeg.reset(); }
    if (worker_.joinable()) worker_.join();
}
void LiveSessions::run(LivePreset preset, bool archive) {
    auto archive_config = config_.archive; archive_config.enabled = archive;
    // The recorder contains its own I/O/admission failures. Never hold snapshot
    // mutexes while writing files or admitting a finite SQLite job.
    std::unique_ptr<vision::LiveArchive> recorder;
    const auto finish_archive = [&] {
        if (!recorder) return;
        recorder->finish();
        std::lock_guard lock(state_mutex_); archive_ = recorder->snapshot();
    };
    try {
        recorder = std::make_unique<vision::LiveArchive>(archive_config,id_,preset.id,archived_);
        auto detector = detector_(); // This model belongs only to this worker.
        if (!detector) throw std::runtime_error("Null live detector");
        {
            std::lock_guard lock(state_mutex_);
            if (!cancel_) state_ = "running";
        }
        const auto summary = vision::analyze_stream(preset.url, *detector, config_.tracking, config_.stream, cancel_,
            [this,&recorder](const vision::LiveFrame& frame, const AnalysisResult& result, const vision::LiveSummary& current, double, double) {
                if (!cancel_) recorder->accept(frame,current.tracking_epochs);
                std::vector<Detection> boxes;
                for (const auto& t : result.tracks) boxes.push_back({t.bbox,
                    "E" + std::to_string(current.tracking_epochs) + "/ID " + std::to_string(t.track_id) + " " + t.label,
                    t.score, {}, {}});
                auto image = vision::annotate(frame.image, boxes);
                const double scale = std::min({1.0, 960.0 / image.cols, 720.0 / image.rows});
                if (scale < 1) cv::resize(image, image, {}, scale, scale, cv::INTER_AREA);
                auto bytes = std::make_shared<std::vector<unsigned char>>();
                if (!cv::imencode(".jpg", image, *bytes, {cv::IMWRITE_JPEG_QUALITY, 80}) || bytes->size() > 2 * 1024 * 1024)
                    throw std::runtime_error("Cannot encode bounded live preview");
                std::lock_guard lock(state_mutex_);
                summary_ = current;
                archive_ = recorder->snapshot();
                if (cancel_ || current.source.connection_state != "live" || current.source.sessions != frame.session) return;
                arrived_ = frame.arrived;
                preview_ = {std::move(bytes), frame.sequence, frame.session, current.tracking_epochs, 0};
            },
            [this](const vision::LiveSummary& summary) {
                std::lock_guard lock(state_mutex_); summary_ = summary;
                if (summary.source.connection_state != "live" || preview_.source_session != summary.source.sessions)
                    preview_.jpeg.reset();
            }, capture_);
        finish_archive();
        std::lock_guard lock(state_mutex_);
        summary_ = summary; active_ = false; preview_.jpeg.reset(); finished_ = Clock::now();
        state_ = cancel_ || summary.stop_reason == "cancelled" ? "stopped" :
            summary.stop_reason == "duration_limit" && summary.processed_frames > 0 ? "completed" : "failed";
        if (state_ == "failed") error_ = "Live source stopped: " + summary.stop_reason;
    } catch (const std::exception&) {
        finish_archive();
        // Do not expose decoder/model exception strings or source URLs to HTTP clients.
        std::lock_guard lock(state_mutex_);
        active_ = false; preview_.jpeg.reset(); summary_.stop_reason = "worker_error"; finished_ = Clock::now();
        state_ = cancel_ ? "stopped" : "failed"; error_ = cancel_ ? "" : "Live analysis failed; check local model/source setup";
    } catch (...) {
        finish_archive();
        std::lock_guard lock(state_mutex_); active_ = false; preview_.jpeg.reset(); state_ = "failed"; finished_ = Clock::now();
        error_ = "Live worker failed"; summary_.stop_reason = "worker_error";
    }
}
}
