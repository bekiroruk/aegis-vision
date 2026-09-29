#include "aegisvision/video.hpp"
#include "aegisvision/yolo.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/pipeline.hpp"
#include <opencv2/videoio.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
std::string utf8(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return {text.begin(), text.end()};
}
std::string csv_string(const std::string& text) {
    std::string escaped = "\"";
    for (char c : text) { if (c == '"') escaped += '"'; escaped += c; }
    return escaped + '"';
}
}

VideoSummary process_video(const std::filesystem::path& input, const std::filesystem::path& output,
    IDetector& detector, const VideoConfig& config) {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;
    if (config.max_frames < 0 || !std::isfinite(config.fallback_fps) ||
        config.fallback_fps <= 0 || config.fallback_fps > 1000) {
        throw std::invalid_argument("Invalid video configuration");
    }
    IoUTracker tracker(config.tracking_iou, config.max_missed_frames);
    if (!fs::is_regular_file(input)) throw std::runtime_error("Input must be an existing local video file");
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output))) {
        throw std::runtime_error("Output directory must be new or empty");
    }
    cv::VideoCapture capture(utf8(input));
    if (!capture.isOpened()) throw std::runtime_error("Cannot open input video; check file and codec support");
    cv::Mat image;
    if (!capture.read(image) || image.empty()) throw std::runtime_error("Video has no decodable frames");
    const auto size = image.size();
    // MJPEG containers/codecs may silently truncate odd dimensions. Make the contract explicit.
    if (size.width % 2 || size.height % 2) throw std::runtime_error("Video width and height must be even");
    const double reported_fps = capture.get(cv::CAP_PROP_FPS);
    const bool fallback = !std::isfinite(reported_fps) || reported_fps <= 0 || reported_fps > 1000;
    const double fps = fallback ? config.fallback_fps : reported_fps;
    fs::create_directories(output);
    cv::VideoWriter writer(utf8(output / "tracked.avi"), cv::CAP_OPENCV_MJPEG,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    if (!writer.isOpened()) throw std::runtime_error("Cannot open MJPEG output video writer");
    std::ofstream rows(output / "tracks.csv");
    rows << "frame_index,timestamp_ms,track_id,label,confidence,x1,y1,x2,y2,age\n" << std::setprecision(9);
    if (!rows) throw std::runtime_error("Cannot open tracks.csv");
    PipelineConfig pipeline_config;
    pipeline_config.enable_embeddings = false;
    pipeline_config.index_embeddings = false;
    AnalysisPipeline pipeline(pipeline_config, detector, &tracker, nullptr, nullptr, nullptr);
    VideoSummary summary;
    summary.source_fps = fps;
    std::set<std::uint64_t> ids;
    double total_analysis_ms = 0;
    const auto processing_start = Clock::now();
    while (true) {
        if (image.size() != size || image.type() != CV_8UC3) {
            throw std::runtime_error("Video frame format changed during decoding");
        }
        // Constant-frame-rate estimate, not a container presentation timestamp.
        const auto timestamp = static_cast<std::int64_t>(std::llround(summary.processed_frames * 1000.0 / fps));
        const auto analysis_start = Clock::now();
        const auto result = pipeline.analyze(image_frame(image, std::to_string(summary.processed_frames),
            "video-1", timestamp));
        total_analysis_ms += std::chrono::duration<double, std::milli>(Clock::now() - analysis_start).count();
        std::vector<Detection> display;
        for (const auto& track : result.tracks) {
            ids.insert(track.track_id);
            display.push_back({track.bbox, "ID " + std::to_string(track.track_id) + " " + track.label,
                track.score, {}, {}});
            const auto& b = track.bbox;
            rows << summary.processed_frames << ',' << timestamp << ',' << track.track_id << ','
                << csv_string(track.label) << ',' << track.score << ',' << b.x1 << ',' << b.y1 << ','
                << b.x2 << ',' << b.y2 << ',' << track.age << '\n';
        }
        auto drawn = annotate(image, display);
        if (summary.processed_frames == 0) save_image(output / "preview.jpg", drawn);
        writer.write(drawn);
        ++summary.processed_frames;
        if (config.max_frames && summary.processed_frames >= config.max_frames) {
            summary.stop_reason = "frame_limit";
            break;
        }
        if (!capture.read(image) || image.empty()) {
            summary.stop_reason = "end_of_stream_or_decode_stop";
            break;
        }
    }
    writer.release();
    rows.flush();
    if (!rows || !fs::exists(output / "tracked.avi") || fs::file_size(output / "tracked.avi") == 0) {
        throw std::runtime_error("Failed to finalize output files");
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - processing_start).count();
    summary.processing_fps = seconds > 0 ? summary.processed_frames / seconds : 0;
    summary.mean_analysis_ms = total_analysis_ms / summary.processed_frames;
    summary.unique_track_ids = ids.size();
    cv::FileStorage report("summary.json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
        cv::FileStorage::FORMAT_JSON);
    report << "processed_frames" << summary.processed_frames << "unique_track_ids" << static_cast<double>(ids.size());
    report << "source_fps" << fps << "used_fallback_fps" << static_cast<int>(fallback);
    report << "processing_fps" << summary.processing_fps << "mean_analysis_ms" << summary.mean_analysis_ms;
    report << "timestamp_basis" << "frame_index / source_fps (CFR estimate)";
    report << "stop_reason" << summary.stop_reason << "tracker" << "class-aware greedy IoU";
    report << "tracking_iou" << config.tracking_iou << "max_missed_frames" << static_cast<double>(config.max_missed_frames);
    std::ofstream json(output / "summary.json");
    json << report.releaseAndGetString();
    json.flush();
    if (!json) throw std::runtime_error("Cannot write video summary");
    return summary;
}
} // namespace aegisvision::vision
