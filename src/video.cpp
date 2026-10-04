#include "aegisvision/video.hpp"
#include "aegisvision/pipeline.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <opencv2/videoio.hpp>
#include <set>
#include <stdexcept>
#include <utility>

namespace aegisvision::vision {
namespace {
std::string utf8(const std::filesystem::path &path) {
    const auto text = path.u8string();
    return {text.begin(), text.end()};
}
std::string csv_string(const std::string &text) {
    std::string escaped = "\"";
    for (char c : text) {
        if (c == '"')
            escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

class PersonAppearanceDetector final : public IDetector {
  public:
    PersonAppearanceDetector(IDetector &detector, float threshold)
        : detector_(detector), threshold_(threshold) {}

    std::vector<Detection> detect(const Frame &frame) override {
        auto observations = detector_.detect(frame);
        if (observations.size() > KalmanTracker::max_detections)
            throw std::invalid_argument("Appearance detector exceeded bounded Kalman input capacity");
        std::vector<Detection> persons;
        persons.reserve(observations.size());
        for (auto &detection : observations) {
            if (detection.label != "person")
                continue;
            if (!std::isfinite(detection.score) || detection.score < 0.0F || detection.score > 1.0F)
                throw std::invalid_argument("Appearance person confidence must be finite and in [0,1]");
            if (detection.score < threshold_)
                continue;
            const auto &box = detection.bbox;
            for (const auto coordinate : {box.x1, box.y1, box.x2, box.y2}) {
                if (!std::isfinite(coordinate) || std::abs(static_cast<double>(coordinate)) > 1'000'000.0)
                    throw std::invalid_argument("Appearance person coordinates exceed bounded Kalman input");
            }
            const double width = static_cast<double>(box.x2) - box.x1;
            const double height = static_cast<double>(box.y2) - box.y1;
            if (width <= 0.0 || height <= 0.0 || width > 1'000'000.0 || height > 1'000'000.0)
                throw std::invalid_argument("Appearance person dimensions exceed bounded Kalman input");
            persons.push_back(std::move(detection));
        }
        return persons;
    }

  private:
    IDetector &detector_;
    float threshold_;
};
} // namespace

VideoSummary process_video(const std::filesystem::path &input, const std::filesystem::path &output,
                           IDetector &detector, const VideoConfig &config, IEmbedder *appearance) {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;
    if (config.max_frames < 0 || !std::isfinite(config.fallback_fps) || config.fallback_fps <= 0 ||
        config.fallback_fps > 1000) {
        throw std::invalid_argument("Invalid video configuration");
    }
    if (config.use_appearance && config.tracker_mode != TrackerMode::Kalman)
        throw std::invalid_argument("Appearance video tracking requires the Kalman backend");
    if (config.use_appearance != (appearance != nullptr))
        throw std::invalid_argument("An appearance embedder is required exactly when appearance tracking is enabled");
    std::unique_ptr<ITracker> tracker;
    TwoStageTracker *two_stage = nullptr;
    KalmanTracker *kalman = nullptr;
    if (config.tracker_mode == TrackerMode::IoU) {
        tracker = std::make_unique<IoUTracker>(config.tracking_iou, config.max_missed_frames);
    } else if (config.tracker_mode == TrackerMode::TwoStage) {
        auto instance = std::make_unique<TwoStageTracker>(TwoStageConfig{
            config.low_confidence, config.high_confidence, config.new_track_confidence,
            config.tracking_iou, config.max_missed_frames, true});
        two_stage = instance.get();
        tracker = std::move(instance);
    } else if (config.tracker_mode == TrackerMode::Kalman) {
        auto instance = std::make_unique<KalmanTracker>(KalmanTrackerConfig{
            config.low_confidence, config.high_confidence, config.new_track_confidence,
            config.tracking_iou, config.max_missed_frames, config.kalman_gating_threshold,
            config.kalman_gate_mode, config.use_appearance, 512, config.max_cosine_distance,
            config.appearance_weight, config.appearance_momentum});
        kalman = instance.get();
        tracker = std::move(instance);
    } else {
        throw std::invalid_argument("Unknown tracker mode");
    }
    std::unique_ptr<IDetector> person_detector;
    IDetector *pipeline_detector = &detector;
    if (config.use_appearance) {
        person_detector = std::make_unique<PersonAppearanceDetector>(detector, config.low_confidence);
        pipeline_detector = person_detector.get();
    }
    if (!fs::is_regular_file(input))
        throw std::runtime_error("Input must be an existing local video file");
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output))) {
        throw std::runtime_error("Output directory must be new or empty");
    }
    cv::VideoCapture capture(utf8(input));
    if (!capture.isOpened())
        throw std::runtime_error("Cannot open input video; check file and codec support");
    cv::Mat image;
    if (!capture.read(image) || image.empty())
        throw std::runtime_error("Video has no decodable frames");
    const auto size = image.size();
    // MJPEG containers/codecs may silently truncate odd dimensions. Make the contract explicit.
    if (size.width % 2 || size.height % 2)
        throw std::runtime_error("Video width and height must be even");
    const double reported_fps = capture.get(cv::CAP_PROP_FPS);
    const bool fallback = !std::isfinite(reported_fps) || reported_fps <= 0 || reported_fps > 1000;
    const double fps = fallback ? config.fallback_fps : reported_fps;
    fs::create_directories(output);
    cv::VideoWriter writer(utf8(output / "tracked.avi"), cv::CAP_OPENCV_MJPEG,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    if (!writer.isOpened())
        throw std::runtime_error("Cannot open MJPEG output video writer");
    std::ofstream rows(output / "tracks.csv");
    rows << "frame_index,timestamp_ms,track_id,label,confidence,x1,y1,x2,y2,age\n"
         << std::setprecision(9);
    if (!rows)
        throw std::runtime_error("Cannot open tracks.csv");
    PipelineConfig pipeline_config;
    pipeline_config.enable_embeddings = config.use_appearance;
    pipeline_config.index_embeddings = false;
    AnalysisPipeline pipeline(pipeline_config, *pipeline_detector, tracker.get(), appearance, nullptr, nullptr);
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
        const auto timestamp =
            static_cast<std::int64_t>(std::llround(summary.processed_frames * 1000.0 / fps));
        const auto analysis_start = Clock::now();
        const auto result = pipeline.analyze(
            image_frame(image, std::to_string(summary.processed_frames), "video-1", timestamp));
        total_analysis_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - analysis_start).count();
        std::vector<Detection> display;
        for (const auto &track : result.tracks) {
            ids.insert(track.track_id);
            display.push_back({track.bbox,
                               "ID " + std::to_string(track.track_id) + " " + track.label,
                               track.score,
                               {},
                               {}});
            const auto &b = track.bbox;
            rows << summary.processed_frames << ',' << timestamp << ',' << track.track_id << ','
                 << csv_string(track.label) << ',' << track.score << ',' << b.x1 << ',' << b.y1
                 << ',' << b.x2 << ',' << b.y2 << ',' << track.age << '\n';
        }
        auto drawn = annotate(image, display);
        if (summary.processed_frames == 0)
            save_image(output / "preview.jpg", drawn);
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
    if (!rows || !fs::exists(output / "tracked.avi") ||
        fs::file_size(output / "tracked.avi") == 0) {
        throw std::runtime_error("Failed to finalize output files");
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - processing_start).count();
    summary.processing_fps = seconds > 0 ? summary.processed_frames / seconds : 0;
    summary.mean_analysis_ms = total_analysis_ms / summary.processed_frames;
    summary.unique_track_ids = ids.size();
    if (two_stage)
        summary.tracking_stats = two_stage->stats();
    if (kalman) {
        summary.kalman_stats = kalman->stats();
        const auto &s = kalman->stats();
        summary.tracking_stats = {s.low_confidence_matches, s.reactivations, s.created_tracks,
                                  s.expired_tracks};
    }
    cv::FileStorage report("summary.json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                                               cv::FileStorage::FORMAT_JSON);
    report << "processed_frames" << summary.processed_frames << "unique_track_ids"
           << static_cast<double>(ids.size());
    report << "source_fps" << fps << "used_fallback_fps" << static_cast<int>(fallback);
    report << "processing_fps" << summary.processing_fps << "mean_analysis_ms"
           << summary.mean_analysis_ms;
    report << "timestamp_basis" << "frame_index / source_fps (CFR estimate)";
    report << "stop_reason" << summary.stop_reason << "tracker"
           << (config.use_appearance ? "kalman appearance-assisted active-first Hungarian"
               : kalman      ? "kalman active-first Hungarian"
               : two_stage ? "two-stage linear-motion Hungarian"
                           : "class-aware greedy IoU");
    if (two_stage || kalman) {
        report << "low_confidence_threshold" << config.low_confidence << "high_confidence_threshold"
               << config.high_confidence;
        report << "new_track_confidence_threshold" << config.new_track_confidence;
        report << "low_confidence_matches"
               << static_cast<double>(summary.tracking_stats.low_confidence_matches);
        report << "reactivations" << static_cast<double>(summary.tracking_stats.reactivations);
        report << "created_tracks" << static_cast<double>(summary.tracking_stats.created_tracks);
        report << "expired_tracks" << static_cast<double>(summary.tracking_stats.expired_tracks);
    }
    if (kalman) {
        report << "kalman_gate" << config.kalman_gating_threshold << "motion_dt"
               << "one decoded frame";
        report << "kalman_gating_mode"
               << (config.kalman_gate_mode == KalmanGateMode::CenterOnly ? "center" : "full-box");
        report << "kalman_gating_dimensions"
               << (config.kalman_gate_mode == KalmanGateMode::CenterOnly ? 2 : 4);
        report << "gate_rejections" << static_cast<double>(summary.kalman_stats.gate_rejections);
        report << "numerical_resets" << static_cast<double>(summary.kalman_stats.numerical_resets);
        report << "capacity_rejections" << static_cast<double>(summary.kalman_stats.capacity_rejections);
        if (config.use_appearance) {
            report << "appearance_enabled" << 1 << "appearance_scope" << "person only; local bounded lifecycle";
            report << "appearance_dimension" << 512 << "max_cosine_distance" << config.max_cosine_distance;
            report << "appearance_weight" << config.appearance_weight << "appearance_momentum" << config.appearance_momentum;
            report << "appearance_rejections" << static_cast<double>(summary.kalman_stats.appearance_rejections);
            report << "appearance_matches" << static_cast<double>(summary.kalman_stats.appearance_matches);
            report << "appearance_updates" << static_cast<double>(summary.kalman_stats.appearance_updates);
        }
    }
    report << "tracking_iou" << config.tracking_iou << "max_missed_frames"
           << static_cast<double>(config.max_missed_frames);
    std::ofstream json(output / "summary.json");
    json << report.releaseAndGetString();
    json.flush();
    if (!json)
        throw std::runtime_error("Cannot write video summary");
    return summary;
}
} // namespace aegisvision::vision
