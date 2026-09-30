#include "aegisvision/video.hpp"
#include "aegisvision/yolo.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>

namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if ((args.size() == 2 && args[1] == "--help") || args.size() < 4) {
            std::cout << "Usage: aegisvision_video MODEL.onnx INPUT_VIDEO NEW_OUTPUT_DIR [MAX_FRAMES] [--tracker iou|two-stage]\n"
                "CPU YOLOv8; default: IoU tracker. Two-stage adds low-score recovery and motion prediction.\n"
                "Writes tracked.avi (MJPEG), preview.jpg, tracks.csv and summary.json.\n"
                "MAX_FRAMES: positive integer, omitted = entire file. Output directory must be empty.\n";
            return args.size() == 2 && args[1] == "--help" ? 0 : 2;
        }
        aegisvision::vision::VideoConfig config;
        bool saw_limit = false, saw_tracker = false;
        for (std::size_t i = 4; i < args.size(); ++i) {
            if (args[i] == "--tracker") {
                if (saw_tracker || i + 1 >= args.size()) throw std::invalid_argument("Expected one --tracker iou|two-stage");
                saw_tracker = true;
                const auto& mode = args[++i];
                if (mode == "two-stage") config.tracker_mode = aegisvision::vision::TrackerMode::TwoStage;
                else if (mode != "iou") throw std::invalid_argument("Unknown tracker; use iou or two-stage");
                continue;
            }
            if (saw_limit) throw std::invalid_argument("Unexpected extra argument");
            saw_limit = true;
            const auto text = args[i].string();
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), config.max_frames);
            if (error != std::errc{} || end != text.data() + text.size() || config.max_frames < 1) {
                throw std::invalid_argument("MAX_FRAMES must be a positive integer");
            }
        }
        aegisvision::vision::YoloConfig yolo_config;
        yolo_config.confidence_threshold = config.tracker_mode == aegisvision::vision::TrackerMode::TwoStage
            ? config.low_confidence : config.high_confidence;
        aegisvision::vision::YoloDetector detector(args[1], yolo_config);
        const auto summary = aegisvision::vision::process_video(args[2], args[3], detector, config);
        std::cout << "frames=" << summary.processed_frames << " unique_track_ids=" << summary.unique_track_ids
            << " processing_fps=" << summary.processing_fps << " mean_analysis_ms=" << summary.mean_analysis_ms
            << " low_confidence_matches=" << summary.tracking_stats.low_confidence_matches << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
