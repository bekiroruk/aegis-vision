#include "aegisvision/video.hpp"
#include "aegisvision/yolo.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>

namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if ((args.size() == 2 && args[1] == "--help") || args.size() < 4 || args.size() > 5) {
            std::cout << "Usage: aegisvision_video MODEL.onnx INPUT_VIDEO NEW_OUTPUT_DIR [MAX_FRAMES]\n"
                "CPU YOLOv8 + class-aware IoU tracking; processes every frame.\n"
                "Writes tracked.avi (MJPEG), preview.jpg, tracks.csv and summary.json.\n"
                "MAX_FRAMES: positive integer, omitted = entire file. Output directory must be empty.\n";
            return args.size() == 2 && args[1] == "--help" ? 0 : 2;
        }
        aegisvision::vision::VideoConfig config;
        if (args.size() == 5) {
            const auto text = args[4].string();
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), config.max_frames);
            if (error != std::errc{} || end != text.data() + text.size() || config.max_frames < 1) {
                throw std::invalid_argument("MAX_FRAMES must be a positive integer");
            }
        }
        aegisvision::vision::YoloDetector detector(args[1]);
        const auto summary = aegisvision::vision::process_video(args[2], args[3], detector, config);
        std::cout << "frames=" << summary.processed_frames << " unique_track_ids=" << summary.unique_track_ids
            << " processing_fps=" << summary.processing_fps << " mean_analysis_ms=" << summary.mean_analysis_ms << '\n';
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
