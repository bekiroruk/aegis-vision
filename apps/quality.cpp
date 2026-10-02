#include "aegisvision/application_config.hpp"
#include "aegisvision/quality_benchmark.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace {
int run(const std::vector<std::filesystem::path> &args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "Usage: aegisvision_quality CONFIG.toml MANIFEST.json NEW_OUTPUT_DIR "
                         "[WARMUP_ITERATIONS]\n"
                      << "CPU/FP32 YOLO quality baseline; images=COCO-style AP, "
                         "video=IoU/two-stage CLEAR+IDF1.\n"
                      << "One OpenCV thread. Default five unmeasured warmup calls; same detector "
                         "outputs for both trackers.\n";
            return 0;
        }
        if (args.size() != 4 && args.size() != 5)
            throw std::invalid_argument("Use --help for quality benchmark arguments");
        const auto settings = aegisvision::vision::load_application_settings(args[1]);
        if (settings.mode != aegisvision::vision::ApplicationMode::Image)
            throw std::invalid_argument("Quality CLI requires image detector configuration");
        aegisvision::evaluation::QualityRunConfig config;
        if (args.size() == 5) {
            const auto s = args[4].string();
            const auto [end, ec] =
                std::from_chars(s.data(), s.data() + s.size(), config.warmup_iterations);
            if (ec != std::errc{} || end != s.data() + s.size() || config.warmup_iterations < 0 ||
                config.warmup_iterations > 30)
                throw std::invalid_argument("Warmup iterations must be 0..30");
        }
        cv::setNumThreads(1);
        const auto start = std::chrono::steady_clock::now();
        auto detector = aegisvision::vision::make_configured_detector(settings);
        const auto model_load_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        config.provenance = {
            {"model_sha256", aegisvision::evaluation::quality_file_sha256(settings.detector_model)},
            {"config_sha256", aegisvision::evaluation::quality_file_sha256(args[1])},
            {"device", "cpu"},
            {"precision", "fp32"},
            {"engine", "opencv-dnn"},
            {"confidence_threshold", settings.detector.confidence_threshold},
            {"nms_iou_threshold", settings.detector.nms_iou_threshold},
            {"input_size", settings.detector.input_size},
            {"max_detections", settings.detector.max_detections},
            {"model_load_ms", model_load_ms}};
        const auto report =
            aegisvision::evaluation::run_quality_benchmark(args[2], args[3], *detector, config);
        std::cout << "Quality baseline completed: " << report.at("dataset")
                  << "; report=" << (args[3] / "report.json").string() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
} // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
