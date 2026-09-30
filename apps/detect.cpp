#include "aegisvision/application_config.hpp"
#include "aegisvision/yolo.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/pipeline.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using namespace aegisvision;

float threshold(const fs::path& value) {
    const auto text = value.string();
    std::size_t end = 0;
    const auto number = std::stof(text, &end);
    if (end != text.size() || !std::isfinite(number) || number <= 0 || number > 1) {
        throw std::invalid_argument("Threshold must be a finite number in (0,1]");
    }
    return number;
}

void help() {
    std::cout << "AegisVision YOLOv8 detector (C++ / OpenCV DNN / CPU)\n"
        "Usage: aegisvision_detect MODEL.onnx IMAGE OUTPUT_DIR [CONFIDENCE [NMS_IOU]]\n"
        "Export contract: float32, batch=1, imgsz=640, COCO 80 classes, nms=False.\n"
        "Defaults: confidence=0.35, NMS IoU=0.45. Writes annotated.png, detections.tsv, report.json.\n";
    std::cout << "Config mode: aegisvision_detect --config CONFIG.toml IMAGE OUTPUT_DIR\n";
}

int run(const std::vector<fs::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") { help(); return 0; }
        const bool configured = args.size() > 1 && args[1] == "--config";
        if (configured ? args.size() != 5 : (args.size() < 4 || args.size() > 6)) { help(); return 2; }
        vision::YoloConfig config;
        fs::path image_path, output_path;
        std::unique_ptr<IDetector> detector;
        vision::ApplicationSettings settings;
        if (configured) {
            settings = vision::load_application_settings(args[2]);
            if (settings.mode != vision::ApplicationMode::Image)
                throw std::invalid_argument("Detection CLI requires pipeline.mode=image");
            config = settings.detector;
            image_path = args[3]; output_path = args[4];
        } else {
            if (args.size() >= 5) config.confidence_threshold = threshold(args[4]);
            if (args.size() == 6) config.nms_iou_threshold = threshold(args[5]);
            image_path = args[2]; output_path = args[3];
        }
        const auto image = vision::load_image(image_path);
        const auto start = std::chrono::steady_clock::now();
        if (configured) detector = vision::make_configured_detector(settings);
        else detector = std::make_unique<vision::YoloDetector>(args[1], config);
        const auto loaded = std::chrono::steady_clock::now();
        PipelineConfig pipeline_config;
        pipeline_config.enable_tracking = false;
        pipeline_config.enable_embeddings = false;
        pipeline_config.index_embeddings = false;
        AnalysisPipeline pipeline(pipeline_config, *detector, nullptr, nullptr, nullptr, nullptr);
        const auto frame = vision::image_frame(image, "image-1", "local-image");
        const auto inference_start = std::chrono::steady_clock::now();
        const auto result = pipeline.analyze(frame);
        const double inference_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inference_start).count();
        const double model_load_ms = std::chrono::duration<double, std::milli>(loaded - start).count();
        vision::save_image(output_path / "annotated.png", vision::annotate(image, result.detections));
        std::ofstream boxes(output_path / "detections.tsv");
        boxes << "# x1 y1 x2 y2 score label\n" << std::setprecision(8);
        cv::FileStorage report("report.json", cv::FileStorage::WRITE |
            cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        report << "backend" << "OpenCV DNN CPU" << "model_family" << "YOLOv8 detect";
        report << "image_width" << image.cols << "image_height" << image.rows;
        report << "confidence_threshold" << config.confidence_threshold << "nms_iou_threshold" << config.nms_iou_threshold;
        report << "model_load_ms" << model_load_ms << "first_inference_pipeline_ms" << inference_ms;
        report << "detections" << "[";
        for (const auto& detection : result.detections) {
            const auto& b = detection.bbox;
            boxes << b.x1 << '\t' << b.y1 << '\t' << b.x2 << '\t' << b.y2 << '\t'
                << detection.score << '\t' << std::quoted(detection.label) << '\n';
            report << "{" << "label" << detection.label << "class_id" << std::stoi(detection.attributes.at("class_id"))
                << "confidence" << detection.score << "xyxy" << "[" << b.x1 << b.y1 << b.x2 << b.y2 << "]" << "}";
            std::cout << detection.label << " confidence=" << detection.score << '\n';
        }
        report << "]";
        std::ofstream json(output_path / "report.json");
        json << report.releaseAndGetString();
        boxes.flush(); json.flush();
        if (!boxes || !json) throw std::runtime_error("Cannot write detection report");
        std::cout << "detections=" << result.detections.size() << " first_inference_pipeline_ms=" << inference_ms << '\n';
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
