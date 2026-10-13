#include "aegisvision/tensorrt.hpp"
#include "aegisvision/application_config.hpp"
#include "aegisvision/quality_benchmark.hpp"
#include <chrono>
#include <iostream>
namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_trt_quality BUNDLE CPU_REFERENCE_CONFIG.toml MANIFEST.json NEW_OUTPUT\n"
                "Same config/model hash, letterbox/NMS and quality evaluator as CPU. Five warmup calls.\n"
                "Synchronous detector timing includes H2D, compute, D2H, preprocessing and NMS; no CPU fallback.\n";
            return 0;
        }
        if(args.size()!=5) throw std::invalid_argument("Use --help");
        const auto settings=aegisvision::vision::load_application_settings(args[2]);
        if(settings.mode!=aegisvision::vision::ApplicationMode::Image) throw std::invalid_argument("Image config required");
        cv::setNumThreads(1);
        const auto start=std::chrono::steady_clock::now();
        aegisvision::vision::TensorRtYoloDetector detector(args[1],settings.detector);
        auto info=detector.metadata();
        if(info.at("model_sha256")!=aegisvision::evaluation::quality_file_sha256(settings.detector_model))
            throw std::invalid_argument("Engine source differs from CPU reference ONNX");
        aegisvision::evaluation::QualityRunConfig config;
        config.provenance={{"model_sha256",info.at("model_sha256")},
            {"config_sha256",aegisvision::evaluation::quality_file_sha256(args[2])},
            {"engine","tensorrt"},{"device","cuda:0"},{"precision",info.at("precision")},
            {"confidence_threshold",settings.detector.confidence_threshold},{"nms_iou_threshold",settings.detector.nms_iou_threshold},
            {"input_size",settings.detector.input_size},{"max_detections",settings.detector.max_detections},
            {"model_load_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()},
            {"tensorrt",info}};
        (void)aegisvision::evaluation::run_quality_benchmark(args[3],args[4],detector,config);
        std::cout << "Native TensorRT benchmark completed: " << args[4].string() << '\n';return 0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
