#include "aegisvision/application_config.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void save(const fs::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary);
    stream << text;
    if (!stream) throw std::runtime_error("Fixture write failed");
}
template<class F> void rejects(F action, const char* why) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(why);
}
}

int main() {
    const auto root = fs::temp_directory_path() / ("aegis-config-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root / "models");
        save(root / "models/detector.onnx", "placeholder");
        for (auto name : {"manifest.json", "vision.onnx", "text.onnx", "tokenizer.json"})
            save(root / "models" / name, "placeholder");
        const std::string detector = "[detector]\nbackend = \"yolo_onnx\"\nmodel = \"models/detector.onnx\"\n"
            "confidence_threshold = 0.4\ninput_size = 640\n";
        const std::string video = "version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "[tracking]\nbackend = \"two-stage\"\nlow_confidence = 0.1\nnew_track_confidence = 0.6\n"
            "max_missed_frames = 2\n[video]\nmax_frames = 5\n";
        save(root / "video.toml", video);
        auto config = aegisvision::vision::load_application_settings(root / "video.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Video &&
            config.detector_model == fs::weakly_canonical(root / "models/detector.onnx"), "Model path resolution failed");
        require(config.video.tracker_mode == aegisvision::vision::TrackerMode::TwoStage &&
            config.video.high_confidence == .4F && config.video.new_track_confidence == .6F &&
            config.detector.confidence_threshold == .1F && config.video.max_frames == 5,
            "Video settings did not reach adapters");
        save(root / "image.toml", "version = 1\n[pipeline]\nmode = \"image\"\n" + detector);
        config = aegisvision::vision::load_application_settings(root / "image.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Image &&
            config.detector.confidence_threshold == .4F, "Image setting lost");
        const std::string search = "version = 1\n[pipeline]\nmode = \"search\"\n"
            "[embedding]\nbackend = \"clip_onnx\"\nbundle = \"models\"\ndimension = 512\n"
            "[vector_store]\nbackend = \"qdrant\"\ncollection = \"test_config\"\ndimension = 512\n";
        save(root / "search.toml", search);
        config = aegisvision::vision::load_application_settings(root / "search.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Search &&
            config.qdrant.collection == "test_config", "Search setting lost");
        rejects([&] { (void)aegisvision::vision::make_configured_detector(config); }, "Search created a detector");
        const auto bad = [&](const std::string& text) {
            save(root / "bad.toml", text);
            rejects([&] { (void)aegisvision::vision::load_application_settings(root / "bad.toml"); },
                "Invalid configuration accepted");
        };
        bad(video + "[unexpected]\nvalue = 1\n");
        bad("version = 1.0\n[pipeline]\nmode = \"image\"\n" + detector);
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "unknown = 1\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "input_size = 641\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "nms_iou_threshold = true\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "[tracking]\nbackend = \"iou\"\n");
        bad("version = 1\n[pipeline]\nmode = \"search\"\n"
            "[embedding]\nbackend = \"clip_onnx\"\nbundle = \"models\"\ndimension = 768\n"
            "[vector_store]\nbackend = \"qdrant\"\ncollection = \"test_config\"\ndimension = 512\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n[detector]\n"
            "backend = \"yolo_onnx\"\nmodel = \"missing.onnx\"\n");
        bad(video + "[runtime]\ndevice = \"cuda\"\n");
        bad("version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "[tracking]\nbackend = \"two-stage\"\nlow_confidence = 0.5\nnew_track_confidence = 0.6\n");
        fs::remove_all(root); // Unique scratch directory created above.
        std::cout << "TOML modes, relative paths, thresholds, invalid schemas passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << " (fixtures: " << root << ")\n";
        return 1;
    }
}
