#include "aegisvision/application_config.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
void require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
void save(const fs::path &path, const std::string &text) {
    std::ofstream stream(path, std::ios::binary);
    stream << text;
    if (!stream)
        throw std::runtime_error("Fixture write failed");
}
template <class F> void rejects(F action, const char *why) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(why);
}
} // namespace

int main() {
    const auto root = fs::temp_directory_path() /
                      ("aegis-config-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root / "models");
        save(root / "models/detector.onnx", "placeholder");
        for (auto name : {"manifest.json", "vision.onnx", "text.onnx", "tokenizer.json"})
            save(root / "models" / name, "placeholder");
        fs::create_directories(root / "models/reid");
        fs::create_directories(root / "models/reid-missing-model");
        fs::create_directories(root / "models/reid-missing-manifest");
        save(root / "models/reid/manifest.json", "placeholder");
        save(root / "models/reid/model.onnx", "placeholder");
        save(root / "models/reid-missing-model/manifest.json", "placeholder");
        save(root / "models/reid-missing-manifest/model.onnx", "placeholder");
        const std::string detector =
            "[detector]\nbackend = \"yolo_onnx\"\nmodel = \"models/detector.onnx\"\n"
            "confidence_threshold = 0.4\ninput_size = 640\n";
        const std::string video = "version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
                                  "[tracking]\nbackend = \"two-stage\"\nlow_confidence = "
                                  "0.1\nnew_track_confidence = 0.6\n"
                                  "max_missed_frames = 2\n[video]\nmax_frames = 5\n";
        save(root / "video.toml", video);
        auto config = aegisvision::vision::load_application_settings(root / "video.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Video &&
                    config.detector_model == fs::weakly_canonical(root / "models/detector.onnx"),
                "Model path resolution failed");
        require(config.video.tracker_mode == aegisvision::vision::TrackerMode::TwoStage &&
                    config.video.high_confidence == .4F &&
                    config.video.new_track_confidence == .6F &&
                    config.detector.confidence_threshold == .1F && config.video.max_frames == 5,
                "Video settings did not reach adapters");
        save(root / "image.toml", "version = 1\n[pipeline]\nmode = \"image\"\n" + detector);
        config = aegisvision::vision::load_application_settings(root / "image.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Image &&
                    config.detector.confidence_threshold == .4F,
                "Image setting lost");
        const std::string kalman_base = "version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
                                       "[tracking]\nbackend = \"kalman\"\n";
        const std::string kalman = kalman_base + "mahalanobis_gate = 13.2767\n";
        save(root / "kalman.toml", kalman);
        config = aegisvision::vision::load_application_settings(root / "kalman.toml");
        require(config.video.tracker_mode == aegisvision::vision::TrackerMode::Kalman &&
                    config.detector.confidence_threshold == .1F &&
                    config.video.high_confidence == .4F &&
                    std::abs(config.video.kalman_gating_threshold - 13.2767) < 1e-9,
                "Kalman config did not reach adapters");
        save(root / "kalman-default.toml", kalman_base);
        config = aegisvision::vision::load_application_settings(root / "kalman-default.toml");
        require(config.video.kalman_gate_mode == aegisvision::KalmanGateMode::FullBox &&
                    config.video.kalman_gating_threshold == aegisvision::kalman_box_gate99,
                "Legacy Kalman default gate changed");
        save(root / "center.toml", kalman_base + "gating_mode = \"center\"\n");
        config = aegisvision::vision::load_application_settings(root / "center.toml");
        require(config.video.kalman_gate_mode == aegisvision::KalmanGateMode::CenterOnly &&
                    config.video.kalman_gating_threshold == aegisvision::kalman_center_gate99,
                "Center gate mode/dimension default did not reach adapters");
        save(root / "center-custom.toml", kalman_base + "gating_mode = \"center\"\nmahalanobis_gate = 7\n");
        config = aegisvision::vision::load_application_settings(root / "center-custom.toml");
        require(config.video.kalman_gating_threshold == 7.0, "Explicit center gate override lost");
        const std::string reid_base = "version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
                                      "[tracking]\nbackend = \"kalman-reid\"\n";
        const std::string appearance = "[appearance]\nbackend = \"osnet_onnx\"\nbundle = \"models/reid\"\n";
        save(root / "reid.toml", reid_base + appearance);
        config = aegisvision::vision::load_application_settings(root / "reid.toml");
        require(config.video.tracker_mode == aegisvision::vision::TrackerMode::Kalman &&
                    config.video.use_appearance && config.video.kalman_gate_mode == aegisvision::KalmanGateMode::CenterOnly &&
                    config.video.kalman_gating_threshold == aegisvision::kalman_center_gate99 &&
                    config.video.max_cosine_distance == 0.20 && config.video.appearance_weight == 0.50 &&
                    config.video.appearance_momentum == 0.90 &&
                    config.reid_bundle == fs::weakly_canonical(root / "models/reid") &&
                    config.detector.confidence_threshold == 0.10F,
                "Appearance config defaults, resolved bundle or low-score detector floor were lost");
        // Placeholder files intentionally are not loadable model/manifest graphs;
        // configuration parsing checks existence without running model loading.
        save(root / "reid-full-box.toml", reid_base + "gating_mode = \"full-box\"\n" + appearance);
        config = aegisvision::vision::load_application_settings(root / "reid-full-box.toml");
        require(config.video.kalman_gate_mode == aegisvision::KalmanGateMode::FullBox &&
                    config.video.kalman_gating_threshold == aegisvision::kalman_box_gate99,
                "Explicit full-box appearance motion gate override was lost");
        save(root / "reid-custom.toml", reid_base + appearance + "max_cosine_distance = 0\nweight = 1\nmomentum = 0\n");
        config = aegisvision::vision::load_application_settings(root / "reid-custom.toml");
        require(config.video.max_cosine_distance == 0 && config.video.appearance_weight == 1 &&
                    config.video.appearance_momentum == 0,
                "Valid appearance parameter boundaries were rejected or ignored");
        const std::string search =
            "version = 1\n[pipeline]\nmode = \"search\"\n"
            "[embedding]\nbackend = \"clip_onnx\"\nbundle = \"models\"\ndimension = 512\n"
            "[vector_store]\nbackend = \"qdrant\"\ncollection = \"test_config\"\ndimension = 512\n";
        save(root / "search.toml", search);
        config = aegisvision::vision::load_application_settings(root / "search.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Search &&
                    config.qdrant.collection == "test_config",
                "Search setting lost");
        rejects([&] { (void)aegisvision::vision::make_configured_detector(config); },
                "Search created a detector");
        const auto bad = [&](const std::string &text) {
            save(root / "bad.toml", text);
            rejects(
                [&] { (void)aegisvision::vision::load_application_settings(root / "bad.toml"); },
                "Invalid configuration accepted");
        };
        const std::string live =
            "version = 1\n[pipeline]\nmode = \"stream\"\n" + detector +
            "[tracking]\nbackend = \"iou\"\n[stream]\nduration_seconds = 12\nqueue_capacity = 2\n";
        save(root / "stream.toml", live);
        config = aegisvision::vision::load_application_settings(root / "stream.toml");
        require(config.mode == aegisvision::vision::ApplicationMode::Stream &&
                    config.live.duration_seconds == 12 && config.live.queue_capacity == 2,
                "Live settings lost");
        bad(live + "unknown = 1\n");
        bad(live + "output_fps = false\n");
        bad(live + "reconnect_initial_ms = 4000\nreconnect_max_ms = 2\n");
        bad(live + "[video]\nmax_frames = 1\n");
        bad(reid_base);
        bad(video + appearance);
        bad(kalman_base + appearance);
        bad(live + appearance);
        bad(search + appearance);
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + appearance);
        bad("version = 1\n[pipeline]\nmode = \"stream\"\n" + detector +
            "[tracking]\nbackend = \"kalman-reid\"\n[stream]\nduration_seconds = 1\n");
        bad(reid_base + "[appearance]\nbackend = \"clip_onnx\"\nbundle = \"models/reid\"\n");
        bad(reid_base + "[appearance]\nbackend = \"osnet_onnx\"\n");
        bad(reid_base + "[appearance]\nbackend = \"osnet_onnx\"\nbundle = \"missing\"\n");
        bad(reid_base + "[appearance]\nbackend = \"osnet_onnx\"\nbundle = \"models/detector.onnx\"\n");
        bad(reid_base + "[appearance]\nbackend = \"osnet_onnx\"\nbundle = \"models/reid-missing-model\"\n");
        bad(reid_base + "[appearance]\nbackend = \"osnet_onnx\"\nbundle = \"models/reid-missing-manifest\"\n");
        for (const auto &parameter : {"max_cosine_distance = -0.01\n", "max_cosine_distance = 1.01\n",
                                     "max_cosine_distance = nan\n", "max_cosine_distance = true\n",
                                     "weight = -0.01\n", "weight = 1.01\n", "weight = inf\n",
                                     "momentum = -0.01\n", "momentum = 1\n", "momentum = nan\n",
                                     "dimension = 512\n", "unknown = 1\n"})
            bad(reid_base + appearance + parameter);
        bad("version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "max_detections = 513\n[tracking]\nbackend = \"kalman-reid\"\n" + appearance);
        bad(video + "[stream]\nduration_seconds = 12\n");
        bad("version = 1\n[pipeline]\nmode = \"stream\"\n" + detector +
            "[tracking]\nbackend = \"kalman\"\n[stream]\nduration_seconds = 1\n");
        bad(kalman + "low_confidence = 0.5\n");
        bad(kalman_base + "mahalanobis_gate = 0\n");
        bad(kalman_base + "mahalanobis_gate = -1\n");
        bad(kalman_base + "mahalanobis_gate = 101\n");
        bad(kalman_base + "mahalanobis_gate = nan\n");
        bad(kalman_base + "gating_mode = \"unknown\"\n");
        bad(kalman_base + "gating_mode = true\n");
        bad(video + "[tracking-extra]\ngating_mode = \"center\"\n");
        bad("version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "[tracking]\nbackend = \"two-stage\"\ngating_mode = \"center\"\n");
        bad("version = 1\n[pipeline]\nmode = \"stream\"\n" + detector +
            "[tracking]\nbackend = \"kalman\"\ngating_mode = \"center\"\n[stream]\nduration_seconds = 1\n");
        bad("version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "[tracking]\nbackend = \"iou\"\nmahalanobis_gate = 13.2767\n");
        bad(video + "[unexpected]\nvalue = 1\n");
        bad("version = 1.0\n[pipeline]\nmode = \"image\"\n" + detector);
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "unknown = 1\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector + "input_size = 641\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector +
            "nms_iou_threshold = true\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n" + detector +
            "[tracking]\nbackend = \"iou\"\n");
        bad("version = 1\n[pipeline]\nmode = \"search\"\n"
            "[embedding]\nbackend = \"clip_onnx\"\nbundle = \"models\"\ndimension = 768\n"
            "[vector_store]\nbackend = \"qdrant\"\ncollection = \"test_config\"\ndimension = "
            "512\n");
        bad("version = 1\n[pipeline]\nmode = \"image\"\n[detector]\n"
            "backend = \"yolo_onnx\"\nmodel = \"missing.onnx\"\n");
        bad(video + "[runtime]\ndevice = \"cuda\"\n");
        bad("version = 1\n[pipeline]\nmode = \"video\"\n" + detector +
            "[tracking]\nbackend = \"two-stage\"\nlow_confidence = 0.5\nnew_track_confidence = "
            "0.6\n");
        fs::remove_all(root); // Unique scratch directory created above.
        std::cout << "TOML modes, relative paths, thresholds, invalid schemas passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << " (fixtures: " << root << ")\n";
        return 1;
    }
}
