#include "aegisvision/quality_benchmark.hpp"
#include "aegisvision/vision.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F action, const char *message) {
    try {
        action();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}
void save(const fs::path &path, const Json &value) {
    std::ofstream stream(path);
    stream << value.dump(2);
    if (!stream)
        throw std::runtime_error("Fixture write failed");
}
Json load(const fs::path &path) {
    std::ifstream stream(path);
    return Json::parse(stream);
}
class Detector final : public aegisvision::IDetector {
  public:
    int calls{};
    std::vector<std::string> ids;
    bool empty{};
    bool invalid_score{};
    std::vector<aegisvision::Detection> detect(const aegisvision::Frame &frame) override {
        ++calls;
        ids.push_back(frame.frame_id);
        require(frame.image && frame.image->width == 32, "Detector must receive owned pixels");
        if (empty)
            return {};
        if (invalid_score)
            return {{{4, 4, 16, 24}, "person", std::numeric_limits<float>::quiet_NaN(), {}, {}}};
        return {{{4, 4, 16, 24}, "person", .9F, {}, {}}, {{0, 0, 4, 4}, "excluded", .99F, {}, {}}};
    }
};
Json image_manifest(const fs::path &root) {
    return {{"version", 1},
            {"kind", "images"},
            {"dataset", "unit-images"},
            {"classes", Json::array({{{"label", "person"}, {"category_id", 1}}})},
            {"images",
             Json::array(
                 {{{"id", "frame-1"},
                   {"coco_image_id", 42},
                   {"path", "image.jpg"},
                   {"sha256", aegisvision::evaluation::quality_file_sha256(root / "image.jpg")},
                   {"width", 32},
                   {"height", 32},
                   {"ground_truth",
                    Json::array(
                        {{{"label", "person"}, {"crowd", false}, {"bbox", {4, 4, 16, 24}}}})}}})}};
}
} // namespace

int main() {
    const auto root = fs::temp_directory_path() /
                      ("aegis-quality-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root);
        aegisvision::vision::save_image(root / "image.jpg",
                                        cv::Mat(32, 32, CV_8UC3, cv::Scalar(30, 60, 90)));
        auto manifest = image_manifest(root);
        save(root / "manifest.json", manifest);
        const auto digest = aegisvision::evaluation::quality_file_sha256(root / "manifest.json");
        Detector detector;
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(root/"manifest.json",root/"image-kalman",detector,{0,Json::object(),true}); },
            "Image manifest silently accepted video-only Kalman comparison");
        const auto report = aegisvision::evaluation::run_quality_benchmark(
            root / "manifest.json", root / "images", detector, {2, Json::object()});
        require(detector.calls == 3 &&
                    detector.ids == std::vector<std::string>{"warmup", "warmup", "frame-1"},
                "Warmup contaminates evaluated samples");
        require(report["detection"]["ap50"] == 1.0 && report["detection"]["predictions"] == 1,
                "Selected image AP or taxonomy filtering wrong");
        require(report["performance"]["detector"]["samples"] == 1,
                "Warmup included in latency samples");
        const auto predictions = load(root / "images/predictions.json");
        require(predictions.size() == 1 && predictions[0]["image_id"] == 42 &&
                    predictions[0]["bbox"] == Json::array({4, 4, 12, 20}),
                "COCO export not xywh/original IDs");
        require(aegisvision::evaluation::quality_file_sha256(root / "manifest.json") == digest,
                "Runner changed its input manifest");
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(root / "manifest.json",
                                                                     root / "images", detector);
            },
            "Existing output overwritten");
        auto bad = manifest;
        bad["images"][0]["sha256"] = "invalid";
        save(root / "bad.json", bad);
        const int calls = detector.calls;
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(root / "bad.json",
                                                                     root / "bad-hash", detector);
            },
            "Bad checksum accepted");
        require(calls == detector.calls, "Inference before checksum validation");
        bad = manifest;
        bad["images"][0]["path"] = "../outside.jpg";
        save(root / "bad.json", bad);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(root / "bad.json",
                                                                     root / "bad-path", detector);
            },
            "Escaped file accepted");
        bad = manifest;
        bad["images"][0]["width"] = 31;
        save(root / "bad.json", bad);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(root / "bad.json",
                                                                     root / "bad-size", detector);
            },
            "Wrong decoded dimensions accepted");
        bad = manifest;
        bad["classes"].push_back({{"label", "other"}, {"category_id", 1}});
        save(root / "bad.json", bad);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(
                    root / "bad.json", root / "bad-category", detector);
            },
            "Duplicate category number accepted");
        bad = manifest;
        bad["images"][0]["ground_truth"] = Json::array();
        save(root / "no-gt.json", bad);
        const auto no_gt = aegisvision::evaluation::run_quality_benchmark(
            root / "no-gt.json", root / "no-gt", detector, {0, Json::object()});
        require(no_gt["detection"]["ap50"].is_null(), "No GT must not fabricate AP");

        const auto video = root / "fixture.avi";
        cv::VideoWriter writer(video.string(), cv::CAP_OPENCV_MJPEG,
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, {32, 32});
        require(writer.isOpened(), "Cannot create deterministic video fixture");
        for (int i = 0; i < 3; ++i)
            writer.write(cv::Mat(32, 32, CV_8UC3, cv::Scalar(i * 30, 60, 90)));
        writer.release();
        Json gt = Json::array();
        for (int i = 1; i <= 3; ++i)
            gt.push_back({{"frame_index", i}, {"id", 19}, {"bbox", {4, 4, 16, 24}}});
        auto movie = Json{{"version", 1},
                          {"kind", "video"},
                          {"dataset", "unit-video"},
                          {"classes", {"person"}},
                          {"video", "fixture.avi"},
                          {"video_sha256", aegisvision::evaluation::quality_file_sha256(video)},
                          {"source_fps", 25},
                          {"frames", 3},
                          {"width", 32},
                          {"height", 32},
                          {"ground_truth", gt}};
        save(root / "movie.json", movie);
        Detector video_detector;
        auto invalid_movie = movie;
        invalid_movie["ground_truth"][0]["id"] = -1;
        save(root / "invalid-id.json", invalid_movie);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(
                    root / "invalid-id.json", root / "negative-id", detector, {0, Json::object()});
            },
            "Negative GT identity accepted");
        invalid_movie["ground_truth"][0]["id"] = 1.5;
        save(root / "invalid-id.json", invalid_movie);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(root / "invalid-id.json",
                                                                     root / "fractional-id",
                                                                     detector, {0, Json::object()});
            },
            "Fractional GT identity accepted");
        Detector invalid;
        invalid.invalid_score = true;
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(
                    root / "movie.json", root / "invalid-score", invalid, {0, Json::object()});
            },
            "NaN score silently filtered by tracker");
        const auto result = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "video", video_detector, {1, Json::object()});
        require(video_detector.calls == 4, "Video must infer once per frame plus warmup");
        for (const auto name : {"iou", "two_stage"}) {
            const auto &metrics = result["tracking"][name];
            require(metrics["tp"] == 3 && metrics["idtp"] == 3 && metrics["idf1"] == 1.0 &&
                        metrics["id_switches"] == 0,
                    "Perfect track metrics wrong");
        }
        const auto frames = load(root / "video/tracking-frames.json");
        require(frames.size() == 3 && frames[2]["frame_index"] == 3 &&
                    frames[0]["ground_truth"][0]["id"] == 19,
                "Video export omitted frames or changed GT");
        require(frames[0]["iou"] == frames[0]["two_stage"],
                "Trackers did not receive same detections");
        std::ifstream mot(root / "video/iou-mot.txt");
        std::string line;
        std::getline(mot, line);
        require(line.starts_with("1,1,5,5,12,20,"),
                "MOT export must restore one-based coordinates");
        mot.close();
        cv::VideoCapture comparison((root / "video/comparison.avi").string());
        cv::Mat frame;
        int count = 0;
        while (comparison.read(frame)) {
            ++count;
            require(frame.cols == 64 && frame.rows == 32, "Comparison layout wrong");
        }
        comparison.release();
        require(count == 3, "Comparison lost frames");
        Detector kalman_detector;
        const auto three = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "kalman", kalman_detector, {0, Json::object(), true});
        require(three["tracking"]["kalman"]["idf1"] == 1.0 &&
                    three["performance"]["kalman_tracker"]["samples"] == 3,
                "Kalman comparison metrics or timing samples wrong");
        const auto three_frames = load(root / "kalman/tracking-frames.json");
        require(three_frames[0]["raw_detections"].size() == 1 &&
                    three_frames[0]["kalman"] == three_frames[0]["iou"],
                "Kalman/raw detection export changed shared inputs");
        cv::VideoCapture three_video((root / "kalman/comparison.avi").string());
        int three_count = 0;
        while (three_video.read(frame)) {
            ++three_count;
            require(frame.cols == 96, "Kalman panel missing");
        }
        three_video.release();
        require(three_count == 3, "Kalman comparison lost frames");
        Detector four_detector;
        const auto four = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "four", four_detector, {0, Json::object(), true, true});
        require(four_detector.calls == 3 && four["tracking"]["kalman_center"]["idf1"] == 1.0 &&
                    four["performance"]["kalman_center_tracker"]["samples"] == 3,
                "Fourth tracker duplicated inference or omitted metrics/timings");
        const auto four_frames = load(root / "four/tracking-frames.json");
        require(four_frames[0]["kalman_center"] == four_frames[0]["kalman"] &&
                    four_frames[0]["kalman"] == four_frames[0]["iou"] &&
                    fs::exists(root / "four/kalman-center-mot.txt") &&
                    four["tracking"]["parameters"]["kalman_center"]["gating_dimensions"] == 2 &&
                    four["tracking"]["parameters"]["kalman_center"]["mahalanobis_gate"] == 9.2103,
                "Center comparison did not preserve shared inputs/parameters/export");
        cv::VideoCapture four_video((root / "four/comparison.avi").string());
        int four_count = 0;
        while (four_video.read(frame)) {
            ++four_count;
            require(frame.cols == 128 && frame.rows == 32, "Four-panel layout wrong");
        }
        four_video.release();
        require(four_count == 3, "Four-panel comparison lost frames");
        Detector center_detector;
        const auto center_only = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "center-only", center_detector, {0, Json::object(), false, true});
        require(!center_only["tracking"].contains("kalman") &&
                    center_only["tracking"]["kalman_center"]["idf1"] == 1.0 &&
                    center_detector.calls == 3,
                "Center flag must be independent of legacy Kalman flag");
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
            root / "manifest.json", root / "center-images", detector, {0, Json::object(), false, true}); },
            "Images accepted center-only tracking comparison");
        require(!fs::exists(root / "center-images"), "Image flag validation created output side effects");
        movie["frames"] = 4;
        save(root / "movie-bad.json", movie);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(
                    root / "movie-bad.json", root / "early-end", detector, {0, Json::object()});
            },
            "Early video EOF accepted");
        movie["frames"] = 2;
        movie["ground_truth"].erase(2);
        save(root / "movie-bad.json", movie);
        rejects(
            [&] {
                (void)aegisvision::evaluation::run_quality_benchmark(
                    root / "movie-bad.json", root / "extra-frames", detector, {0, Json::object()});
            },
            "Extra decoded frames accepted");
        Detector empty;
        empty.empty = true;
        const auto none = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "empty-detections", empty, {0, Json::object()});
        require(none["tracking"]["iou"]["fn"] == 3 && none["tracking"]["iou"]["idf1"] == 0.0,
                "Empty predictions not reported honestly");
        fs::remove_all(root);
        std::cout << "Quality benchmark integration tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "; fixtures: " << root << '\n';
        return 1;
    }
}
