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
class Appearance final : public aegisvision::IEmbedder {
  public:
    int calls{};
    std::vector<float> embed_image(const aegisvision::Frame &frame,
                                   const aegisvision::Detection &detection) override {
        require(frame.image && detection.label == "person", "Appearance received non-person or no pixels");
        ++calls;
        std::vector<float> feature(512);
        feature[0] = 1;
        return feature;
    }
    std::vector<float> embed_text(std::string_view) override { throw std::logic_error("No text Re-ID"); }
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
        Detector five_detector;
        Appearance appearance;
        const auto five = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "five", five_detector,
            {0, Json::object(), true, true, &appearance});
        require(five_detector.calls == 3 && appearance.calls == 3 &&
                    five["tracking"]["kalman_reid"]["idf1"] == 1.0 &&
                    five["performance"]["appearance"]["samples"] == 3 &&
                    five["performance"]["appearance_crop_calls"] == 3 &&
                    five["tracking"]["kalman_reid"]["diagnostics"]["appearance_matches"] == 2 &&
                    five["tracking"]["parameters"]["kalman_reid"]["appearance"]["max_cosine_distance"] == .20,
                "Appearance comparison omitted embeddings, frozen defaults or diagnostics");
        const auto five_frames = load(root / "five/tracking-frames.json");
        for (const auto name : {"iou", "two_stage", "kalman", "kalman_center"}) {
            require(five["tracking"][name] == four["tracking"][name], "Appearance changed an old tracker metric");
            for (std::size_t i = 0; i < five_frames.size(); ++i)
                require(five_frames[i][name] == four_frames[i][name] &&
                            five_frames[i]["raw_detections"] == four_frames[i]["raw_detections"],
                        "Appearance changed shared geometry-only inputs or predictions");
        }
        require(fs::exists(root / "five/kalman-reid-mot.txt"), "Re-ID MOT export missing");
        cv::VideoCapture five_video((root / "five/comparison.avi").string());
        int five_count = 0;
        while (five_video.read(frame)) {
            ++five_count;
            require(frame.cols == 160 && frame.rows == 32, "Five-panel layout wrong");
        }
        five_video.release();
        require(five_count == 3, "Five-panel comparison lost frames");
        Detector six_detector; Appearance six_appearance;
        const auto six = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "six", six_detector,
            {0, Json::object(), true, true, &six_appearance, false, true});
        const auto six_frames = load(root / "six/tracking-frames.json");
        require(six_detector.calls == 3 && six_appearance.calls == 3 &&
            six["tracking"]["kalman_reid_active"]["hota"]["mean"] == 1 &&
            fs::exists(root / "six/kalman-reid-active-mot.txt"), "Ablation repeated inference or omitted metrics");
        for (const auto* name : {"iou", "two_stage", "kalman", "kalman_center", "kalman_reid"}) {
            require(six["tracking"][name] == five["tracking"][name], "Ablation changed old metrics");
            for (std::size_t i = 0; i < six_frames.size(); ++i)
                require(six_frames[i][name] == five_frames[i][name], "Ablation changed old tracks");
        }
        require(aegisvision::evaluation::audit_tracking_output(root / "six", root / "six-audit")
                    ["trackers"].size() == 6, "Audit omitted sixth tracker");
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "no-reid-ablation", six_detector,
            {0, Json::object(), true, true, nullptr, false, true}); }, "Ablation without encoder accepted");
        Detector trace_detector;
        const auto traced = aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "traced", trace_detector,
            {0, Json::object(), true, true, &appearance, true});
        require(traced["tracking"] == five["tracking"] &&
            load(root / "traced/tracking-frames.json") == five_frames &&
            !fs::exists(root / "five/association-trace.jsonl"), "Trace changed metrics/outputs or default behavior");
        std::ifstream trace_stream(root / "traced/association-trace.jsonl");
        std::string trace_line; int trace_rows = 0;
        while (std::getline(trace_stream, trace_line)) {
            const auto row = Json::parse(trace_line); ++trace_rows;
            require(row["observations"].size() == 1 && row["candidates"].size() == 1 &&
                row["candidates"][0]["detection_index"] == 0, "Trace JSONL mapping differs");
        }
        require(trace_rows == 9 && traced["association_trace"]["bytes"] == fs::file_size(root / "traced/association-trace.jsonl"),
                "Trace missing frame/tracker rows or size");
        trace_stream.close(); // Release the fixture handle before Windows cleanup.
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
            root / "movie.json", root / "no-tracker-trace", trace_detector,
            {0, Json::object(), false, false, nullptr, true}); }, "Trace without tracker accepted");
        const auto audit = aegisvision::evaluation::audit_tracking_output(root / "five", root / "audit");
        require(audit["trackers"].size() == 5 && audit["trackers"]["kalman_reid"]["events"].empty() &&
                fs::exists(root / "audit/switch-events.csv"), "Offline audit omitted trackers or CSV");
        rejects([&] { (void)aegisvision::evaluation::audit_tracking_output(root / "five", root / "audit"); },
                "Audit overwrote previous output");
        auto changed_report = five;
        changed_report["tracking"]["iou"]["id_switches"] = 99;
        save(root / "five/report.json", changed_report);
        rejects([&] { (void)aegisvision::evaluation::audit_tracking_output(root / "five", root / "bad-audit"); },
                "Audit accepted metrics inconsistent with exported frames");
        require(!fs::exists(root / "bad-audit"), "Invalid audit created output");
        save(root / "five/report.json", five);
        auto changed_frames = five_frames;
        changed_frames[0]["ground_truth"][0]["id"] = -1;
        save(root / "five/tracking-frames.json", changed_frames);
        rejects([&] { (void)aegisvision::evaluation::audit_tracking_output(root / "five", root / "bad-audit"); },
                "Audit accepted invalid GT identity");
        changed_frames = five_frames;
        changed_frames[0]["ground_truth"][0]["bbox"][0] = -500;
        save(root / "five/tracking-frames.json", changed_frames);
        rejects([&] { (void)aegisvision::evaluation::audit_tracking_output(root / "five", root / "bad-audit"); },
                "Audit accepted GT differing from manifest");
        save(root / "five/tracking-frames.json", five_frames);
        require(five["tracking"]["kalman_reid"]["hota"]["mean"] == 1.0 &&
                five["tracking"]["kalman_reid"]["hota"]["thresholds"].size() == 19,
                "Native HOTA did not reach benchmark JSON");
        auto transfer_movie = movie;
        transfer_movie["video"] = "fixture-raw.mp4";
        fs::copy_file(video, root / "fixture-raw.mp4");
        transfer_movie["project_split"] = "validation";
        transfer_movie["evaluation_protocol"] = {
            {"version", 1}, {"protocol", "tracking-transfer-v1"},
            {"trackers", {"iou", "two_stage", "kalman", "kalman_center", "kalman_reid"}},
            {"validation_sequences", {"fixture"}}, {"test_sequences", Json::array()},
            {"model_sha256", "detector"}, {"detector_config_sha256", "config"},
            {"appearance_model_sha256", "appearance"},
            {"parameters", {{"low", .1}, {"high", .35}, {"new", .5}, {"match_iou", .3},
                {"max_missed_frames", 20}, {"full_box_gate", 13.2767}, {"center_gate", 9.2103},
                {"max_cosine_distance", .2}, {"appearance_weight", .5}, {"appearance_momentum", .9}}}};
        aegisvision::evaluation::QualityRunConfig frozen{0,
            {{"model_sha256", "detector"}, {"config_sha256", "config"},
             {"appearance", {{"model_sha256", "appearance"}}}}, true, true, &appearance};
        save(root / "transfer.json", transfer_movie);
        const auto transferred = aegisvision::evaluation::run_quality_benchmark(
            root / "transfer.json", root / "transfer", five_detector, frozen);
        require(transferred["data_provenance"]["project_split"] == "validation",
            "Transfer protocol was not retained in results");
        auto forbidden_ablation = frozen; forbidden_ablation.active_appearance_ablation = true;
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
            root / "transfer.json", root / "transfer-ablation", five_detector, forbidden_ablation); },
            "Ablation silently altered frozen evaluation protocol");
        for (int change = 0; change < 5; ++change) {
            auto config = frozen;
            auto input = transfer_movie;
            if (change == 0) config.compare_kalman = false;
            if (change == 1) config.provenance["model_sha256"] = "changed";
            if (change == 2) config.provenance["appearance"]["model_sha256"] = "changed";
            if (change == 3) input["evaluation_protocol"]["parameters"]["max_cosine_distance"] = .3;
            if (change == 4) input["project_split"] = "test";
            save(root / "bad-transfer.json", input);
            const auto calls_before = five_detector.calls;
            rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
                root / "bad-transfer.json", root / "bad-transfer", five_detector, config); },
                "Transfer protocol drift was accepted");
            require(five_detector.calls == calls_before && !fs::exists(root / "bad-transfer"),
                "Transfer drift was rejected after inference/output creation");
        }
        rejects([&] { (void)aegisvision::evaluation::run_quality_benchmark(
            root / "manifest.json", root / "reid-images", detector,
            {0, Json::object(), false, false, &appearance}); }, "Images accepted appearance comparison");
        require(!fs::exists(root / "reid-images"), "Re-ID image validation created output");
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
