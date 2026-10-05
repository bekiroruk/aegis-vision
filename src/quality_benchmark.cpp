#include "aegisvision/quality_benchmark.hpp"
#include "aegisvision/evaluation.hpp"
#include "aegisvision/kalman_tracker.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/tracking_evaluation.hpp"
#include "aegisvision/two_stage_tracker.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <picosha2.h>
#include <set>
#include <stdexcept>
#include <utility>

namespace aegisvision::evaluation {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
std::string utf8(const fs::path &p) {
    const auto u = p.generic_u8string();
    return {u.begin(), u.end()};
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template <class T> Json nullable(const std::optional<T> &value) {
    return value ? Json(*value) : Json(nullptr);
}
Json box(const BoundingBox &b) { return {b.x1, b.y1, b.x2, b.y2}; }
BoundingBox read_box(const Json &value) {
    if (!value.is_array() || value.size() != 4)
        throw std::invalid_argument("Expected xyxy box");
    BoundingBox b{value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>(),
                  value.at(3).get<float>()};
    if (!std::isfinite(b.x1) || !std::isfinite(b.x2) || !std::isfinite(b.y1) ||
        !std::isfinite(b.y2) || b.x2 <= b.x1 || b.y2 <= b.y1)
        throw std::invalid_argument("Non-finite or degenerate benchmark box");
    return b;
}
std::uint64_t read_id(const Json &value) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
        throw std::invalid_argument("Tracking GT identity must be a non-negative integer");
    return value.get<std::uint64_t>();
}
void validate_prediction(const Detection &value) {
    if (!std::isfinite(value.score) || value.score < 0 || value.score > 1)
        throw std::invalid_argument("Invalid detector probability");
    (void)read_box(box(value.bbox));
}
int number(const Json &root, const char *key, int low, int high) {
    if (!root.at(key).is_number_integer())
        throw std::invalid_argument("Expected integer benchmark dimension/count");
    const auto n = root.at(key).get<std::int64_t>();
    if (n < low || n > high)
        throw std::invalid_argument("Benchmark dimension/count limit");
    return static_cast<int>(n);
}
std::string text(const Json &root, const char *key) {
    const auto s = root.at(key).get<std::string>();
    if (s.empty() || s.size() > 256 ||
        s.find_first_of(std::string("\0\r\n", 3)) != std::string::npos)
        throw std::invalid_argument("Invalid benchmark identifier");
    return s;
}
fs::path local_file(const fs::path &base, const std::string &name) {
    const auto relative = fs::u8path(name);
    if (relative.empty() || relative.is_absolute() || relative.has_root_name())
        throw std::invalid_argument("Benchmark paths must be relative");
    const auto path = fs::weakly_canonical(base / relative), inside = path.lexically_relative(base);
    if (inside.empty() || inside.is_absolute() || *inside.begin() == ".." ||
        !fs::is_regular_file(path))
        throw std::invalid_argument("Missing or escaped benchmark file");
    return path;
}
void write_json(const fs::path &path, const Json &data) {
    std::ofstream stream(path, std::ios::binary);
    stream << data.dump(2) << '\n';
    stream.flush();
    if (!stream)
        throw std::runtime_error("Could not finalize benchmark JSON");
}
Json latency(const std::vector<double> &values) {
    const auto s = summarize_latencies(values);
    if (!s)
        return nullptr;
    return {{"samples", s->sample_count}, {"mean_ms", s->mean_ms}, {"min_ms", s->min_ms},
            {"p50_ms", s->p50_ms},        {"p95_ms", s->p95_ms},   {"max_ms", s->max_ms}};
}
Json operating(const OperatingPoint &p) {
    return {{"tp", p.true_positives},
            {"fp", p.false_positives},
            {"fn", p.false_negatives},
            {"ignored", p.ignored_predictions},
            {"precision", nullable(p.precision)},
            {"recall", nullable(p.recall)}};
}
Json detection_report(const DetectionReport &r) {
    auto classes = Json::array();
    for (const auto &c : r.classes)
        classes.push_back({{"label", c.label},
                           {"gt", c.ground_truth_count},
                           {"crowd_gt", c.crowd_ground_truth_count},
                           {"predictions", c.prediction_count},
                           {"ap50", nullable(c.ap50)},
                           {"ap50_95", nullable(c.ap50_95)},
                           {"operating_point", operating(c.operating_point)}});
    return {{"images", r.image_count},
            {"ground_truth", r.ground_truth_count},
            {"crowd_ground_truth", r.crowd_ground_truth_count},
            {"predictions", r.prediction_count},
            {"ap50", nullable(r.ap50)},
            {"ap50_95", nullable(r.ap50_95)},
            {"operating_point", operating(r.operating_point)},
            {"per_class", classes},
            {"protocol", "category-aware bbox, all area, maxDets 100 per image/category, 101 "
                         "recall points, IoU .50:.05:.95; operating score .25/IoU .5"}};
}
Json tracking_report(const TrackingReport &r) {
    return {{"frames", r.frames},
            {"gt", r.ground_truth_detections},
            {"predictions", r.predicted_detections},
            {"gt_identities", r.ground_truth_identities},
            {"predicted_identities", r.predicted_identities},
            {"tp", r.true_positives},
            {"fp", r.false_positives},
            {"fn", r.false_negatives},
            {"id_switches", r.id_switches},
            {"idtp", r.id_true_positives},
            {"idfp", r.id_false_positives},
            {"idfn", r.id_false_negatives},
            {"precision", nullable(r.precision)},
            {"recall", nullable(r.recall)},
            {"mota", nullable(r.mota)},
            {"motp", nullable(r.motp)},
            {"idf1", nullable(r.idf1)},
            {"iou_threshold", r.iou_threshold}};
}
Json tracking_report(const std::vector<TrackingFrame>& frames) {
    auto report = tracking_report(evaluate_tracking(frames));
    const auto h = evaluate_hota(frames);
    Json levels = Json::array();
    for (const auto& a : h.thresholds)
        levels.push_back({{"alpha", a.alpha}, {"tp", a.true_positives},
            {"fp", a.false_positives}, {"fn", a.false_negatives}, {"hota", a.hota},
            {"det_a", a.detection_accuracy}, {"ass_a", a.association_accuracy},
            {"loc_a", a.localization_accuracy}});
    report["hota"] = {{"mean", nullable(h.hota)}, {"det_a", nullable(h.detection_accuracy)},
        {"ass_a", nullable(h.association_accuracy)}, {"loc_a", nullable(h.localization_accuracy)},
        {"thresholds", levels}, {"protocol", "global alignment then one frame assignment; alpha .05:.05:.95; arithmetic mean of per-alpha HOTA"}};
    return report;
}
struct ImageEntry {
    std::string id;
    int coco_id{}, width{}, height{};
    fs::path path;
    std::string sha256;
    std::vector<GroundTruth> gt;
};
void dimensions(const cv::Mat &image, int width, int height) {
    if (image.empty() || image.type() != CV_8UC3 || image.cols != width || image.rows != height)
        throw std::runtime_error("Decoded benchmark dimensions differ from manifest");
}
std::vector<Detection> selected(const std::vector<Detection> &values,
                                const std::set<std::string> &labels) {
    std::vector<Detection> out;
    for (const auto &d : values)
        if (labels.contains(d.label))
            out.push_back(d);
    return out;
}
void warmup(IDetector &detector, const cv::Mat &image, int iterations) {
    for (int i = 0; i < iterations; ++i)
        (void)detector.detect(vision::image_frame(image, "warmup", "quality-warmup"));
}
Json image_run(const Json &manifest, const fs::path &base, const fs::path &output,
               IDetector &detector, int iterations) {
    std::vector<std::string> labels;
    std::map<std::string, int> category_ids;
    std::set<int> category_numbers;
    for (const auto &c : manifest.at("classes")) {
        const auto label = text(c, "label");
        const auto id = number(c, "category_id", 1, 10000);
        if (!category_ids.emplace(label, id).second || !category_numbers.insert(id).second)
            throw std::invalid_argument("Duplicate category");
        labels.push_back(label);
    }
    if (labels.empty() || labels.size() > 80)
        throw std::invalid_argument("Taxonomy must contain 1..80 labels");
    const std::set<std::string> label_set(labels.begin(), labels.end());
    std::vector<ImageEntry> entries;
    std::set<std::string> ids;
    std::set<int> coco_ids;
    if (!manifest.at("images").is_array() || manifest.at("images").empty() ||
        manifest.at("images").size() > 1000)
        throw std::invalid_argument("Require 1..1000 images");
    for (const auto &row : manifest.at("images")) {
        ImageEntry e;
        e.id = text(row, "id");
        e.coco_id = number(row, "coco_image_id", 1, 2000000000);
        if (!ids.insert(e.id).second || !coco_ids.insert(e.coco_id).second)
            throw std::invalid_argument("Duplicate benchmark image");
        e.width = number(row, "width", 1, 4096);
        e.height = number(row, "height", 1, 4096);
        e.path = local_file(base, text(row, "path"));
        e.sha256 = row.at("sha256").get<std::string>();
        if (quality_file_sha256(e.path) != e.sha256)
            throw std::runtime_error("Benchmark image SHA256 changed");
        if (!row.at("ground_truth").is_array() || row.at("ground_truth").size() > 1000)
            throw std::invalid_argument("Ground-truth image limit");
        for (const auto &g : row.at("ground_truth")) {
            const auto label = text(g, "label");
            if (!label_set.contains(label))
                throw std::invalid_argument("Unknown GT category");
            e.gt.push_back({read_box(g.at("bbox")), label, g.at("crowd").get<bool>()});
        }
        entries.push_back(std::move(e));
    }
    auto first = vision::load_image(entries.front().path);
    dimensions(first, entries.front().width, entries.front().height);
    const auto warm_start = Clock::now();
    warmup(detector, first, iterations);
    const auto warm_ms = elapsed(warm_start);
    std::vector<Frame> frames;
    std::vector<double> inference, read_times;
    auto predictions = Json::array();
    std::ofstream timings(output / "timings.csv");
    timings << "frame_id,read_ms,detector_ms\n" << std::setprecision(12);
    const auto run_start = Clock::now();
    for (const auto &e : entries) {
        const auto read_start = Clock::now();
        auto image = vision::load_image(e.path);
        dimensions(image, e.width, e.height);
        const auto read_ms = elapsed(read_start);
        const auto start = Clock::now();
        auto all = detector.detect(vision::image_frame(image, e.id, "quality-images"));
        const auto ms = elapsed(start);
        auto detections = selected(all, label_set);
        frames.push_back({e.id, e.gt, detections});
        inference.push_back(ms);
        read_times.push_back(read_ms);
        for (const auto &d : detections)
            predictions.push_back(
                {{"image_id", e.coco_id},
                 {"category_id", category_ids.at(d.label)},
                 {"bbox", {d.bbox.x1, d.bbox.y1, d.bbox.x2 - d.bbox.x1, d.bbox.y2 - d.bbox.y1}},
                 {"score", d.score}});
        timings << e.id << ',' << read_ms << ',' << ms << '\n';
        if (frames.size() == 1)
            vision::save_image(output / "preview.jpg",
                               vision::annotate(image, selected(all, label_set)));
    }
    const auto run_ms = elapsed(run_start);
    timings.flush();
    if (!timings)
        throw std::runtime_error("Benchmark timings write failed");
    for (const auto &e : entries)
        if (quality_file_sha256(e.path) != e.sha256)
            throw std::runtime_error("Benchmark image changed during evaluation");
    write_json(output / "predictions.json", predictions);
    return {
        {"detection", detection_report(evaluate_detection(frames, labels))},
        {"performance",
         {{"detector", latency(inference)},
          {"image_read", latency(read_times)},
          {"warmup_iterations", iterations},
          {"warmup_ms", warm_ms},
          {"loop_wall_ms", run_ms},
          {"loop_fps", 1000.0 * frames.size() / run_ms},
          {"scope", "detector includes owned frame copy, preprocessing, forward, NMS; loop "
                    "includes image reads, CSV formatting/writes and first preview; excludes model "
                    "loading, warmup, metrics, hashes, output flush and JSON serialization"}}},
        {"limitations",
         {"Selected large-object-biased 64-scene subset when using shipped preparer; not the "
          "official 80-category COCO score.",
          "No fine-tuning or threshold selection was performed on this run."}}};
}
Json objects(const std::vector<TrackObject> &values) {
    auto out = Json::array();
    for (const auto &o : values)
        out.push_back({{"id", o.id}, {"bbox", box(o.bbox)}});
    return out;
}
cv::Mat display(const cv::Mat &image, const std::vector<Track> &tracks,
                const std::vector<TrackObject> &gt, const std::string &title) {
    std::vector<Detection> drawn;
    for (const auto &t : tracks)
        drawn.push_back({t.bbox, "ID " + std::to_string(t.track_id), t.score, {}, {}});
    auto result = vision::annotate(image, drawn);
    for (const auto &g : gt) {
        const auto &b = g.bbox;
        if (b.x2 <= 0 || b.y2 <= 0 || b.x1 >= image.cols || b.y1 >= image.rows)
            continue;
        const auto x = [&](float value) {
            return static_cast<int>(std::clamp<double>(value, 0, image.cols - 1));
        };
        const auto y = [&](float value) {
            return static_cast<int>(std::clamp<double>(value, 0, image.rows - 1));
        };
        cv::rectangle(result, cv::Point(x(b.x1), y(b.y1)), cv::Point(x(b.x2), y(b.y2)),
                      cv::Scalar(0, 220, 255), 1);
    }
    cv::rectangle(result, {0, 0}, {image.cols, 26}, cv::Scalar(10, 10, 10), cv::FILLED);
    cv::putText(result, title + " | yellow GT / green track", {6, 18}, cv::FONT_HERSHEY_SIMPLEX,
                0.43, {255, 255, 255}, 1);
    return result;
}
struct KalmanComparison {
    std::string name, title;
    KalmanTrackerConfig config;
    KalmanTracker tracker;
    std::vector<TrackingFrame> frames;
    std::vector<double> times;
    std::vector<Track> observed;
    std::ofstream rows;
    double last_ms{};
    KalmanComparison(std::string key, std::string label, KalmanTrackerConfig settings)
        : name(std::move(key)), title(std::move(label)), config(settings), tracker(settings) {}
};
Json video_run(const Json &manifest, const fs::path &base, const fs::path &output,
               IDetector &detector, int iterations, bool compare_kalman, bool compare_center,
               IEmbedder *appearance) {
    if (manifest.at("classes") != Json::array({"person"}))
        throw std::invalid_argument("Tracking baseline evaluates person only");
    const auto input = local_file(base, text(manifest, "video"));
    if (quality_file_sha256(input) != manifest.at("video_sha256").get<std::string>())
        throw std::runtime_error("Video SHA256 changed");
    const int count = number(manifest, "frames", 1, 1000),
              width = number(manifest, "width", 2, 1920),
              height = number(manifest, "height", 2, 1080);
    const auto fps = manifest.at("source_fps").get<double>();
    if (!std::isfinite(fps) || fps < 1 || fps > 120 || width % 2 || height % 2)
        throw std::invalid_argument("Invalid video dimensions/FPS");
    std::vector<std::vector<TrackObject>> gt(static_cast<std::size_t>(count));
    if (!manifest.at("ground_truth").is_array() || manifest.at("ground_truth").size() > 100000)
        throw std::invalid_argument("Video GT limit");
    for (const auto &g : manifest.at("ground_truth"))
        gt.at(number(g, "frame_index", 1, count) - 1)
            .push_back({read_id(g.at("id")), read_box(g.at("bbox"))});
    std::vector<TrackingFrame> validate;
    for (int i = 0; i < count; ++i)
        validate.push_back({i + 1, gt.at(i), {}});
    (void)evaluate_tracking(validate);
    cv::VideoCapture capture(utf8(input));
    if (!capture.isOpened())
        throw std::runtime_error("Cannot decode benchmark video");
    if (std::abs(capture.get(cv::CAP_PROP_FPS) - fps) > .01)
        throw std::runtime_error("Decoded video FPS differs from annotation metadata");
    cv::Mat image;
    if (!capture.read(image))
        throw std::runtime_error("Empty benchmark video");
    dimensions(image, width, height);
    const auto warm_start = Clock::now();
    warmup(detector, image, iterations);
    const auto warm_ms = elapsed(warm_start);
    IoUTracker iou(.30F, 20);
    TwoStageTracker two_stage;
    std::vector<std::unique_ptr<KalmanComparison>> comparisons;
    if (compare_kalman)
        comparisons.push_back(std::make_unique<KalmanComparison>(
            "kalman", "Kalman active-first", KalmanTrackerConfig{}));
    if (compare_center) {
        KalmanTrackerConfig center;
        center.gate_mode = KalmanGateMode::CenterOnly;
        center.gating_threshold = kalman_center_gate99;
        comparisons.push_back(std::make_unique<KalmanComparison>(
            "kalman_center", "Kalman center-only", center));
    }
    if (appearance) {
        KalmanTrackerConfig reid;
        reid.gate_mode = KalmanGateMode::CenterOnly;
        reid.gating_threshold = kalman_center_gate99;
        reid.use_appearance = true;
        comparisons.push_back(std::make_unique<KalmanComparison>(
            "kalman_reid", "Kalman + OSNet appearance", reid));
    }
    std::vector<TrackingFrame> iou_frames, two_frames;
    std::vector<double> inference, read_times, iou_times, two_times, appearance_times;
    std::size_t appearance_calls = 0;
    auto exported = Json::array();
    std::ofstream timings(output / "timings.csv"), iou_rows(output / "iou-mot.txt"),
        two_rows(output / "two-stage-mot.txt");
    timings << "frame_index,read_ms,detector_ms,iou_ms,two_stage_ms";
    for (auto &entry : comparisons) {
        timings << ',' << entry->name << "_ms";
        const auto filename = entry->name == "kalman" ? "kalman-mot.txt"
            : entry->name == "kalman_center" ? "kalman-center-mot.txt" : "kalman-reid-mot.txt";
        entry->rows.open(output / filename);
        entry->rows << std::setprecision(9);
        if (!entry->rows) throw std::runtime_error("Cannot open Kalman MOT export");
    }
    if (appearance) timings << ",appearance_ms,appearance_crops";
    timings << '\n' << std::setprecision(12);
    iou_rows << std::setprecision(9);
    two_rows << std::setprecision(9);
    cv::VideoWriter writer(utf8(output / "comparison.avi"), cv::CAP_OPENCV_MJPEG,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                           {width * static_cast<int>(2 + comparisons.size()), height});
    if (!writer.isOpened())
        throw std::runtime_error("Cannot open comparison AVI");
    const auto run_start = Clock::now();
    double read_ms = 0;
    for (int i = 0; i < count; ++i) {
        if (i) {
            const auto start = Clock::now();
            if (!capture.read(image))
                throw std::runtime_error("Video ended before labelled frame count");
            read_ms = elapsed(start);
        }
        dimensions(image, width, height);
        const auto detect_start = Clock::now();
        auto detections =
            detector.detect(vision::image_frame(image, std::to_string(i + 1), "quality-video"));
        const auto ms = elapsed(detect_start);
        std::vector<Detection> high, low;
        auto raw = Json::array();
        for (const auto &d : detections)
            if (d.label == "person") {
                validate_prediction(d);
                raw.push_back({{"bbox", box(d.bbox)}, {"label", d.label}, {"score", d.score}});
                if (d.score >= .10F)
                    low.push_back(d);
                if (d.score >= .35F)
                    high.push_back(d);
            }
        std::vector<Detection> appearance_low;
        double appearance_ms = 0;
        if (appearance) {
            if (low.size() > KalmanTracker::max_detections)
                throw std::invalid_argument("Appearance benchmark exceeds bounded detection capacity");
            const auto start = Clock::now();
            appearance_low = low; // Never change the inputs of the four geometry-only trackers.
            if (!appearance_low.empty()) {
                const auto frame = vision::image_frame(image, std::to_string(i + 1), "quality-appearance");
                for (auto &d : appearance_low) {
                    d.embedding = appearance->embed_image(frame, d);
                    ++appearance_calls;
                }
            }
            appearance_ms = elapsed(start);
            appearance_times.push_back(appearance_ms);
        }
        const auto iou_start = Clock::now();
        auto a = iou.update(high);
        const auto a_ms = elapsed(iou_start);
        const auto two_start = Clock::now();
        auto b = two_stage.update(low);
        const auto b_ms = elapsed(two_start);
        for (auto &entry : comparisons) {
            const auto start = Clock::now();
            entry->observed = entry->tracker.update(entry->config.use_appearance ? appearance_low : low);
            entry->last_ms = elapsed(start);
            entry->times.push_back(entry->last_ms);
        }
        TrackingFrame fa{i + 1, gt.at(i), {}}, fb{i + 1, gt.at(i), {}};
        const auto append = [&](const auto &tracks, auto &frame, std::ofstream &rows) {
            for (const auto &t : tracks) {
                frame.predictions.push_back({t.track_id, t.bbox});
                const auto &z = t.bbox;
                // Export to original MOT 1-based image coordinates for reference tools.
                rows << i + 1 << ',' << t.track_id << ',' << z.x1 + 1 << ',' << z.y1 + 1 << ','
                     << z.x2 - z.x1 << ',' << z.y2 - z.y1 << ',' << t.score << ",-1,-1,-1\n";
            }
        };
        append(a, fa, iou_rows);
        append(b, fb, two_rows);
        for (auto &entry : comparisons) {
            TrackingFrame frame{i + 1, gt.at(i), {}};
            append(entry->observed, frame, entry->rows);
            entry->frames.push_back(std::move(frame));
        }
        iou_frames.push_back(fa);
        two_frames.push_back(fb);
        Json frame_export = {{"frame_index", i + 1},
                             {"ground_truth", objects(fa.ground_truth)},
                             {"iou", objects(fa.predictions)},
                             {"two_stage", objects(fb.predictions)},
                             {"raw_detections", raw}};
        for (const auto &entry : comparisons)
            frame_export[entry->name] = objects(entry->frames.back().predictions);
        exported.push_back(std::move(frame_export));
        cv::Mat combined;
        cv::hconcat(display(image, a, fa.ground_truth, "IoU"),
                    display(image, b, fb.ground_truth, "Two-stage"), combined);
        for (const auto &entry : comparisons) {
            cv::Mat extended;
            cv::hconcat(combined, display(image, entry->observed, gt.at(i), entry->title), extended);
            combined = extended;
        }
        writer.write(combined);
        if (!i)
            vision::save_image(output / "preview.jpg", combined);
        inference.push_back(ms);
        iou_times.push_back(a_ms);
        two_times.push_back(b_ms);
        if (i)
            read_times.push_back(read_ms);
        timings << i + 1 << ',' << read_ms << ',' << ms << ',' << a_ms << ',' << b_ms;
        for (const auto &entry : comparisons)
            timings << ',' << entry->last_ms;
        if (appearance) timings << ',' << appearance_ms << ',' << appearance_low.size();
        timings << '\n';
    }
    const auto run_ms = elapsed(run_start);
    if (capture.read(image))
        throw std::runtime_error("Video has frames beyond labelled sequence");
    writer.release();
    if (quality_file_sha256(input) != manifest.at("video_sha256").get<std::string>())
        throw std::runtime_error("Video changed during evaluation");
    timings.flush();
    iou_rows.flush();
    two_rows.flush();
    for (auto &entry : comparisons) {
        entry->rows.flush();
        if (!entry->rows) throw std::runtime_error("Kalman MOT export failed");
    }
    if (!timings || !iou_rows || !two_rows)
        throw std::runtime_error("Benchmark output failed");
    write_json(output / "tracking-frames.json", exported);
    Json report = {
        {"tracking",
         {{"iou", tracking_report(iou_frames)},
          {"two_stage", tracking_report(two_frames)},
          {"protocol", "CLEAR and Identity at IoU .5; single person class; same decoded frames "
                       "and detector outputs; independent tracker states; no GT tracker input"},
          {"parameters",
           {{"match_iou", .30},
            {"max_missed_frames", 20},
            {"iou_score", .35},
            {"two_stage_low", .10},
            {"two_stage_high", .35},
            {"two_stage_new", .50}}}}},
        {"performance",
         {{"detector", latency(inference)},
          {"decode", latency(read_times)},
          {"iou_tracker", latency(iou_times)},
          {"two_stage_tracker", latency(two_times)},
          {"warmup_iterations", iterations},
          {"warmup_ms", warm_ms},
          {"loop_wall_ms", run_ms},
          {"loop_fps", 1000.0 * count / run_ms},
          {"scope",
           "detector includes owned copy/preprocessing/forward/NMS; loop includes decode "
           "except first frame, selected trackers, overlays, MJPEG writes and CSV/MOT formatting; "
           "excludes model load, warmup, metrics, hashes, output flush/finalization and JSON "
           "serialization; first decoded frame is excluded from decode samples"}}},
        {"limitations",
         {"Single MOT15 training sequence using official reencoded preview video, not "
          "challenge leaderboard-comparable scores.",
          "No multi-camera or standalone Re-ID retrieval evaluation; no parameter tuning in this run.",
          "Tracking throughput includes comparison rendering and is not a live RTSP FPS "
          "guarantee."}}};
    for (const auto &entry : comparisons) {
        report["tracking"][entry->name] = tracking_report(entry->frames);
        report["tracking"]["parameters"][entry->name] = {
            {"low", .10},
            {"high", .35},
            {"new", .50},
            {"match_iou", .30},
            {"max_missed_frames", 20},
            {"mahalanobis_gate", entry->config.gating_threshold},
            {"gating_mode", entry->config.gate_mode == KalmanGateMode::CenterOnly ? "center" : "full-box"},
            {"gating_dimensions", entry->config.gate_mode == KalmanGateMode::CenterOnly ? 2 : 4},
            {"motion_dt", "one decoded frame"},
            {"association", "active-high, active-low, lost-high"}};
        const auto &s = entry->tracker.stats();
        report["tracking"][entry->name]["diagnostics"] = {
            {"gate_rejections", s.gate_rejections},
            {"numerical_resets", s.numerical_resets},
            {"reactivations", s.reactivations},
            {"low_confidence_matches", s.low_confidence_matches},
            {"capacity_rejections", s.capacity_rejections}};
        report["performance"][entry->name + "_tracker"] = latency(entry->times);
        if (entry->config.use_appearance) {
            report["tracking"]["parameters"][entry->name]["appearance"] = {
                {"dimension", entry->config.appearance_dimension},
                {"max_cosine_distance", entry->config.max_cosine_distance},
                {"weight", entry->config.appearance_weight},
                {"momentum", entry->config.appearance_momentum},
                {"prototype_updates", "high-score matches only; births seed normalized features"},
                {"hard_gates", "class, IoU, center Mahalanobis, cosine distance"}};
            auto &diagnostics = report["tracking"][entry->name]["diagnostics"];
            diagnostics["appearance_matches"] = s.appearance_matches;
            diagnostics["appearance_rejections"] = s.appearance_rejections;
            diagnostics["appearance_updates"] = s.appearance_updates;
        }
    }
    if (appearance) {
        report["performance"]["appearance"] = latency(appearance_times);
        report["performance"]["appearance_crop_calls"] = appearance_calls;
        report["performance"]["appearance_warmup_iterations"] = 0;
        report["performance"]["appearance_scope"] =
            "per decoded frame: low-score person copy, one owned frame copy, crop/resize/normalization, "
            "one model forward per crop and output L2; no separate crop warmup, first real crop included; "
            "model constructor/load/probe excluded; tracker timing excludes embedding; loop includes both";
        report["limitations"].push_back(
            "Appearance-assisted single-camera association is not full DeepSORT, cross-camera identity "
            "or proof of who a person is. Dataset split and prior exposure are documented in data_provenance.");
    }
    return report;
}
} // namespace
std::string quality_file_sha256(const fs::path &path) {
    if (!fs::is_regular_file(path) || fs::file_size(path) == 0 ||
        fs::file_size(path) > 256ULL * 1024 * 1024)
        throw std::invalid_argument("Benchmark input must be 1 byte..256 MiB");
    std::ifstream file(path, std::ios::binary);
    picosha2::hash256_one_by_one hash;
    std::array<char, 65536> buffer{};
    if (!file)
        throw std::runtime_error("Cannot open SHA256 input");
    while (file) {
        file.read(buffer.data(), buffer.size());
        hash.process(buffer.begin(), buffer.begin() + file.gcount());
    }
    if (file.bad())
        throw std::runtime_error("SHA256 input read failed");
    hash.finish();
    return picosha2::get_hash_hex_string(hash);
}
Json run_quality_benchmark(const fs::path &manifest_path, const fs::path &output,
                           IDetector &detector, const QualityRunConfig &config) {
    if (config.warmup_iterations < 0 || config.warmup_iterations > 30 ||
        !config.provenance.is_object())
        throw std::invalid_argument("Invalid quality run configuration");
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output)))
        throw std::invalid_argument("Output directory must be new or empty");
    if (!fs::is_regular_file(manifest_path) || fs::file_size(manifest_path) > 16 * 1024 * 1024)
        throw std::invalid_argument("Manifest must be <=16 MiB");
    std::ifstream input(manifest_path, std::ios::binary);
    const auto manifest = Json::parse(input);
    if (manifest.at("version") != 1)
        throw std::invalid_argument("Quality manifest version must be 1");
    const auto kind = text(manifest, "kind"), dataset = text(manifest, "dataset");
    if (kind != "images" && kind != "video")
        throw std::invalid_argument("Unknown benchmark kind");
    if (kind=="images" && (config.compare_kalman || config.compare_kalman_center || config.appearance_embedder))
        throw std::invalid_argument("Kalman comparison requires a video manifest");
    if (manifest.contains("evaluation_protocol")) {
        const auto& protocol = manifest.at("evaluation_protocol");
        if (kind != "video" || protocol.at("version") != 1 ||
            protocol.at("protocol") != "tracking-transfer-v1" ||
            !config.compare_kalman || !config.compare_kalman_center || !config.appearance_embedder ||
            protocol.at("trackers") != Json::array({"iou", "two_stage", "kalman", "kalman_center", "kalman_reid"}))
            throw std::invalid_argument("Frozen transfer protocol requires the five selected trackers");
        const auto& provenance = config.provenance;
        if (provenance.value("model_sha256", "") != protocol.at("model_sha256").get<std::string>() ||
            provenance.value("config_sha256", "") != protocol.at("detector_config_sha256").get<std::string>() ||
            !provenance.contains("appearance") ||
            provenance.at("appearance").value("model_sha256", "") != protocol.at("appearance_model_sha256").get<std::string>())
            throw std::invalid_argument("Models/config differ from frozen transfer protocol");
        const KalmanTrackerConfig defaults;
        const std::map<std::string, double> actual = {
            {"low", defaults.low_threshold}, {"high", defaults.high_threshold},
            {"new", defaults.new_track_threshold}, {"match_iou", defaults.match_iou},
            {"max_missed_frames", defaults.max_missed_frames}, {"full_box_gate", kalman_box_gate99},
            {"center_gate", kalman_center_gate99}, {"max_cosine_distance", defaults.max_cosine_distance},
            {"appearance_weight", defaults.appearance_weight}, {"appearance_momentum", defaults.appearance_momentum}};
        const auto& parameters = protocol.at("parameters");
        if (!parameters.is_object() || parameters.size() != actual.size())
            throw std::invalid_argument("Frozen transfer parameter set differs");
        for (const auto& [key, value] : actual) {
            const auto& declared = parameters.at(key);
            if (!declared.is_number() || !std::isfinite(declared.get<double>()) ||
                std::abs(declared.get<double>() - value) > 1e-6)
                throw std::invalid_argument("Tracker defaults differ from frozen transfer protocol");
        }
        const auto split = manifest.at("project_split").get<std::string>();
        if (split != "validation" && split != "test")
            throw std::invalid_argument("Invalid transfer split");
        bool selected_sequence = false;
        for (const auto& sequence : protocol.at(split + "_sequences"))
            if (manifest.at("video") == sequence.get<std::string>() + "-raw.mp4") selected_sequence = true;
        if (!selected_sequence) throw std::invalid_argument("Sequence differs from frozen transfer split");
    }
    const auto digest = quality_file_sha256(manifest_path);
    const auto base = fs::canonical(manifest_path).parent_path();
    fs::create_directories(output);
    auto report = kind == "images"
                      ? image_run(manifest, base, output, detector, config.warmup_iterations)
                      : video_run(manifest, base, output, detector, config.warmup_iterations,
                                  config.compare_kalman, config.compare_kalman_center, config.appearance_embedder);
    if (quality_file_sha256(manifest_path) != digest)
        throw std::runtime_error("Manifest changed during evaluation");
    report["version"] = 1;
    report["dataset"] = dataset;
    report["kind"] = kind;
    report["manifest_sha256"] = digest;
    report["manifest"] = utf8(fs::canonical(manifest_path));
    report["provenance"] = config.provenance;
    report["opencv_version"] = CV_VERSION;
    report["opencv_threads"] = cv::getNumThreads();
    report["percentiles"] = "nearest rank, ceil(p*n)-1, excluding warmup";
    report["data_provenance"] =
        manifest; // Local-only, keeps exact labels/selection/hash protocol auditable.
    write_json(output / "report.json", report);
    return report;
}
} // namespace aegisvision::evaluation
