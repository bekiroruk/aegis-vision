#include "aegisvision/tracklets.hpp"
#include "aegisvision/reid.hpp"
#include "aegisvision/tracking.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string utf8(const fs::path& p) {
    const auto s = p.u8string(); return {s.begin(), s.end()};
}
Json read(const fs::path& file) {
    if (!fs::is_regular_file(file) || fs::file_size(file) > 16 * 1024 * 1024)
        throw std::invalid_argument("JSON must be a regular file of at most 16 MiB");
    std::ifstream in(file);
    return Json::parse(in, [](int depth, Json::parse_event_t, Json&) {
        if (depth > 16) throw std::invalid_argument("JSON nesting exceeds 16");
        return true;
    });
}
void keys(const Json& j, std::initializer_list<const char*> names) {
    if (!j.is_object() || j.size() != names.size()) throw std::invalid_argument("Unexpected JSON fields");
    for (const auto name : names) if (!j.contains(name)) throw std::invalid_argument(std::string("Missing field: ") + name);
}
std::int64_t number(const Json& j, std::int64_t low, std::int64_t high) {
    if ((!j.is_number_integer() && !j.is_number_unsigned()) ||
        (j.is_number_unsigned() && j.get<std::uint64_t>() > static_cast<std::uint64_t>(high)))
        throw std::invalid_argument("Expected bounded integer");
    auto n = j.get<std::int64_t>();
    if (n < low || n > high) throw std::invalid_argument("Integer out of range");
    return n;
}
std::string hash(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot hash input");
    auto value = picosha2::hash256_hex_string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("Input hash read failed");
    return value;
}
void save(const fs::path& file, const Json& j) {
    std::ofstream out(file); out << j.dump(2) << '\n'; out.close();
    if (!out) throw std::runtime_error("Cannot write JSON output");
}
void fresh(const fs::path& out) {
    if (fs::exists(out)) throw std::invalid_argument("Output directory must not exist");
    fs::create_directories(out.parent_path());
    if (!fs::create_directory(out)) throw std::runtime_error("Cannot reserve output directory");
}
Json item(const CameraTracklet& t) {
    return {{"camera", t.camera}, {"epoch", t.epoch}, {"local_id", t.local_id},
        {"start_ms", t.start_ms}, {"end_ms", t.end_ms}, {"embedding_space", t.embedding_space}, {"embedding", t.embedding}};
}
int extract(const fs::path& job, const fs::path& output) {
    const auto j = read(job);
    keys(j, {"version", "input", "detector_model", "reid_bundle", "camera", "epoch", "max_frames",
             "max_missed_frames", "iou_threshold", "min_observations"});
    number(j.at("version"), 1, 1);
    const auto base = fs::absolute(job).parent_path();
    const auto input = base / fs::u8path(j.at("input").get<std::string>());
    const auto model = base / fs::u8path(j.at("detector_model").get<std::string>());
    const auto bundle = base / fs::u8path(j.at("reid_bundle").get<std::string>());
    const auto max_frames = number(j.at("max_frames"), 1, 1000000);
    const auto missed = static_cast<std::uint32_t>(number(j.at("max_missed_frames"), 0, 10000));
    const auto minimum = number(j.at("min_observations"), 1, 1000000);
    const auto epoch = number(j.at("epoch"), 1, std::numeric_limits<std::int64_t>::max());
    const auto camera = j.at("camera").get<std::string>();
    IoUTracker tracker(j.at("iou_threshold").get<float>(), missed);
    // Validate namespace/output before expensive model construction.
    TrackletCollector validate(camera, epoch, "validation", missed);
    if (fs::exists(output)) throw std::invalid_argument("Output directory must not exist");
    if (!fs::is_regular_file(input)) throw std::invalid_argument("Input must be a local video file");
    const auto input_hash = hash(input), model_hash = hash(model);
    cv::setNumThreads(1);
    vision::YoloDetector detector(model);
    ReIdEmbedder embedder(bundle);
    TrackletCollector collector(camera, epoch, embedder.space_id(), missed);
    cv::VideoCapture capture(utf8(input), cv::CAP_FFMPEG);
    const double fps = capture.get(cv::CAP_PROP_FPS);
    cv::Mat image;
    if (!capture.isOpened() || !std::isfinite(fps) || fps < .001 || fps > 1000 || !capture.read(image))
        throw std::runtime_error("Cannot read video with valid FPS; no timing fallback is allowed");
    const auto size = image.size();
    if (image.type() != CV_8UC3 || size.width % 2 || size.height % 2)
        throw std::runtime_error("Video must have even-sized BGR frames");
    fresh(output);
    // Until tracklets.json is written this directory is incomplete, never mergeable.
    cv::VideoWriter writer(utf8(output / "tracked.avi"), cv::CAP_OPENCV_MJPEG,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, size);
    if (!writer.isOpened()) throw std::runtime_error("Cannot open annotated video output");
    std::ofstream csv(output / "observations.csv");
    csv << "frame,relative_ms,local_id,score,x1,y1,x2,y2\n";
    std::int64_t frames = 0;
    std::size_t observations = 0, preview_count = 0;
    bool limit = false;
    for (;;) {
        if (image.size() != size || image.type() != CV_8UC3) throw std::runtime_error("Video format changed");
        const auto ms = static_cast<std::int64_t>(std::llround(static_cast<double>(frames) * 1000 / fps));
        const auto frame = vision::image_frame(image, std::to_string(frames), camera, ms);
        auto detections = detector.detect(frame);
        std::erase_if(detections, [](const auto& d) { return d.label != "person"; });
        const auto tracks = tracker.update_indexed(detections);
        std::vector<PersonObservation> samples;
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            samples.push_back({tracks[i].track_id, embedder.embed_image(frame, detections[i])});
            const auto& b = detections[i].bbox;
            csv << frames << ',' << ms << ',' << tracks[i].track_id << ',' << detections[i].score
                << ',' << b.x1 << ',' << b.y1 << ',' << b.x2 << ',' << b.y2 << '\n';
        }
        collector.update(ms, samples);
        observations += samples.size();
        auto annotated = image.clone();
        for (const auto& t : tracks) {
            cv::rectangle(annotated, cv::Point(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1)),
                cv::Point(static_cast<int>(t.bbox.x2), static_cast<int>(t.bbox.y2)), cv::Scalar(40, 230, 80), 2);
            const auto label = "#" + std::to_string(t.track_id);
            int baseline = 0;
            const auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, .55, 1, &baseline);
            const int left = std::clamp(static_cast<int>(t.bbox.x1), 0, std::max(0, size.width - text_size.width - 8));
            const int bottom = std::clamp(static_cast<int>(t.bbox.y1) - 2, text_size.height + 6,
                                          std::max(text_size.height + 6, size.height - 4));
            cv::rectangle(annotated, {left, bottom - text_size.height - 5},
                {left + text_size.width + 7, bottom + 3}, cv::Scalar(15, 15, 15), cv::FILLED);
            cv::putText(annotated, label, {left + 3, bottom}, cv::FONT_HERSHEY_SIMPLEX,
                .55, cv::Scalar(40, 230, 80), 1, cv::LINE_AA);
        }
        cv::rectangle(annotated, {0, 0}, {size.width, 25}, cv::Scalar(15, 15, 15), cv::FILLED);
        cv::putText(annotated, camera.substr(0, 24) + " | LOCAL TRACK IDs | frame " + std::to_string(frames),
            {8, 18}, cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        writer.write(annotated);
        if (frames == 0 || samples.size() > preview_count) {
            vision::save_image(output / "preview.jpg", annotated); preview_count = samples.size();
        }
        ++frames;
        if (frames % 50 == 0) std::cout << "processed_frames=" << frames << std::endl;
        if (frames >= max_frames) { limit = true; break; }
        if (!capture.read(image)) break;
    }
    collector.finish(limit ? "frame_limit" : "end_of_input");
    writer.release(); csv.close();
    if (!csv) throw std::runtime_error("Cannot write observations");
    Json entries = Json::array();
    std::size_t eligible = 0;
    for (const auto& r : collector.results()) {
        const bool accepted = r.closure == "missing_timeout" && r.observations >= static_cast<std::size_t>(minimum);
        auto record = item(r.tracklet);
        record["observations"] = r.observations;
        record["closure"] = r.closure;
        record["eligible"] = accepted;
        eligible += accepted;
        entries.push_back(std::move(record));
    }
    save(output / "tracklets.json", {{"version", 1}, {"complete", true}, {"time_basis", "relative_cfr_estimate_ms"},
        {"clock_verified", false}, {"input", utf8(fs::absolute(input).lexically_normal())}, {"input_sha256", input_hash},
        {"detector_sha256", model_hash}, {"embedding_space", embedder.space_id()}, {"config", j},
        {"fps", fps}, {"processed_frames", frames}, {"person_observations", observations},
        {"eligible_tracklets", eligible}, {"tracklets", entries},
        {"warning", "Local IoU IDs are not person identities. Timeout is not a verified camera exit. EOF/decode stop and frame-limit tracks are censored. CFR estimates are not synchronized camera timestamps."}});
    std::cout << "frames=" << frames << " observations=" << observations << " tracklets=" << entries.size()
              << " eligible_closed=" << eligible << "\n";
    return 0;
}
// Explicit operator-provided clock mapping. Does not estimate or verify clocks.
int prepare(const fs::path& manifest, const fs::path& output) {
    auto j = read(manifest);
    keys(j, {"version", "time_basis", "clock_evidence", "sources", "cameras", "transitions", "min_similarity", "ambiguity_margin"});
    number(j.at("version"), 1, 1);
    if (j.at("time_basis") != "synchronized_ms" || !j.at("clock_evidence").is_string() ||
        j.at("clock_evidence").get<std::string>().empty() || j.at("clock_evidence").get<std::string>().size() > 4096)
        throw std::invalid_argument("Explicit synchronized_ms declaration and clock evidence are required");
    if (!j.at("sources").is_array() || j.at("sources").size() < 2 || j.at("sources").size() > 16)
        throw std::invalid_argument("Expected 2..16 distinct camera exports");
    MultiCameraConfig config;
    config.cameras = j.at("cameras").get<std::vector<std::string>>();
    config.min_similarity = j.at("min_similarity").get<double>();
    config.ambiguity_margin = j.at("ambiguity_margin").get<double>();
    if (!j.at("transitions").is_array() || j.at("transitions").size() > 240)
        throw std::invalid_argument("Invalid transitions");
    for (const auto& edge : j.at("transitions")) {
        keys(edge, {"from", "to", "min_gap_ms", "max_gap_ms"});
        config.transitions.push_back({edge.at("from").get<std::string>(), edge.at("to").get<std::string>(),
            number(edge.at("min_gap_ms"), 1, 86400000), number(edge.at("max_gap_ms"), 1, 86400000)});
    }
    std::vector<CameraTracklet> tracks;
    Json provenance = Json::array();
    std::set<std::string> source_cameras;
    const auto base = fs::absolute(manifest).parent_path();
    for (const auto& source : j.at("sources")) {
        keys(source, {"file", "origin_ms"});
        const auto origin = number(source.at("origin_ms"), 0, std::numeric_limits<std::int64_t>::max());
        const auto file = base / fs::u8path(source.at("file").get<std::string>());
        const auto e = read(file);
        number(e.at("version"), 1, 1);
        if (e.at("complete") != true || e.at("time_basis") != "relative_cfr_estimate_ms")
            throw std::invalid_argument("Unsupported or incomplete tracklet export");
        const auto camera = e.at("config").at("camera").get<std::string>();
        if (!source_cameras.insert(camera).second) throw std::invalid_argument("Camera export repeated");
        const auto space = e.at("embedding_space").get<std::string>();
        if (config.embedding_space.empty()) config.embedding_space = space;
        if (space != config.embedding_space) throw std::invalid_argument("Mixed embedding spaces");
        if (!e.at("tracklets").is_array() || e.at("tracklets").size() > 512)
            throw std::invalid_argument("Invalid exported tracklet count");
        for (const auto& r : e.at("tracklets")) {
            if (r.at("eligible") != true) continue;
            if (r.at("closure") != "missing_timeout" || r.at("camera") != camera || r.at("embedding_space") != space ||
                number(r.at("epoch"), 1, INT64_MAX) != number(e.at("config").at("epoch"), 1, INT64_MAX) ||
                number(r.at("observations"), 1, 1000000) < number(e.at("config").at("min_observations"), 1, 1000000))
                throw std::invalid_argument("Inconsistent eligible tracklet");
            CameraTracklet t{camera, static_cast<std::uint64_t>(number(r.at("epoch"), 1, INT64_MAX)),
                static_cast<std::uint64_t>(number(r.at("local_id"), 1, INT64_MAX)),
                number(r.at("start_ms"), 0, INT64_MAX - origin) + origin,
                number(r.at("end_ms"), 0, INT64_MAX - origin) + origin, space, r.at("embedding").get<std::vector<float>>()};
            tracks.push_back(std::move(t));
            if (tracks.size() > 512) throw std::invalid_argument("Association batch exceeds 512");
        }
        provenance.push_back({{"file", utf8(fs::absolute(file).lexically_normal())}, {"sha256", hash(file)}, {"origin_ms", origin}});
    }
    if (source_cameras != std::set<std::string>(config.cameras.begin(), config.cameras.end()))
        throw std::invalid_argument("Camera list must match export cameras");
    // The exact same core validates the candidate batch before emitting CLI input.
    const auto validation = associate_cameras(tracks, config);
    Json records = Json::array();
    for (const auto& t : tracks) records.push_back(item(t));
    Json cfg{{"cameras", config.cameras}, {"transitions", j.at("transitions")}, {"embedding_space", config.embedding_space},
        {"min_similarity", config.min_similarity}, {"ambiguity_margin", config.ambiguity_margin}};
    fresh(output);
    save(output / "provenance.json", {{"version", 1}, {"sources", provenance}, {"clock_evidence", j.at("clock_evidence")},
        {"clock_verified_by_software", false}, {"warning", "Origin offsets and CFR clock suitability are operator assertions, not measured synchronization."}});
    save(output / "association-input.json", {{"version", 1}, {"time_basis", "synchronized_ms"}, {"config", cfg}, {"tracklets", records}});
    std::cout << "prepared_tracklets=" << tracks.size() << "\n";
    return 0;
}
int run(const std::vector<fs::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "aegisvision_tracklets CONFIG.json NEW_OUTPUT_DIR\n"
                         "aegisvision_tracklets --prepare-association CLOCK_MAP.json NEW_OUTPUT_DIR\n"
                         "CPU YOLO + OSNet person tracklets. Local CFR timing; censored tracks are excluded from handoffs.\n";
            return 0;
        }
        if (args.size() == 4 && args[1] == "--prepare-association") return prepare(args[2], fs::absolute(args[3]));
        if (args.size() == 3) return extract(args[1], fs::absolute(args[2]));
        throw std::invalid_argument("Use --help for usage");
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
} // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv, argv + argc));
}
