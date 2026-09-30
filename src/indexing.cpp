#include "aegisvision/indexing.hpp"
#include "aegisvision/vision.hpp"
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace aegisvision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string utf8(const fs::path& path) {
    const auto value = path.generic_u8string();
    return {value.begin(), value.end()};
}
std::string digest(const Json& value) { return picosha2::hash256_hex_string(value.dump()); }
void check_cancelled(const IndexCancellation& cancelled) {
    if (cancelled && cancelled()) throw IndexCancelled();
}
std::string file_hash(const fs::path& path, const IndexCancellation& cancelled = {}) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot hash input: " + utf8(path));
    picosha2::hash256_one_by_one hash;
    std::array<char, 65536> buffer{};
    while (input) {
        check_cancelled(cancelled);
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        hash.process(buffer.begin(), buffer.begin() + input.gcount());
    }
    if (input.bad()) throw std::runtime_error("Failed reading input: " + utf8(path));
    hash.finish();
    return picosha2::get_hash_hex_string(hash);
}
bool image_extension(const fs::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto* supported : {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"})
        if (extension == supported) return true;
    return false;
}
std::string bbox_json(const BoundingBox& box) {
    return Json::array({box.x1, box.y1, box.x2, box.y2}).dump();
}
void validate_detection(const Detection& detection, const cv::Mat& image) {
    const auto& b = detection.bbox;
    if (detection.label.empty() || !std::isfinite(detection.score) || detection.score < 0 || detection.score > 1 ||
        !std::isfinite(b.x1) || !std::isfinite(b.y1) || !std::isfinite(b.x2) || !std::isfinite(b.y2) ||
        b.x1 < 0 || b.y1 < 0 || b.x2 > image.cols || b.y2 > image.rows || b.x2 <= b.x1 || b.y2 <= b.y1)
        throw std::invalid_argument("Detector returned an invalid label, score or crop");
}
}

std::string yolo_index_signature(const std::filesystem::path& model, const vision::YoloConfig& config) {
    return digest(Json::array({"yolov8-opencv-coco80-letterbox-v1", file_hash(model),
        config.input_size, config.confidence_threshold, config.nms_iou_threshold, config.max_detections}));
}

IndexSummary index_directory(const std::filesystem::path& directory, IEmbedder& embedder,
    IVectorStore& store, const DirectoryIndexConfig& config, const IndexProgress& progress,
    const IndexCancellation& cancelled) {
    if (config.max_images < 1 || config.max_images > 100000)
        throw std::invalid_argument("Directory max_images must be 1..100000");
    if (!fs::is_directory(directory)) throw std::invalid_argument("Input must be a local directory");
    const auto root = fs::canonical(directory);
    IndexSummary summary;
    summary.source_id = digest(Json::array({"directory-v1", utf8(root)}));
    std::vector<fs::path> files;
    const auto collect = [&](const fs::directory_entry& entry) {
        check_cancelled(cancelled);
        const auto status = entry.symlink_status();
        if (fs::is_symlink(status)) { ++summary.skipped_entries; return; }
        if (fs::is_directory(status)) return;
        if (!fs::is_regular_file(status) || !image_extension(entry.path())) {
            ++summary.skipped_entries; return;
        }
        files.push_back(entry.path());
        if (files.size() > config.max_images)
            throw std::invalid_argument("Directory image limit exceeded before any writes");
    };
    if (config.recursive) {
        // Default iterator options do not follow directory symlinks.
        for (const auto& entry : fs::recursive_directory_iterator(root)) collect(entry);
    } else {
        for (const auto& entry : fs::directory_iterator(root)) collect(entry);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::invalid_argument("Directory has no supported image files");
    for (const auto& path : files) {
        try {
            check_cancelled(cancelled);
            const auto image = vision::load_image(path);
            const BoundingBox box{0, 0, static_cast<float>(image.cols), static_cast<float>(image.rows)};
            // Path identity updates a changed image in place and is independent of traversal order.
            const auto id = "image-" + digest(Json::array({"directory-image-v1", utf8(path)}));
            auto vector = embedder.embed_image(vision::image_frame(image, id, summary.source_id),
                {box, "image", 1, {}, {}});
            const auto hash = file_hash(path, cancelled);
            check_cancelled(cancelled);
            store.upsert(id, std::move(vector), {{"kind", "image"}, {"path", utf8(path)},
                {"source_id", summary.source_id}, {"sha256", hash}, {"bbox", bbox_json(box)}});
            ++summary.indexed_items;
            if (progress) progress(summary);
        } catch (const IndexCancelled&) { throw;
        } catch (const std::exception& error) {
            throw std::runtime_error("Indexing " + utf8(path) + " after " +
                std::to_string(summary.indexed_items) + " completed items: " + error.what());
        }
    }
    summary.stop_reason = "directory_complete";
    return summary;
}

IndexSummary index_video(const std::filesystem::path& input, IDetector& detector,
    IEmbedder& embedder, IVectorStore& store, const VideoIndexConfig& config, const IndexProgress& progress,
    const IndexCancellation& cancelled) {
    if (config.frame_stride < 1 || config.max_frames < 0 || config.detector_signature.empty() ||
        !std::isfinite(config.fallback_fps) || config.fallback_fps <= 0 || config.fallback_fps > 1000)
        throw std::invalid_argument("Invalid video indexing configuration");
    if (!fs::is_regular_file(input)) throw std::invalid_argument("Input must be an existing local video file");
    const auto path = fs::canonical(input);
    const auto content_hash = file_hash(path, cancelled);
    IndexSummary summary;
    summary.source_id = digest(Json::array({"video-crops-v1", content_hash, config.detector_signature}));
    cv::VideoCapture capture(utf8(path));
    if (!capture.isOpened()) throw std::runtime_error("Cannot open input video; check codec support");
    const double fps = capture.get(cv::CAP_PROP_FPS);
    summary.used_fallback_fps = !std::isfinite(fps) || fps <= 0 || fps > 1000;
    summary.source_fps = summary.used_fallback_fps ? config.fallback_fps : fps;
    cv::Mat image;
    while (!config.max_frames || summary.decoded_frames < static_cast<std::uint64_t>(config.max_frames)) {
        check_cancelled(cancelled);
        if (!capture.read(image) || image.empty()) {
            summary.stop_reason = "end_of_stream_or_decode_stop"; break;
        }
        if (image.type() != CV_8UC3) throw std::runtime_error("Expected an 8-bit BGR video frame");
        const auto frame_index = summary.decoded_frames++;
        if (frame_index % static_cast<std::uint64_t>(config.frame_stride) != 0) continue;
        ++summary.sampled_frames;
        const double timestamp = static_cast<double>(frame_index) * 1000.0 / summary.source_fps;
        if (timestamp >= static_cast<double>(std::numeric_limits<std::int64_t>::max()))
            throw std::runtime_error("Video timestamp overflow");
        const auto timestamp_ms = static_cast<std::int64_t>(std::llround(timestamp));
        const auto frame = vision::image_frame(image, std::to_string(frame_index), summary.source_id, timestamp_ms);
        auto detections = detector.detect(frame);
        for (const auto& detection : detections) validate_detection(detection, image);
        // Stable ordering also tolerates adapters that return the same detections in a different order.
        std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
            return std::tie(a.label, a.bbox.x1, a.bbox.y1, a.bbox.x2, a.bbox.y2, a.score) <
                std::tie(b.label, b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2, b.score);
        });
        for (std::size_t i = 0; i < detections.size(); ++i) {
            check_cancelled(cancelled);
            const auto& detection = detections[i];
            const auto id = "video-" + digest(Json::array({summary.source_id, frame_index, i}));
            auto vector = embedder.embed_image(frame, detection);
            check_cancelled(cancelled);
            store.upsert(id, std::move(vector), {{"kind", "video_crop"}, {"path", utf8(path)},
                {"source_id", summary.source_id}, {"sha256", content_hash},
                {"detector_signature", config.detector_signature}, {"frame_index", std::to_string(frame_index)},
                {"timestamp_ms", std::to_string(timestamp_ms)}, {"timestamp_basis", "frame_index/source_fps (CFR estimate)"},
                {"source_fps", Json(summary.source_fps).dump()}, {"used_fallback_fps", summary.used_fallback_fps ? "true" : "false"},
                {"bbox", bbox_json(detection.bbox)}, {"label", detection.label}, {"confidence", Json(detection.score).dump()}});
            ++summary.indexed_items;
            if (progress) progress(summary);
        }
        if (progress) progress(summary);
    }
    if (summary.decoded_frames == 0) throw std::runtime_error("Video has no decodable frames");
    if (summary.stop_reason.empty()) summary.stop_reason = "frame_limit";
    return summary;
}
}
