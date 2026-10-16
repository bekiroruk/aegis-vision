#include "aegisvision/redis_image.hpp"
#include "aegisvision/vision.hpp"
#include <picosha2.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace aegisvision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::vector<unsigned char> bytes(const fs::path& path, std::size_t limit) {
    if (!fs::is_regular_file(path)) throw std::invalid_argument("Expected a regular file");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    if (!input || size <= 0 || size > static_cast<std::streamoff>(limit))
        throw std::invalid_argument("File missing, empty or larger than allowed");
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("File read failed or file changed during read");
    return data;
}
vision::YoloConfig config() {
    vision::YoloConfig value;
    value.max_detections = 100; // Keeps the complete JSON result within the queue's 64 KiB bound.
    return value;
}
void require(bool ok) { if (!ok) throw std::invalid_argument("Invalid detect_image request"); }
}
RedisImageWorker::RedisImageWorker(const fs::path& model, const fs::path& media_root)
    : root_(fs::canonical(media_root)), model_bytes_(bytes(model, 256U * 1024 * 1024)),
      model_sha_(picosha2::hash256_hex_string(model_bytes_)) {
    if (!fs::is_directory(root_)) throw std::invalid_argument("Media root must be a directory");
    const auto root_utf8 = root_.generic_u8string();
    const auto c = config();
    const Json contract{{"version", "detect-image-v1"}, {"model_sha256", model_sha_},
        {"media_root", std::string(root_utf8.begin(), root_utf8.end())}, {"opencv", CV_VERSION},
        {"backend", "opencv-dnn-cpu"}, {"input_size", c.input_size},
        {"confidence", c.confidence_threshold}, {"nms", c.nms_iou_threshold},
        {"max_detections", c.max_detections}, {"labels", vision::coco_labels()}};
    context_ = "detect-image-v1-" + picosha2::hash256_hex_string(contract.dump());
}
std::vector<unsigned char> RedisImageWorker::read_image(const std::string& relative) const {
    require(!relative.empty() && relative.size() <= 512 && relative.find('\0') == std::string::npos);
    // Use portable slash-separated paths; reject Windows drives/ADS and backslashes on all hosts.
    require(relative.find(':') == std::string::npos && relative.find('\\') == std::string::npos);
    const auto path = fs::u8path(relative);
    require(!path.is_absolute() && !path.has_root_path());
    for (const auto& part : path) require(part != ".." && part != ".");
    const auto resolved = fs::canonical(root_ / path);
    const auto inside = resolved.lexically_relative(root_);
    require(!inside.empty() && !inside.is_absolute());
    for (const auto& part : inside) require(part != "..");
    // Trusted local directory: no concurrent symlink swaps or hostile file writers.
    return bytes(resolved, 16U * 1024 * 1024);
}
Json RedisImageWorker::request(const std::string& image) const {
    return {{"type", "detect_image"}, {"image", image},
        {"sha256", picosha2::hash256_hex_string(read_image(image))}};
}
cv::Mat RedisImageWorker::decode(const Json& payload) const {
    require(payload.is_object() && payload.size() == 3 && payload.value("type", "") == "detect_image"
        && payload.contains("image") && payload.at("image").is_string()
        && payload.contains("sha256") && payload.at("sha256").is_string());
    const auto expected = payload.at("sha256").get<std::string>();
    require(expected.size() == 64 && expected.find_first_not_of("0123456789abcdef") == std::string::npos);
    const auto encoded = read_image(payload.at("image").get<std::string>());
    if (picosha2::hash256_hex_string(encoded) != expected) throw std::invalid_argument("Image SHA256 changed");
    // This is a trusted-media worker, not a sandbox for hostile image codecs.
    const auto image = cv::imdecode(encoded, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    require(!image.empty() && image.total() <= 16000000);
    return image;
}
Json RedisImageWorker::analyze(const Json& payload, const StopCheck& cancelled) {
    if (cancelled()) throw std::runtime_error("Job no longer owned");
    const auto image = decode(payload);
    const auto expected = payload.at("sha256").get<std::string>();
    if (cancelled()) throw std::runtime_error("Job no longer owned");
    if (!detector_) {
        detector_ = std::make_unique<vision::YoloDetector>(model_bytes_, config());
        model_bytes_.clear(); model_bytes_.shrink_to_fit();
    }
    if (cancelled()) throw std::runtime_error("Job no longer owned");
    const auto start = std::chrono::steady_clock::now();
    const auto detections = detector_->detect(vision::image_frame(image, "redis-image", "local-media"));
    if (cancelled()) throw std::runtime_error("Job no longer owned");
    Json result{{"type", "detect_image"}, {"context", context_}, {"model_sha256", model_sha_},
        {"image_sha256", expected}, {"image", payload.at("image")}, {"width", image.cols}, {"height", image.rows},
        {"inference_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()},
        {"detections", Json::array()}};
    for (const auto& detection : detections) {
        const auto& b = detection.bbox;
        result["detections"].push_back({{"label", detection.label}, {"confidence", detection.score},
            {"class_id", std::stoi(detection.attributes.at("class_id"))}, {"xyxy", {b.x1, b.y1, b.x2, b.y2}}});
    }
    return result;
}
std::vector<unsigned char> RedisImageWorker::preview(const Json& payload, const Json& result) const {
    require(result.at("context") == context_ && result.at("image_sha256") == payload.at("sha256"));
    const auto image = decode(payload); // Recheck bytes before showing an old job's boxes.
    require(result.at("width") == image.cols && result.at("height") == image.rows);
    require(result.at("detections").is_array() && result.at("detections").size() <= 100);
    std::vector<Detection> detections;
    for (const auto& d : result.at("detections")) {
        const auto& b = d.at("xyxy");
        require(b.is_array() && b.size() == 4);
        const float x1=b.at(0), y1=b.at(1), x2=b.at(2), y2=b.at(3), score=d.at("confidence");
        require(x1>=0 && y1>=0 && x2>x1 && y2>y1 && x2<=image.cols && y2<=image.rows && score>=0 && score<=1);
        const auto label = d.at("label").get<std::string>(); require(label.size()<=64);
        detections.push_back({{x1,y1,x2,y2},label,score,{},{}});
    }
    auto annotated = vision::annotate(image, detections);
    const double scale = std::min(1., 960. / std::max(image.cols, image.rows));
    if (scale < 1) cv::resize(annotated, annotated, {}, scale, scale, cv::INTER_AREA);
    std::vector<unsigned char> encoded;
    if (!cv::imencode(".jpg", annotated, encoded, {cv::IMWRITE_JPEG_QUALITY,80}) || encoded.size()>1024*1024)
        throw std::runtime_error("Preview encoding failed");
    return encoded;
}
}
