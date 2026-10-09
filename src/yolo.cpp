#include "aegisvision/yolo.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace aegisvision::vision {
namespace {
void validate(const YoloConfig& config) {
    if (config.input_size < 32 || config.input_size > 2048 || config.input_size % 32 != 0 ||
        !std::isfinite(config.confidence_threshold) || config.confidence_threshold <= 0 ||
        config.confidence_threshold > 1 || !std::isfinite(config.nms_iou_threshold) ||
        config.nms_iou_threshold <= 0 || config.nms_iou_threshold > 1 ||
        config.max_detections == 0 || config.max_detections > 3000) {
        throw std::invalid_argument("Invalid YOLO configuration");
    }
}

void require_bgr(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("YOLO requires a nonempty 8-bit BGR image");
    }
}
}

std::vector<std::string> coco_labels() {
    return {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};
}

Frame image_frame(const cv::Mat& bgr, std::string frame_id, std::string source_id,
    std::int64_t timestamp_ms) {
    require_bgr(bgr);
    const auto contiguous = bgr.clone();
    auto buffer = std::make_shared<ImageBuffer>();
    buffer->width = bgr.cols;
    buffer->height = bgr.rows;
    buffer->stride = static_cast<std::size_t>(bgr.cols) * 3;
    buffer->pixels.assign(contiguous.data, contiguous.data + contiguous.total() * 3);
    return {std::move(frame_id), std::move(source_id), timestamp_ms, {}, std::move(buffer)};
}

Letterbox prepare_yolo(const cv::Mat& bgr, int input_size) {
    require_bgr(bgr);
    YoloConfig config;
    config.input_size = input_size;
    validate(config);
    const double scale = std::min(static_cast<double>(input_size) / bgr.cols,
        static_cast<double>(input_size) / bgr.rows);
    const int width = std::clamp(static_cast<int>(std::round(bgr.cols * scale)), 1, input_size);
    const int height = std::clamp(static_cast<int>(std::round(bgr.rows * scale)), 1, input_size);
    Letterbox result;
    result.original_size = bgr.size();
    result.scale_x = static_cast<double>(width) / bgr.cols;
    result.scale_y = static_cast<double>(height) / bgr.rows;
    result.pad_left = (input_size - width) / 2;
    result.pad_top = (input_size - height) / 2;
    cv::Mat resized, padded(input_size, input_size, CV_8UC3, cv::Scalar::all(114));
    cv::resize(bgr, resized, {width, height}, 0, 0, cv::INTER_LINEAR);
    resized.copyTo(padded(cv::Rect(result.pad_left, result.pad_top, width, height)));
    result.blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, {}, {}, true, false, CV_32F);
    return result;
}

std::vector<Detection> decode_yolo(const cv::Mat& output, const Letterbox& transform,
    const std::vector<std::string>& labels, const YoloConfig& config, bool include_anchor_index) {
    validate(config);
    if (labels.empty() || labels.size() > 10000 ||
        std::any_of(labels.begin(), labels.end(), [](const auto& label) { return label.empty(); })) {
        throw std::invalid_argument("YOLO class labels must be nonempty");
    }
    if (output.type() != CV_32F || output.dims != 3 || output.size[0] != 1 ||
        output.size[1] != static_cast<int>(labels.size()) + 4 || output.size[2] < 1 ||
        !output.isContinuous()) {
        throw std::invalid_argument("Expected YOLOv8 float32 output [1,4+classes,anchors] without NMS");
    }
    if (transform.original_size.width <= 0 || transform.original_size.height <= 0 ||
        !std::isfinite(transform.scale_x) || !std::isfinite(transform.scale_y) ||
        transform.scale_x <= 0 || transform.scale_y <= 0) {
        throw std::invalid_argument("Invalid letterbox transform");
    }
    const int count = output.size[2];
    const auto value = [&](int channel, int anchor) { return output.ptr<float>(0, channel)[anchor]; };
    struct Candidate { Detection detection; int class_id; };
    std::vector<Candidate> candidates;
    for (int i = 0; i < count; ++i) {
        float score = -1;
        int class_id = -1;
        for (int c = 0; c < static_cast<int>(labels.size()); ++c) {
            const float probability = value(c + 4, i);
            if (std::isfinite(probability) && probability >= 0 && probability <= 1 && probability > score) {
                score = probability;
                class_id = c;
            }
        }
        if (class_id < 0 || score < config.confidence_threshold) continue;
        const double cx = value(0, i), cy = value(1, i), w = value(2, i), h = value(3, i);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
            w <= 0 || h <= 0) continue;
        const auto map_x = [&](double x) { return static_cast<float>(std::clamp(
            (x - transform.pad_left) / transform.scale_x, 0.0, static_cast<double>(transform.original_size.width))); };
        const auto map_y = [&](double y) { return static_cast<float>(std::clamp(
            (y - transform.pad_top) / transform.scale_y, 0.0, static_cast<double>(transform.original_size.height))); };
        BoundingBox box(map_x(cx - w / 2), map_y(cy - h / 2), map_x(cx + w / 2), map_y(cy + h / 2));
        if (box.area() <= 0) continue;
        candidates.push_back({{box, labels[class_id], score, {}, {{"class_id", std::to_string(class_id)}}}, class_id});
        if (include_anchor_index) candidates.back().detection.attributes["anchor_index"] = std::to_string(i);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.detection.score > b.detection.score;
    });
    // Bound the quadratic NMS work. Same-class overlaps only; other classes survive.
    if (candidates.size() > 3000) candidates.resize(3000, candidates.front());
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
        const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const auto& other) {
            return candidate.class_id == other.class_id &&
                candidate.detection.bbox.iou(other.detection.bbox) > config.nms_iou_threshold;
        });
        if (!suppressed) kept.push_back(candidate);
        if (kept.size() >= config.max_detections) break;
    }
    std::vector<Detection> result;
    for (auto& candidate : kept) result.push_back(std::move(candidate.detection));
    return result;
}

YoloDetector::YoloDetector(const std::filesystem::path& model, YoloConfig config,
    std::vector<std::string> labels) : config_(config), labels_(std::move(labels)) {
    validate(config_);
    if (labels_.empty()) throw std::invalid_argument("Class labels are required");
    std::ifstream input(model, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open ONNX model; see docs/yolo.md for export instructions");
    const auto length = input.tellg();
    if (length <= 0 || length > 1024LL * 1024 * 1024) throw std::runtime_error("Invalid ONNX model size");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("Cannot read ONNX model");
    net_ = cv::dnn::readNetFromONNX(bytes);
    if (net_.empty()) throw std::runtime_error("ONNX network is empty");
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> YoloDetector::detect(const Frame& frame) {
    if (!frame.image) throw std::invalid_argument("Frame has no image pixels");
    const auto& image = *frame.image;
    if (image.width <= 0 || image.height <= 0 ||
        image.stride < static_cast<std::size_t>(image.width) * 3 ||
        image.stride > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(image.height) ||
        image.pixels.size() < image.stride * static_cast<std::size_t>(image.height)) {
        throw std::invalid_argument("Invalid frame image buffer");
    }
    // OpenCV Mat has no const data view. prepare_yolo only reads this view and owns its output.
    const cv::Mat view(image.height, image.width, CV_8UC3,
        const_cast<std::uint8_t*>(image.pixels.data()), image.stride);
    const auto input = prepare_yolo(view, config_.input_size);
    net_.setInput(input.blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.size() != 1) throw std::runtime_error("Expected one YOLO detect output");
    return decode_yolo(outputs.front(), input, labels_, config_);
}
} // namespace aegisvision::vision
