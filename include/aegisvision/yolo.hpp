#pragma once

#include "aegisvision/contracts.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

namespace aegisvision::vision {

struct YoloConfig {
    int input_size{640};
    float confidence_threshold{0.35F};
    float nms_iou_threshold{0.45F};
    std::size_t max_detections{300};
};

struct Letterbox {
    cv::Mat blob; // float32 NCHW RGB, [0,1], padding=114/255
    cv::Size original_size;
    double scale_x{};
    double scale_y{};
    int pad_left{};
    int pad_top{};
};

[[nodiscard]] std::vector<std::string> coco_labels();
[[nodiscard]] Frame image_frame(const cv::Mat& bgr, std::string frame_id,
    std::string source_id, std::int64_t timestamp_ms = 0);
[[nodiscard]] Letterbox prepare_yolo(const cv::Mat& bgr, int input_size = 640);
// Only YOLOv8 detect export: [1, 4+classes, anchors], xywh pixels + class probabilities.
// No objectness channel, embedded NMS, segmentation or end-to-end output support.
[[nodiscard]] std::vector<Detection> decode_yolo(const cv::Mat& output,
    const Letterbox& transform, const std::vector<std::string>& labels, const YoloConfig& config);

// Each instance owns a mutable OpenCV network; use one instance per worker.
class YoloDetector final : public IDetector {
public:
    explicit YoloDetector(const std::filesystem::path& model,
        YoloConfig config = {}, std::vector<std::string> labels = coco_labels());
    [[nodiscard]] std::vector<Detection> detect(const Frame& frame) override;
private:
    cv::dnn::Net net_;
    YoloConfig config_;
    std::vector<std::string> labels_;
};
} // namespace aegisvision::vision
