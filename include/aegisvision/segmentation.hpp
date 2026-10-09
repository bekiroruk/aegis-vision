#pragma once
#include "aegisvision/yolo.hpp"

namespace aegisvision::vision {
struct InstanceMask {
    Detection detection;
    cv::Rect region; // Original image coordinates, half-open pixel bounds.
    cv::Mat mask;    // Owned CV_8UC1 ROI, 0 or 255; zero outside region.
};
class ISegmenter {
public:
    virtual ~ISegmenter() = default;
    virtual std::vector<InstanceMask> segment(const cv::Mat& bgr) = 0;
};
// Static YOLOv8-seg: [1,4+C+32,N] and [1,32,S/4,S/4]. CPU FP32.
std::vector<InstanceMask> decode_segmentation(const cv::Mat& predictions, const cv::Mat& prototypes,
    const Letterbox& transform, const std::vector<std::string>& labels, const YoloConfig& config);
cv::Mat paint_masks(const cv::Mat& bgr, const std::vector<InstanceMask>& instances);
class YoloSegmenter final : public ISegmenter {
public:
    explicit YoloSegmenter(const std::filesystem::path& model, YoloConfig config = {640,.35F,.45F,100});
    std::vector<InstanceMask> segment(const cv::Mat& bgr) override;
private:
    cv::dnn::Net net_;
    YoloConfig config_;
};
} // namespace aegisvision::vision
