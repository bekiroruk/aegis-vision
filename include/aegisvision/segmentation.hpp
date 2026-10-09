#pragma once
#include "aegisvision/yolo.hpp"
#include "aegisvision/tracking.hpp"

namespace aegisvision::vision {
struct InstanceMask {
    Detection detection;
    cv::Rect region; // Original image coordinates, half-open pixel bounds.
    cv::Mat mask;    // Owned CV_8UC1 ROI, 0 or 255; zero outside region.
    std::uint64_t track_id{}; // 0=untracked; never the rank in a frame.
};
class ISegmenter {
public:
    virtual ~ISegmenter() = default;
    virtual std::vector<InstanceMask> segment(const cv::Mat& bgr) = 0;
};
// Sequential local frames only; fail before advancing tracker on invalid masks.
class SegmentationPipeline {
public:
    SegmentationPipeline(ISegmenter& segmenter, bool track = false,
        float match_iou = .30F, std::uint32_t max_missed = 20)
        : segmenter_(segmenter), track_(track), tracker_(match_iou,max_missed) {}
    std::vector<InstanceMask> analyze(const cv::Mat& bgr);
private:
    ISegmenter& segmenter_;
    bool track_;
    IoUTracker tracker_;
    cv::Size size_;
};
// COCO uncompressed RLE: column-major, initial zero run; full image coordinates.
std::vector<std::uint32_t> mask_rle(const InstanceMask& instance, cv::Size image_size);
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
