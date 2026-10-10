#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <array>
#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace aegisvision::vision {
using TextQuad = std::array<cv::Point2f,4>; // BL, TL, TR, BR; original pixels.
struct RecognizedText {
    std::string text; // This backend: lowercase ASCII a-z and digits only.
    float confidence{}; // Mean emitted-character softmax, NOT calibrated accuracy.
};
struct TextRegion {
    TextQuad polygon;
    float detection_confidence{};
    RecognizedText recognition;
};
struct OcrConfig {
    std::size_t max_regions{64};
    int timeout_ms{10000}; // Cooperative: checked between/after DNN calls.
};
RecognizedText decode_english_ctc(const cv::Mat& logits); // [T,1,37], blank=0.
cv::Mat rectify_text(const cv::Mat& bgr,const TextQuad& polygon);
std::size_t ascii_edit_distance(const std::string& reference,const std::string& prediction);
class IOcr {
public:
    virtual ~IOcr() = default;
    virtual std::vector<TextRegion> read(const cv::Mat& bgr,const std::atomic_bool* cancel=nullptr) = 0;
};
// Sequential worker-owned models, CPU FP32; no shared concurrent calls.
class PpocrCrnn final : public IOcr {
public:
    PpocrCrnn(const std::filesystem::path& detector,const std::filesystem::path& recognizer,OcrConfig config={});
    std::vector<TextRegion> read(const cv::Mat& bgr,const std::atomic_bool* cancel=nullptr) override;
private:
    OcrConfig config_;
    cv::dnn::TextDetectionModel_DB detector_;
    cv::dnn::Net recognizer_;
};
} // namespace aegisvision::vision
