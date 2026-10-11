#pragma once
#include "aegisvision/ocr.hpp"
#include <memory>
#include <span>
namespace aegisvision::vision {
RecognizedText decode_unicode_ctc(std::span<const float> probabilities,std::size_t steps,
                                  const std::vector<std::string>& dictionary);
// PP-OCRv3 EN detector + official PP-OCRv6 tiny multilingual recognizer.
// Pinned CPU ONNX Runtime backend; sequential worker ownership.
class PpocrMultilingual final : public IOcr {
public:
    explicit PpocrMultilingual(const std::filesystem::path& bundle,OcrConfig config={});
    ~PpocrMultilingual() override;
    std::vector<TextRegion> read(const cv::Mat& bgr,const std::atomic_bool* cancel=nullptr) override;
    bool unicode_output() const override {return true;}
    const std::string& signature() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
