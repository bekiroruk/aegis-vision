#pragma once

#include "aegisvision/contracts.hpp"

#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>

namespace aegisvision {
// Experimental augmentation is opt-in; deployment defaults stay unchanged.
enum class ReIdAugmentation { None, HorizontalFlipMean };

// Pinned OSNet x0.25/x1.0, MSMT17-combineall, CPU/FP32 OpenCV DNN. The immutable bundle
// contains manifest.json and model.onnx. One instance is not thread-safe.
// Person appearance only: neither language embeddings nor cross-camera/global
// identity guarantees are provided by this encoder.
class ReIdEmbedder final : public IEmbedder {
public:
    explicit ReIdEmbedder(const std::filesystem::path& bundle,
                         ReIdAugmentation augmentation = ReIdAugmentation::None);
    ~ReIdEmbedder();
    [[nodiscard]] std::vector<float> embed_image(const Frame&, const Detection&) override;
    [[nodiscard]] std::vector<float> embed_text(std::string_view) override;
    [[nodiscard]] const std::string& space_id() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Owned float32 NCHW [1,3,256,128]: INTER_LINEAR resize of an 8-bit BGR crop,
// RGB conversion, division by 255, then ImageNet channel mean/std normalization.
[[nodiscard]] cv::Mat prepare_reid(const cv::Mat& crop);
// Validates the owned BGR buffer and person detection, then clips/crops its box.
// Separate from model execution so malformed inputs are testable without weights.
[[nodiscard]] cv::Mat prepare_reid(const Frame&, const Detection&);
// Exactly 512 finite components, nonzero norm; returns an L2-normalized copy.
[[nodiscard]] std::vector<float> normalize_reid(std::vector<float> embedding);
// Flip only the width axis of a finite contiguous [1,3,256,128] float tensor.
[[nodiscard]] cv::Mat flip_reid_input(const cv::Mat& input);
// Equal-weight mean of unit features, normalized again; zero cancellation fails.
[[nodiscard]] std::vector<float> mean_reid_features(std::vector<float> first, std::vector<float> second);

}  // namespace aegisvision
