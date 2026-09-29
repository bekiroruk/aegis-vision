#pragma once

#include "aegisvision/contracts.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace aegisvision {

class PayloadDetector final : public IDetector {
public:
    [[nodiscard]] std::vector<Detection> detect(const Frame& frame) override {
        return frame.candidate_detections;
    }
};

class HashEmbedder final : public IEmbedder {
public:
    explicit HashEmbedder(std::size_t dimension = 8);
    [[nodiscard]] std::vector<float> embed_image(
        const Frame& frame, const Detection& detection) override;
    [[nodiscard]] std::vector<float> embed_text(std::string_view text) override;

private:
    [[nodiscard]] std::vector<float> hash(std::string_view value) const;
    std::size_t dimension_;
};

}  // namespace aegisvision

