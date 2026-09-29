#include "aegisvision/adapters.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace aegisvision {

HashEmbedder::HashEmbedder(const std::size_t dimension) : dimension_(dimension) {
    if (dimension == 0) {
        throw std::invalid_argument("Embedding dimension must be positive");
    }
}

std::vector<float> HashEmbedder::embed_image(
    const Frame& frame, const Detection& detection) {
    return hash(frame.frame_id + ":" + detection.label + ":" +
        std::to_string(detection.bbox.x1) + ":" + std::to_string(detection.bbox.y1));
}

std::vector<float> HashEmbedder::embed_text(const std::string_view text) {
    return hash(text);
}

std::vector<float> HashEmbedder::hash(const std::string_view value) const {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t state = offset;
    for (const auto character : value) {
        state ^= static_cast<unsigned char>(character);
        state *= prime;
    }

    std::vector<float> embedding;
    embedding.reserve(dimension_);
    for (std::size_t index = 0; index < dimension_; ++index) {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        embedding.push_back(static_cast<float>((state % 1000U) + 1U) / 1000.0F);
    }
    return embedding;
}

}  // namespace aegisvision

