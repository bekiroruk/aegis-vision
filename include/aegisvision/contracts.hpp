#pragma once

#include "aegisvision/domain.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace aegisvision {

class IDetector {
public:
    virtual ~IDetector() = default;
    [[nodiscard]] virtual std::vector<Detection> detect(const Frame& frame) = 0;
};

class ITracker {
public:
    virtual ~ITracker() = default;
    [[nodiscard]] virtual std::vector<Track> update(const std::vector<Detection>& detections) = 0;
};

class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    [[nodiscard]] virtual std::vector<float> embed_image(
        const Frame& frame, const Detection& detection) = 0;
    [[nodiscard]] virtual std::vector<float> embed_text(std::string_view text) = 0;
};

class ITextExtractor {
public:
    virtual ~ITextExtractor() = default;
    [[nodiscard]] virtual std::vector<std::string> extract(
        const Frame& frame, const Detection& detection) = 0;
};

class IVectorStore {
public:
    virtual ~IVectorStore() = default;
    virtual void upsert(
        std::string item_id,
        std::vector<float> vector,
        std::map<std::string, std::string> metadata) = 0;
    [[nodiscard]] virtual std::vector<SearchResult> search(
        const std::vector<float>& vector, std::size_t limit = 10) const = 0;
};

}  // namespace aegisvision

