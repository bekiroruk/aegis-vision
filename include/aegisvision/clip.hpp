#pragma once
#include "aegisvision/contracts.hpp"
#include <array>
#include <filesystem>
#include <memory>

namespace aegisvision {
struct ClipTokens {
    std::array<std::int64_t, 77> ids{};
    std::array<std::int64_t, 77> mask{};
};
class ClipTokenizer {
public:
    explicit ClipTokenizer(const std::filesystem::path& tokenizer_json);
    ~ClipTokenizer();
    [[nodiscard]] ClipTokens encode(std::string_view text) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// CPU FP32 CLIP ViT-B/32 bundle produced by scripts/export_clip.py. Not thread-safe.
class ClipEmbedder final : public IEmbedder {
public:
    explicit ClipEmbedder(const std::filesystem::path& bundle);
    ~ClipEmbedder();
    [[nodiscard]] std::vector<float> embed_image(const Frame&, const Detection&) override;
    [[nodiscard]] std::vector<float> embed_text(std::string_view) override;
    [[nodiscard]] const std::string& space_id() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Rejects empty/non-finite/zero vectors; returns L2-normalized copy.
[[nodiscard]] std::vector<float> normalize_embedding(std::vector<float> vector);
}
