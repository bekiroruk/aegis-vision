#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace aegisvision {
struct CrossViewPerson {
    std::uint64_t identity{}; // Ground truth, used only for scoring after matching.
    std::vector<float> embedding;
};
struct CrossViewDecision {
    std::optional<std::size_t> best_gallery;
    std::optional<double> similarity;
    bool accepted{}, has_counterpart{}, correct{}, rank1_correct{};
    std::string reason;
};
// Same synchronized instant, different cameras. No temporal handoff or tracking.
// <=512 unique identities per view, finite nonzero 512-D features.
// Label IDs never enter appearance matching. Rank-1 ties count as incorrect.
[[nodiscard]] std::vector<CrossViewDecision> evaluate_crossview(
    const std::vector<CrossViewPerson>& queries, const std::vector<CrossViewPerson>& gallery,
    double min_similarity = .85, double ambiguity_margin = .05);
}
