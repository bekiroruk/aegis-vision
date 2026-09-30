#pragma once
#include <cstddef>
#include <utility>
#include <vector>

namespace aegisvision {
// Maximum-total-weight one-to-one assignment. Weights in [0,1]; -1 means forbidden.
// Rows/columns may remain unmatched. Zero-weight edges are omitted from the result.
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> assign_max_weight(
    const std::vector<std::vector<double>>& weights);
}
