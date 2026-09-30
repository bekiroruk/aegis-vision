#include "aegisvision/assignment.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aegisvision {
std::vector<std::pair<std::size_t, std::size_t>> assign_max_weight(
    const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) return {};
    const std::size_t rows = weights.size(), columns = weights.front().size();
    for (const auto& row : weights) {
        if (row.size() != columns) throw std::invalid_argument("Assignment matrix must be rectangular");
        for (double value : row) {
            if (!std::isfinite(value) || (value != -1 && (value < 0 || value > 1))) {
                throw std::invalid_argument("Assignment weights must be -1 or in [0,1]");
            }
        }
    }
    if (columns == 0) return {};
    // One dummy column per row allows every row to be unmatched at zero reward.
    const std::size_t width = columns + rows;
    std::vector<double> u(rows + 1), v(width + 1);
    std::vector<std::size_t> p(width + 1), way(width + 1);
    for (std::size_t i = 1; i <= rows; ++i) {
        p[0] = i;
        std::size_t column = 0;
        std::vector<double> minimum(width + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(width + 1);
        do {
            used[column] = true;
            const std::size_t row = p[column];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t next = 0;
            for (std::size_t j = 1; j <= width; ++j) {
                if (used[j]) continue;
                const double cost = j <= columns ? 1.0 - weights[row - 1][j - 1] : 1.0;
                const double reduced = cost - u[row] - v[j];
                if (reduced < minimum[j]) { minimum[j] = reduced; way[j] = column; }
                if (minimum[j] < delta) { delta = minimum[j]; next = j; }
            }
            for (std::size_t j = 0; j <= width; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minimum[j] -= delta;
            }
            column = next;
        } while (p[column] != 0);
        do {
            const auto previous = way[column];
            p[column] = p[previous];
            column = previous;
        } while (column != 0);
    }
    std::vector<std::pair<std::size_t, std::size_t>> result;
    for (std::size_t j = 1; j <= columns; ++j) {
        if (p[j] != 0 && weights[p[j] - 1][j - 1] > 0) result.emplace_back(p[j] - 1, j - 1);
    }
    return result;
}
}
