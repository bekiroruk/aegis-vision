#include "aegisvision/vector_store.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace aegisvision {
namespace {

std::vector<float> normalized(std::vector<float> vector) {
    if (vector.empty()) {
        throw std::invalid_argument("Vector must not be empty");
    }
    const auto squared_norm = std::inner_product(
        vector.begin(), vector.end(), vector.begin(), 0.0F);
    const auto norm = std::sqrt(squared_norm);
    if (norm == 0.0F) {
        throw std::invalid_argument("Zero vector cannot be normalized");
    }
    for (auto& value : vector) {
        value /= norm;
    }
    return vector;
}

}  // namespace

void InMemoryVectorStore::upsert(
    std::string item_id,
    std::vector<float> vector,
    std::map<std::string, std::string> metadata) {
    vector = normalized(std::move(vector));
    if (dimension_ == 0) {
        dimension_ = vector.size();
    }
    if (vector.size() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    items_.insert_or_assign(std::move(item_id), Item{std::move(vector), std::move(metadata)});
}

std::vector<SearchResult> InMemoryVectorStore::search(
    const std::vector<float>& vector, const std::size_t limit) const {
    return search_filtered(vector, limit, {});
}
std::vector<SearchResult> InMemoryVectorStore::search_filtered(const std::vector<float>& vector,
    const std::size_t limit, const std::map<std::string, std::string>& metadata) const {
    if (limit == 0) {
        throw std::invalid_argument("Search limit must be positive");
    }
    auto query = normalized(vector);
    if (dimension_ != 0 && query.size() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    std::vector<SearchResult> results;
    results.reserve(items_.size());
    for (const auto& [item_id, item] : items_) {
        if (!std::all_of(metadata.begin(), metadata.end(), [&](const auto& entry) {
            const auto found = item.metadata.find(entry.first);
            return found != item.metadata.end() && found->second == entry.second;
        })) continue;
        const auto score = std::inner_product(
            query.begin(), query.end(), item.vector.begin(), 0.0F);
        results.push_back(SearchResult{item_id, score, item.metadata});
    }
    std::ranges::sort(results, std::greater{}, &SearchResult::score);
    if (results.size() > limit) {
        results.resize(limit);
    }
    return results;
}

}  // namespace aegisvision

