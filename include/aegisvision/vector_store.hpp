#pragma once

#include "aegisvision/contracts.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace aegisvision {

class InMemoryVectorStore final : public IVectorStore {
public:
    void upsert(
        std::string item_id,
        std::vector<float> vector,
        std::map<std::string, std::string> metadata) override;
    [[nodiscard]] std::vector<SearchResult> search(
        const std::vector<float>& vector, std::size_t limit = 10) const override;
    [[nodiscard]] std::vector<SearchResult> search_filtered(const std::vector<float>& vector,
        std::size_t limit, const std::map<std::string, std::string>& metadata) const override;

private:
    struct Item {
        std::vector<float> vector;
        std::map<std::string, std::string> metadata;
    };

    std::size_t dimension_{};
    std::unordered_map<std::string, Item> items_;
};

}  // namespace aegisvision

