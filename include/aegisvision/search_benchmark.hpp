#pragma once

#include "aegisvision/domain.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace aegisvision {
struct SearchBenchmarkItem {
    std::string id;
    std::filesystem::path image;
    std::string label;
    std::optional<BoundingBox> bbox;
};
struct SearchBenchmarkQuery {
    std::string text;
    std::string label;
};
struct SearchBenchmark {
    std::string dataset;
    std::vector<SearchBenchmarkItem> items;
    std::vector<SearchBenchmarkQuery> queries;
};

[[nodiscard]] SearchBenchmark load_search_benchmark(const std::filesystem::path& manifest);
[[nodiscard]] nlohmann::json score_search_benchmark(
    const SearchBenchmark& benchmark,
    const std::vector<std::vector<SearchResult>>& ranked_results);
}
