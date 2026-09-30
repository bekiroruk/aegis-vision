#include "aegisvision/search_benchmark.hpp"
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace aegisvision {
namespace {
std::string nonempty(const nlohmann::json& value,const char* field) {
    const auto text=value.at(field).get<std::string>();
    if(text.empty()) throw std::invalid_argument(std::string(field)+" must not be empty");
    return text;
}
}
SearchBenchmark load_search_benchmark(const std::filesystem::path& manifest) {
    std::ifstream input(manifest,std::ios::binary);
    if(!input) throw std::runtime_error("Cannot open search benchmark manifest");
    const auto root=nlohmann::json::parse(input);
    if(root.at("version")!=1) throw std::invalid_argument("Search benchmark version must be 1");
    SearchBenchmark benchmark;
    benchmark.dataset=nonempty(root,"dataset");
    std::set<std::string> ids,labels;
    const auto base=std::filesystem::absolute(manifest).parent_path();
    for(const auto& entry:root.at("items")) {
        SearchBenchmarkItem item;
        item.id=nonempty(entry,"id");
        item.label=nonempty(entry,"label");
        if(!ids.insert(item.id).second) throw std::invalid_argument("Duplicate benchmark item ID");
        labels.insert(item.label);
        const std::filesystem::path relative=std::filesystem::u8path(nonempty(entry,"image"));
        if(relative.is_absolute()) throw std::invalid_argument("Benchmark image path must be relative to manifest");
        item.image=(base/relative).lexically_normal();
        if(!std::filesystem::is_regular_file(item.image)) throw std::invalid_argument("Benchmark image missing");
        if(entry.contains("bbox")) {
            const auto& values=entry.at("bbox");
            if(!values.is_array() || values.size()!=4) throw std::invalid_argument("bbox must have four coordinates");
            const BoundingBox box{values.at(0).get<float>(),values.at(1).get<float>(),
                values.at(2).get<float>(),values.at(3).get<float>()};
            if(!std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) ||
               !std::isfinite(box.y2) || box.x1<0 || box.y1<0 || box.area()<=0)
                throw std::invalid_argument("Invalid benchmark bbox");
            item.bbox=box;
        }
        benchmark.items.push_back(std::move(item));
    }
    if(benchmark.items.size()<2 || benchmark.items.size()>10000)
        throw std::invalid_argument("Benchmark requires 2..10000 items");
    std::set<std::string> query_texts;
    for(const auto& entry:root.at("queries")) {
        SearchBenchmarkQuery query{nonempty(entry,"text"),nonempty(entry,"label")};
        if(!labels.contains(query.label)) throw std::invalid_argument("Query label has no relevant items");
        if(!query_texts.insert(query.text).second) throw std::invalid_argument("Duplicate benchmark query");
        benchmark.queries.push_back(std::move(query));
    }
    if(benchmark.queries.empty() || benchmark.queries.size()>1000)
        throw std::invalid_argument("Benchmark requires 1..1000 queries");
    return benchmark;
}
nlohmann::json score_search_benchmark(const SearchBenchmark& benchmark,
    const std::vector<std::vector<SearchResult>>& ranked_results) {
    if(ranked_results.size()!=benchmark.queries.size()) throw std::invalid_argument("Query/result count mismatch");
    std::unordered_map<std::string,std::string> labels;
    std::unordered_map<std::string,std::size_t> counts;
    for(const auto& item:benchmark.items) { labels.emplace(item.id,item.label); ++counts[item.label]; }
    nlohmann::json per_query=nlohmann::json::array();
    double mean_r1=0,mean_r5=0,mean_r10=0,mean_h1=0,mean_h5=0,mean_h10=0;
    for(std::size_t q=0;q<benchmark.queries.size();++q) {
        const auto& query=benchmark.queries[q];
        const auto& results=ranked_results[q];
        std::unordered_set<std::string> seen;
        std::size_t hit1=0,hit5=0,hit10=0;
        nlohmann::json ranked=nlohmann::json::array();
        for(std::size_t i=0;i<results.size();++i) {
            const auto& result=results[i];
            const auto found=labels.find(result.item_id);
            if(found==labels.end()) throw std::invalid_argument("Search returned an item outside the benchmark");
            if(!seen.insert(result.item_id).second) throw std::invalid_argument("Search returned a duplicate item");
            const bool relevant=found->second==query.label;
            if(relevant) { if(i<1) ++hit1; if(i<5) ++hit5; if(i<10) ++hit10; }
            if(i<10) ranked.push_back({{"id",result.item_id},{"score",result.score},{"relevant",relevant}});
        }
        const double denominator=static_cast<double>(counts.at(query.label));
        mean_r1+=hit1/denominator; mean_r5+=hit5/denominator; mean_r10+=hit10/denominator;
        mean_h1+=hit1>0; mean_h5+=hit5>0; mean_h10+=hit10>0;
        per_query.push_back({{"text",query.text},{"label",query.label},{"relevant_count",counts.at(query.label)},
            {"recall_at_1",hit1/denominator},{"recall_at_5",hit5/denominator},{"recall_at_10",hit10/denominator},
            {"hit_at_1",hit1>0},{"hit_at_5",hit5>0},{"hit_at_10",hit10>0},{"top_10",ranked}});
    }
    const double n=static_cast<double>(benchmark.queries.size());
    return {{"dataset",benchmark.dataset},{"items",benchmark.items.size()},{"queries",benchmark.queries.size()},
        {"metric","macro mean of relevant items retrieved / relevant items per query"},
        {"macro_recall_at_1",mean_r1/n},{"macro_recall_at_5",mean_r5/n},{"macro_recall_at_10",mean_r10/n},
        {"hit_at_1",mean_h1/n},{"hit_at_5",mean_h5/n},{"hit_at_10",mean_h10/n},{"per_query",per_query}};
}
}
