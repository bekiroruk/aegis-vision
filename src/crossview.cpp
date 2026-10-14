#include "aegisvision/crossview.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace aegisvision {
namespace {
std::vector<std::vector<double>> features(const std::vector<CrossViewPerson>& people) {
    if (people.size() > 512) throw std::invalid_argument("Crossview capacity exceeded");
    std::set<std::uint64_t> identities;
    std::vector<std::vector<double>> result;
    for (const auto& person : people) {
        if (!identities.insert(person.identity).second || person.embedding.size() != 512)
            throw std::invalid_argument("Duplicate identity or invalid crossview feature dimension");
        double norm = 0;
        for (auto x : person.embedding) {
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite crossview feature");
            norm += static_cast<double>(x) * x;
        }
        if (norm < 1e-24) throw std::invalid_argument("Zero crossview feature");
        norm = std::sqrt(norm);
        std::vector<double> unit;
        for (auto x : person.embedding) unit.push_back(x / norm);
        result.push_back(std::move(unit));
    }
    return result;
}
struct Best {
    std::optional<std::size_t> index;
    double score{-2}, second{-2};
    void add(std::size_t i, double s) {
        if (s > score) { second = score; score = s; index = i; }
        else second = std::max(second, s);
    }
};
}
std::vector<CrossViewDecision> evaluate_crossview(const std::vector<CrossViewPerson>& queries,
    const std::vector<CrossViewPerson>& gallery, double threshold, double margin) {
    if (!std::isfinite(threshold) || threshold <= 0 || threshold > 1 ||
        !std::isfinite(margin) || margin <= 0 || margin > 1)
        throw std::invalid_argument("Invalid crossview thresholds");
    const auto q = features(queries), g = features(gallery);
    std::vector<Best> rows(q.size()), columns(g.size());
    for (std::size_t i = 0; i < q.size(); ++i) for (std::size_t j = 0; j < g.size(); ++j) {
        const double score = std::clamp(std::inner_product(q[i].begin(), q[i].end(), g[j].begin(), 0.), -1., 1.);
        rows[i].add(j, score); columns[j].add(i, score);
    }
    std::vector<CrossViewDecision> result;
    for (std::size_t i = 0; i < q.size(); ++i) {
        CrossViewDecision d;
        const auto& row = rows[i];
        d.reason = "no_gallery";
        d.has_counterpart = std::any_of(gallery.begin(), gallery.end(), [&](const auto& person) {
            return person.identity == queries[i].identity;
        });
        if (row.index) {
            d.best_gallery = row.index; d.similarity = row.score;
            const auto& column = columns[*row.index];
            d.rank1_correct = row.score - row.second > 1e-12 && gallery[*row.index].identity == queries[i].identity;
            if (row.score < threshold) d.reason = "below_threshold";
            else if (row.score - row.second < margin) d.reason = "ambiguous_query";
            else if (column.index != i) d.reason = "not_mutual_best";
            else if (column.score - column.second < margin) d.reason = "ambiguous_gallery";
            else { d.reason = "matched"; d.accepted = true; }
            d.correct = d.accepted && gallery[*row.index].identity == queries[i].identity;
        }
        result.push_back(std::move(d));
    }
    return result;
}
}
