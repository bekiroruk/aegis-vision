#include "aegisvision/crossview.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
using namespace aegisvision;
void check(bool ok) { if (!ok) throw std::runtime_error("Crossview assertion failed"); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected crossview rejection");
}
CrossViewPerson person(std::uint64_t id, int axis, float scale = 1) {
    CrossViewPerson p{id, std::vector<float>(512)}; p.embedding[axis] = scale; return p;
}
void tests() {
    auto a = person(1, 0), b = person(2, 1), c = person(3, 2);
    auto r = evaluate_crossview({a, b, c}, {b, a});
    check(r.size() == 3 && r[0].correct && r[1].correct && !r[2].accepted && !r[2].has_counterpart);
    check(r[0].rank1_correct && *r[0].best_gallery == 1);
    auto wrong = a; wrong.identity = 99;
    r = evaluate_crossview({a}, {wrong});
    check(r[0].accepted && !r[0].correct && !r[0].has_counterpart && !r[0].rank1_correct);
    r = evaluate_crossview({a}, {a, wrong});
    check(!r[0].accepted && !r[0].rank1_correct && r[0].reason == "ambiguous_query");
    r = evaluate_crossview({a, wrong}, {a});
    check(!r[0].accepted && !r[1].accepted); // column ambiguity
    r = evaluate_crossview({a}, {});
    check(r[0].reason == "no_gallery" && !r[0].similarity && !r[0].has_counterpart);
    check(evaluate_crossview({}, {a}).empty());
    auto scaled = person(1, 0, 100);
    check(evaluate_crossview({scaled}, {a})[0].correct);
    // Renaming ground truth changes scores, never the inferred match.
    auto renamed = a; renamed.identity = 90;
    auto first = evaluate_crossview({a, b}, {b, a});
    auto second = evaluate_crossview({renamed, b}, {b, a});
    check(first[0].best_gallery == second[0].best_gallery && first[0].accepted == second[0].accepted);
    auto orthogonal = b; orthogonal.embedding[0] = .6F; orthogonal.embedding[1] = .8F;
    r = evaluate_crossview({a, orthogonal}, {a, b}, .5, .05);
    check(r[0].correct && r[1].correct); // each column selects its stronger query
    auto mislabeled = a; mislabeled.identity = 2;
    auto another = b; another.identity = 1;
    r = evaluate_crossview({a, b}, {mislabeled, another});
    check(r[0].has_counterpart && r[0].accepted && !r[0].correct && !r[0].rank1_correct);
    rejects([&] { static_cast<void>(evaluate_crossview({a, a}, {b})); });
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b, b})); });
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b}, 0, .1)); });
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b}, .8, 0)); });
    a.embedding[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b})); });
    a.embedding.assign(512, 0);
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b})); });
    a.embedding = {1};
    rejects([&] { static_cast<void>(evaluate_crossview({a}, {b})); });
    rejects([&] { static_cast<void>(evaluate_crossview(std::vector<CrossViewPerson>(513), {})); });
}
}
int main() {
    try { tests(); std::cout << "Crossview matching/scoring tests passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
