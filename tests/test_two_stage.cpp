#include "aegisvision/assignment.hpp"
#include "aegisvision/two_stage_tracker.hpp"
#include "aegisvision/tracking.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

namespace {
using namespace aegisvision;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* message) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}
Detection box(float x, float score = 0.9F, std::string label = "person") {
    return {{x,0,x+20,40}, std::move(label), score, {}, {}};
}
double brute_force(const std::vector<std::vector<double>>& weights, std::size_t row, std::set<std::size_t>& used) {
    if (row == weights.size()) return 0;
    double best = brute_force(weights, row + 1, used);
    for (std::size_t j = 0; j < weights[row].size(); ++j) {
        if (used.contains(j) || weights[row][j] <= 0) continue;
        used.insert(j);
        best = std::max(best, weights[row][j] + brute_force(weights, row + 1, used));
        used.erase(j);
    }
    return best;
}
void assignment_tests() {
    require(assign_max_weight({}).empty(), "Empty assignment failed");
    require(assign_max_weight({{-1,-1},{-1,-1}}).empty(), "Forbidden edges matched");
    const auto global = assign_max_weight({{0.9,0.8},{0.85,-1}});
    require(global.size() == 2 && global[0].first == 1 && global[1].first == 0, "Global optimum lost to greedy choice");
    rejects([] { (void)assign_max_weight({{0.5},{0.6,0.7}}); }, "Ragged matrix accepted");
    rejects([] { (void)assign_max_weight({{std::numeric_limits<double>::quiet_NaN()}}); }, "NaN weight accepted");
    std::mt19937 random(42);
    for (int trial = 0; trial < 100; ++trial) {
        const std::size_t rows = 1 + random() % 5, cols = 1 + random() % 5;
        std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));
        for (auto& row : weights) for (auto& value : row) value = random() % 4 == 0 ? -1 : (random() % 11) / 10.0;
        double actual = 0;
        std::set<std::size_t> used_rows, used_cols, brute_used;
        for (const auto& [i,j] : assign_max_weight(weights)) {
            require(used_rows.insert(i).second && used_cols.insert(j).second, "Duplicate assignment");
            actual += weights[i][j];
        }
        require(std::abs(actual - brute_force(weights, 0, brute_used)) < 1e-9, "Hungarian differs from exhaustive optimum");
    }
}
void tracker_tests() {
    TwoStageTracker tracker;
    require(tracker.update({box(0,0.2F)}).empty(), "Weak detection started a track");
    require(tracker.update({box(0,0.4F)}).empty(), "Sub-birth threshold started a track");
    const auto id = tracker.update({box(0)}).front().track_id;
    require(tracker.update({box(4,0.2F)}).front().track_id == id, "Low-confidence recovery lost ID");
    require(tracker.stats().low_confidence_matches == 1, "Recovery counter wrong");
    require(tracker.update({}).empty(), "Lost track rendered without observation");
    require(tracker.update({box(12,0.2F)}).empty(), "Weak box revived lost track");
    require(tracker.update({box(16)}).front().track_id == id, "High score did not reactivate predicted track");
    require(tracker.stats().reactivations == 1, "Reactivation counter wrong");

    TwoStageConfig short_life; short_life.max_missed_frames = 1;
    TwoStageTracker expiry(short_life);
    const auto old_id = expiry.update({box(0)}).front().track_id;
    (void)expiry.update({}); (void)expiry.update({});
    require(expiry.update({box(0)}).front().track_id != old_id, "Expired ID reused");
    require(expiry.stats().expired_tracks == 1, "Expiry counter wrong");

    TwoStageTracker motion;
    IoUTracker baseline;
    std::uint64_t moving_id = 0, baseline_id = 0;
    for (float x : {0.0F,8.0F}) {
        moving_id = motion.update({box(x)}).front().track_id;
        baseline_id = baseline.update({box(x)}).front().track_id;
    }
    (void)motion.update({}); (void)baseline.update({});
    require(motion.update({box(24)}).front().track_id == moving_id, "Motion prediction did not bridge gap");
    require(baseline.update({box(24)}).front().track_id != baseline_id, "Fixture must distinguish from static IoU");

    TwoStageTracker classes;
    const auto person = classes.update({box(0)}).front().track_id;
    require(classes.update({box(0,0.9F,"car")}).front().track_id != person, "Class changed on an existing ID");
    TwoStageTracker priority;
    (void)priority.update({box(0)});
    auto high_first = priority.update({box(0,0.2F),box(1,0.8F)});
    require(high_first.size() == 1 && high_first.front().score == 0.8F, "Low score stole high-confidence match");
    TwoStageTracker ties;
    const auto first = ties.update({box(0),box(0)});
    require(first.size() == 2 && first[0].track_id != first[1].track_id, "Duplicate ID");
    require(ties.update({box(0)}).front().track_id == first[0].track_id, "Tie not deterministic");
    TwoStageConfig invalid; invalid.low_threshold = invalid.high_threshold;
    rejects([&] { TwoStageTracker bad(invalid); }, "Invalid thresholds accepted");
    const auto before = tracker.stats().created_tracks;
    rejects([&] { (void)tracker.update({box(0,std::numeric_limits<float>::quiet_NaN())}); }, "Invalid score accepted");
    require(tracker.stats().created_tracks == before, "Invalid input mutated state");
}
}
int main() {
    try {
        assignment_tests(); tracker_tests();
        std::cout << "Assignment optimality (100 exhaustive comparisons), two-stage recovery and motion tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
