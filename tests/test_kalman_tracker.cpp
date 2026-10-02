#include "aegisvision/kalman_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace aegisvision;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function> void rejects(Function function, const char* message) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

Detection box(float x, float score = 0.9F, std::string label = "person") {
    return {{x, 0.0F, x + 20.0F, 40.0F}, std::move(label), score, {}, {}};
}

const Track& at_x(const std::vector<Track>& tracks, float x) {
    const auto found = std::find_if(tracks.begin(), tracks.end(), [&](const auto& track) {
        return track.bbox.x1 == x;
    });
    if (found == tracks.end()) throw std::runtime_error("Expected observed box missing");
    return *found;
}

void lifecycle_and_motion() {
    KalmanTracker tracker;
    require(tracker.update({box(0.0F, 0.09F)}).empty(), "Sub-low box started a track");
    require(tracker.update({box(0.0F, 0.20F)}).empty(), "Low box started a track");
    require(tracker.update({box(0.0F, 0.40F)}).empty(), "Sub-birth high box started a track");
    const auto id = tracker.update({box(0.0F)}).front().track_id;
    const auto rescued = tracker.update({box(3.0F, 0.20F)});
    require(rescued.size() == 1 && rescued.front().track_id == id, "Low observation lost active identity");
    require(rescued.front().bbox.x1 == 3.0F && rescued.front().score == 0.20F,
            "Returned box was filtered/predicted rather than the observation");
    require(tracker.stats().low_confidence_matches == 1, "Low match counter wrong");
    require(tracker.update({}).empty(), "Unobserved predicted track was rendered");
    require(tracker.update({box(6.0F, 0.20F)}).empty(), "Low observation revived a lost identity");
    const auto recovered = tracker.update({box(7.0F)});
    require(recovered.size() == 1 && recovered.front().track_id == id, "High observation failed lost recovery");
    require(tracker.stats().reactivations == 1, "Lost recovery counter wrong");

    KalmanTracker motion;
    std::uint64_t moving_id = 0;
    for (int frame = 0; frame < 20; ++frame) {
        const float x = static_cast<float>(frame * 3) + (frame % 2 == 0 ? 0.25F : -0.25F);
        const auto result = motion.update({box(x)});
        require(result.size() == 1, "Moving jittered object disappeared");
        if (frame == 0) moving_id = result.front().track_id;
        require(result.front().track_id == moving_id, "Moving jittered object changed identity");
        require(result.front().bbox.x1 == x, "Motion track output altered detector coordinates");
    }
    require(motion.update({}).empty() && motion.update({}).empty(), "Gap returned predicted observations");
    require(motion.update({box(66.0F)}).front().track_id == moving_id, "Kalman prediction failed to bridge a gap");
    require(motion.stats().numerical_resets == 0, "Normal motion unexpectedly reset the covariance");

    KalmanTrackerConfig short_life;
    short_life.max_missed_frames = 1;
    KalmanTracker expiry(short_life);
    const auto old_id = expiry.update({box(0.0F)}).front().track_id;
    (void)expiry.update({}); (void)expiry.update({});
    require(expiry.stats().expired_tracks == 1, "Lost track did not expire at configured lifetime");
    require(expiry.update({box(0.0F)}).front().track_id != old_id, "Expired identity was reused");
    short_life.max_missed_frames = 0;
    KalmanTracker immediate(short_life);
    (void)immediate.update({box(0.0F)}); (void)immediate.update({});
    require(immediate.stats().expired_tracks == 1, "Zero lifetime did not expire on the first missed frame");

    KalmanTracker shrinking;
    for (float width : {20.0F, 18.0F, 16.0F, 14.0F, 12.0F}) {
        const Detection observed{{0.0F, 0.0F, width, width * 2.0F}, "person", 0.9F, {}, {}};
        const auto track = shrinking.update({observed});
        require(track.size() == 1 && track.front().bbox.x2 == width, "Shrinking observation was invalid or filtered");
    }
    for (int gap = 0; gap < 21; ++gap) {
        require(shrinking.update({}).empty(), "Contracting lost box became a visible prediction");
    }
    require(shrinking.stats().created_tracks == 1 && shrinking.stats().expired_tracks == 1,
            "Contracting Kalman size failed safe long-gap expiry");
}

void association_order_and_gating() {
    // If active and lost identities are assigned jointly, the optimum swaps
    // the active ID: .538 + .428 > .600. Stage order must protect active ID 1.
    // This fixture is synthetic and does not use labelled benchmark boxes.
    KalmanTracker priority;
    const auto first = priority.update({box(0.0F), box(13.0F)});
    const auto active_id = at_x(first, 0.0F).track_id;
    const auto lost_id = at_x(first, 13.0F).track_id;
    require(priority.update({box(0.0F)}).front().track_id == active_id, "Failed to establish an active/lost pair");
    const auto result = priority.update({box(5.0F), box(-6.0F)});
    require(at_x(result, 5.0F).track_id == active_id, "Lost identity stole the active high-confidence match");
    require(at_x(result, -6.0F).track_id != lost_id, "Lost identity ignored the IoU gate");

    KalmanTracker high_first;
    const auto id = high_first.update({box(0.0F)}).front().track_id;
    const auto matched = high_first.update({box(0.0F, 0.20F), box(1.0F, 0.80F)});
    require(matched.size() == 1 && matched.front().track_id == id && matched.front().score == 0.80F,
            "Low-confidence box stole an active high-confidence match");

    KalmanTracker classes;
    const auto person_id = classes.update({box(0.0F)}).front().track_id;
    require(classes.update({box(0.0F, 0.9F, "car")}).front().track_id != person_id,
            "Different object classes shared an identity");

    KalmanTrackerConfig strict;
    strict.gating_threshold = 0.01;
    KalmanTracker gate(strict);
    const auto original = gate.update({box(0.0F)}).front().track_id;
    require(gate.update({box(1.0F)}).front().track_id != original,
            "Squared Mahalanobis gate did not reject an otherwise overlapping box");
    require(gate.stats().gate_rejections == 1, "Gate rejection counter wrong");

    KalmanTracker ties;
    const auto tied = ties.update({box(0.0F), box(0.0F)});
    require(tied.size() == 2 && tied[0].track_id != tied[1].track_id, "Duplicate observation identities");
    require(ties.update({box(0.0F)}).front().track_id == tied.front().track_id, "Association tie is not deterministic");
}

void bounds_and_validation() {
    for (int field = 0; field < 8; ++field) {
        KalmanTrackerConfig config;
        switch (field) {
        case 0: config.low_threshold = config.high_threshold; break;
        case 1: config.new_track_threshold = 0.20F; break;
        case 2: config.match_iou = std::numeric_limits<float>::quiet_NaN(); break;
        case 3: config.max_missed_frames = 10001; break;
        case 4: config.gating_threshold = std::numeric_limits<double>::infinity(); break;
        case 5: config.gating_threshold = 0.0; break;
        case 6: config.high_threshold = -0.1F; break;
        default: config.gating_threshold = 1'000'001.0; break;
        }
        rejects([&] { KalmanTracker invalid(config); }, "Invalid tracker configuration was accepted");
    }

    KalmanTracker tracker;
    const auto before = tracker.update({box(0.0F)}).front();
    for (int field = 0; field < 8; ++field) {
        auto invalid = box(1.0F);
        switch (field) {
        case 0: invalid.score = std::numeric_limits<float>::quiet_NaN(); break;
        case 1: invalid.score = 1.1F; break;
        case 2: invalid.label.clear(); break;
        case 3: invalid.label.assign(257, 'x'); break;
        case 4: invalid.bbox.x2 = invalid.bbox.x1; break;
        case 5: invalid.bbox.y1 = std::numeric_limits<float>::infinity(); break;
        case 6: invalid.bbox.x1 = -1'000'001.0F; break;
        default: invalid.label = std::string{"per\0son", 7}; break;
        }
        // A valid observation before an invalid one cannot partially update state.
        rejects([&] { (void)tracker.update({box(2.0F), invalid}); }, "Invalid observation was accepted");
    }
    std::vector<Detection> excess(KalmanTracker::max_detections + 1, box(0.0F));
    rejects([&] { (void)tracker.update(excess); }, "Detector input cap was ignored");
    const auto after = tracker.update({box(0.0F)}).front();
    require(after.track_id == before.track_id && after.age == before.age + 1,
            "Rejected observations changed identity or aged tracker state");
    require(tracker.stats().created_tracks == 1 && tracker.stats().expired_tracks == 0,
            "Rejected observations mutated lifecycle counters");
    require(tracker.update({box(0.0F, 0.9F, "car")}).front().track_id == 2,
            "Rejected observations consumed the next identity");

    KalmanTrackerConfig immediate_config;
    immediate_config.max_missed_frames = 0;
    KalmanTracker capacity(immediate_config);
    std::vector<Detection> births(KalmanTracker::max_tracks + 1, box(0.0F));
    const auto limited = capacity.update(births);
    require(limited.size() == KalmanTracker::max_tracks, "Live identity capacity was ignored");
    require(capacity.stats().capacity_rejections == 1 && capacity.stats().created_tracks == KalmanTracker::max_tracks,
            "Capacity suppression counters wrong");
    (void)capacity.update({});
    require(capacity.stats().expired_tracks == KalmanTracker::max_tracks, "Capacity-bounded states failed expiry");
    const auto new_track = capacity.update({box(0.0F)}).front();
    require(new_track.track_id == KalmanTracker::max_tracks + 1, "Suppressed birth consumed an identity");

    // Valid negative image coordinates and tiny boxes remain well-defined.
    KalmanTracker coordinates;
    const Detection tiny{{-0.001F, -0.001F, 0.001F, 0.001F}, "person", 0.9F, {}, {}};
    const auto tiny_id = coordinates.update({tiny}).front().track_id;
    require(coordinates.update({tiny}).front().track_id == tiny_id, "Small negative-coordinate box was not stable");
}
}  // namespace

int main() {
    try {
        lifecycle_and_motion(); association_order_and_gating(); bounds_and_validation();
        std::cout << "Kalman motion, active priority, uncertainty gating, recovery and bounded-state tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
