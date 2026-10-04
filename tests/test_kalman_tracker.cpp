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

KalmanTrackerConfig gate_config(KalmanGateMode mode) {
    KalmanTrackerConfig config;
    config.gate_mode = mode;
    config.gating_threshold = mode == KalmanGateMode::CenterOnly ? kalman_center_gate99 : kalman_box_gate99;
    return config;
}

KalmanTrackerConfig appearance_config() {
    auto config = gate_config(KalmanGateMode::CenterOnly);
    config.use_appearance = true;
    config.appearance_dimension = 2;
    return config;
}

Detection described(float x, std::vector<float> appearance, float score = 0.9F, std::string label = "person") {
    auto result = box(x, score, std::move(label));
    result.embedding = std::move(appearance);
    return result;
}

const Track& at_x(const std::vector<Track>& tracks, float x) {
    const auto found = std::find_if(tracks.begin(), tracks.end(), [&](const auto& track) {
        return track.bbox.x1 == x;
    });
    if (found == tracks.end()) throw std::runtime_error("Expected observed box missing");
    return *found;
}

void lifecycle_and_motion(KalmanGateMode mode) {
    const auto configuration = gate_config(mode);
    KalmanTracker tracker(configuration);
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

    KalmanTracker motion(configuration);
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

    auto short_life = configuration;
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

    KalmanTracker shrinking(configuration);
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

void association_order_and_gating(KalmanGateMode mode) {
    const auto configuration = gate_config(mode);
    // If active and lost identities are assigned jointly, the optimum swaps
    // the active ID: .538 + .428 > .600. Stage order must protect active ID 1.
    // This fixture is synthetic and does not use labelled benchmark boxes.
    KalmanTracker priority(configuration);
    const auto first = priority.update({box(0.0F), box(13.0F)});
    const auto active_id = at_x(first, 0.0F).track_id;
    const auto lost_id = at_x(first, 13.0F).track_id;
    require(priority.update({box(0.0F)}).front().track_id == active_id, "Failed to establish an active/lost pair");
    const auto result = priority.update({box(5.0F), box(-6.0F)});
    require(at_x(result, 5.0F).track_id == active_id, "Lost identity stole the active high-confidence match");
    require(at_x(result, -6.0F).track_id != lost_id, "Lost identity ignored the IoU gate");

    KalmanTracker high_first(configuration);
    const auto id = high_first.update({box(0.0F)}).front().track_id;
    const auto matched = high_first.update({box(0.0F, 0.20F), box(1.0F, 0.80F)});
    require(matched.size() == 1 && matched.front().track_id == id && matched.front().score == 0.80F,
            "Low-confidence box stole an active high-confidence match");

    KalmanTracker classes(configuration);
    const auto person_id = classes.update({box(0.0F)}).front().track_id;
    require(classes.update({box(0.0F, 0.9F, "car")}).front().track_id != person_id,
            "Different object classes shared an identity");

    auto strict = configuration;
    strict.gating_threshold = 0.01;
    KalmanTracker gate(strict);
    const auto original = gate.update({box(0.0F)}).front().track_id;
    require(gate.update({box(1.0F)}).front().track_id != original,
            "Squared Mahalanobis gate did not reject an otherwise overlapping box");
    require(gate.stats().gate_rejections == 1, "Gate rejection counter wrong");

    KalmanTracker ties(configuration);
    const auto tied = ties.update({box(0.0F), box(0.0F)});
    require(tied.size() == 2 && tied[0].track_id != tied[1].track_id, "Duplicate observation identities");
    require(ties.update({box(0.0F)}).front().track_id == tied.front().track_id, "Association tie is not deterministic");
}

void bounds_and_validation(KalmanGateMode mode) {
    const auto configuration = gate_config(mode);
    for (int field = 0; field < 9; ++field) {
        auto config = configuration;
        switch (field) {
        case 0: config.low_threshold = config.high_threshold; break;
        case 1: config.new_track_threshold = 0.20F; break;
        case 2: config.match_iou = std::numeric_limits<float>::quiet_NaN(); break;
        case 3: config.max_missed_frames = 10001; break;
        case 4: config.gating_threshold = std::numeric_limits<double>::infinity(); break;
        case 5: config.gating_threshold = 0.0; break;
        case 6: config.high_threshold = -0.1F; break;
        case 7: config.gating_threshold = 1'000'001.0; break;
        default: config.gate_mode = static_cast<KalmanGateMode>(99); break;
        }
        rejects([&] { KalmanTracker invalid(config); }, "Invalid tracker configuration was accepted");
    }

    KalmanTracker tracker(configuration);
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

    auto immediate_config = configuration;
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
    KalmanTracker coordinates(configuration);
    const Detection tiny{{-0.001F, -0.001F, 0.001F, 0.001F}, "person", 0.9F, {}, {}};
    const auto tiny_id = coordinates.update({tiny}).front().track_id;
    require(coordinates.update({tiny}).front().track_id == tiny_id, "Small negative-coordinate box was not stable");
}

void center_gate_size_jitter() {
    KalmanTrackerConfig legacy;
    require(legacy.gate_mode == KalmanGateMode::FullBox && legacy.gating_threshold == kalman_box_gate99,
            "Legacy full-box defaults changed");
    require(kalman_box_gate99 == 13.2767 && kalman_center_gate99 == 9.2103,
            "Documented chi-square gate constants changed");
    auto center_config = gate_config(KalmanGateMode::CenterOnly);
    KalmanTracker full_box(legacy), center_only(center_config);
    const auto centered = [](float width, float height) -> Detection {
        return {{50.0F - width / 2.0F, 100.0F - height / 2.0F,
                 50.0F + width / 2.0F, 100.0F + height / 2.0F}, "person", 0.9F, {}, {}};
    };
    const auto initial = centered(20.0F, 80.0F);
    const auto full_id = full_box.update({initial}).front().track_id;
    const auto center_id = center_only.update({initial}).front().track_id;
    const auto enlarged = centered(40.0F, 100.0F);
    require(initial.bbox.iou(enlarged.bbox) >= legacy.match_iou,
            "Size-jitter fixture did not pass the common IoU gate");
    const auto rejected = full_box.update({enlarged});
    require(rejected.size() == 1 && rejected.front().track_id != full_id && full_box.stats().gate_rejections == 1,
            "Legacy full-box gate did not reject the fixed-center size jump");
    for (int frame = 0; frame < 24; ++frame) {
        const auto observation = frame % 2 == 0 ? enlarged : initial;
        const auto result = center_only.update({observation});
        require(result.size() == 1 && result.front().track_id == center_id,
                "Center-only gate fragmented a fixed-center size-jitter identity");
        const auto& actual = result.front().bbox;
        require(actual.x1 == observation.bbox.x1 && actual.y1 == observation.bbox.y1 &&
                actual.x2 == observation.bbox.x2 && actual.y2 == observation.bbox.y2,
                "Center-only output altered the observed box dimensions");
    }
    require(center_only.stats().gate_rejections == 0 && center_only.stats().numerical_resets == 0 &&
            center_only.stats().created_tracks == 1,
            "Center-only size-jitter handling changed lifecycle or numerical counters");

    // Same-size motion at IoU .379 passes the full-box 4D nominal gate but
    // exceeds the center 2D nominal gate: d^2 = 81/7.5625, about 10.71.
    KalmanTracker full_motion;
    KalmanTracker center_motion(gate_config(KalmanGateMode::CenterOnly));
    const auto full_motion_id = full_motion.update({box(0.0F)}).front().track_id;
    const auto center_motion_id = center_motion.update({box(0.0F)}).front().track_id;
    require(full_motion.update({box(9.0F)}).front().track_id == full_motion_id,
            "Legacy full-box nominal gate changed");
    require(center_motion.update({box(9.0F)}).front().track_id != center_motion_id &&
                center_motion.stats().gate_rejections == 1,
            "Center-only mode did not apply its two-dimensional nominal gate");

    // Removing size from the uncertainty gate must not remove the IoU gate.
    const auto oversized = centered(200.0F, 400.0F);
    const auto unrelated = center_only.update({oversized});
    require(unrelated.size() == 1 && unrelated.front().track_id != center_id,
            "Center-only gate bypassed the common IoU condition");
}

void center_gate_nearby_crossing() {
    KalmanTracker tracker(gate_config(KalmanGateMode::CenterOnly));
    const auto shifted = [](float x, float y, std::string label = "person") -> Detection {
        return {{x, y, x + 20.0F, y + 40.0F}, std::move(label), 0.9F, {}, {}};
    };
    std::uint64_t right_id = 0, left_id = 0;
    for (int frame = 0; frame < 11; ++frame) {
        const auto rightward = shifted(static_cast<float>(frame * 6), 0.0F);
        const auto leftward = shifted(60.0F - static_cast<float>(frame * 6), 8.0F);
        // Alternate detector order; Hungarian association must remain one-to-one.
        const std::vector<Detection> observations = frame % 2 == 0
            ? std::vector<Detection>{rightward, leftward} : std::vector<Detection>{leftward, rightward};
        const auto result = tracker.update(observations);
        require(result.size() == 2 && result[0].track_id != result[1].track_id,
                "Nearby crossing produced duplicate or missing visible identities");
        const auto right = std::find_if(result.begin(), result.end(), [](const auto& track) { return track.bbox.y1 == 0.0F; });
        const auto left = std::find_if(result.begin(), result.end(), [](const auto& track) { return track.bbox.y1 == 8.0F; });
        require(right != result.end() && left != result.end(), "Crossing output lost an observed vertical offset");
        if (frame == 0) { right_id = right->track_id; left_id = left->track_id; }
        require(right->track_id == right_id && left->track_id == left_id,
                "Center gate swapped the deterministic synthetic crossing identities");
    }
    require(tracker.stats().created_tracks == 2 && tracker.stats().numerical_resets == 0,
            "Synthetic crossing unexpectedly spawned identities or reset covariance");
    // This controlled fixture has distinct offsets and motion. It establishes
    // association invariants, not recovery of indistinguishable people or Re-ID.
    KalmanTracker classes(gate_config(KalmanGateMode::CenterOnly));
    const auto first = classes.update({shifted(0.0F, 0.0F), shifted(0.0F, 0.0F, "car")});
    const auto second = classes.update({shifted(1.0F, 0.0F, "car"), shifted(1.0F, 0.0F)});
    require(second.size() == 2, "Center gate merged coincident objects of different classes");
    for (const auto& previous : first) {
        const auto same_class = std::find_if(second.begin(), second.end(), [&](const auto& track) {
            return track.label == previous.label;
        });
        require(same_class != second.end() && same_class->track_id == previous.track_id,
                "Center gate reassigned a coincident different-class identity");
    }
}

void appearance_ambiguous_assignment() {
    const auto configuration = appearance_config();
    require(configuration.max_cosine_distance == 0.20 && configuration.appearance_weight == 0.50 &&
                configuration.appearance_momentum == 0.90 && KalmanTrackerConfig{}.appearance_dimension == 512 &&
                !KalmanTrackerConfig{}.use_appearance,
            "Frozen appearance defaults changed");
    KalmanTracker geometry(gate_config(KalmanGateMode::CenterOnly)), appearance(configuration);
    const std::vector<Detection> initial{described(0.0F, {3.0F, 0.0F}), described(4.0F, {0.0F, 7.0F})};
    const auto old_geometry = geometry.update(initial);
    const auto old_appearance = appearance.update(initial);
    require(appearance.stats().appearance_matches == 0 && appearance.stats().appearance_updates == 0,
            "Appearance births were counted as matches or EMA updates");
    // Both boxes pass every geometric gate, but overlap prefers the other
    // person's previous box. Distinct synthetic features reject those swaps.
    const std::vector<Detection> crossed{described(1.0F, {0.0F, 14.0F}), described(3.0F, {6.0F, 0.0F})};
    const auto new_geometry = geometry.update(crossed);
    const auto new_appearance = appearance.update(crossed);
    require(at_x(new_geometry, 3.0F).track_id == at_x(old_geometry, 4.0F).track_id &&
                at_x(new_geometry, 1.0F).track_id == at_x(old_geometry, 0.0F).track_id,
            "Ambiguous fixture did not exercise the original geometric swap");
    require(at_x(new_appearance, 3.0F).track_id == at_x(old_appearance, 0.0F).track_id &&
                at_x(new_appearance, 1.0F).track_id == at_x(old_appearance, 4.0F).track_id,
            "Appearance gate failed to retain distinct synthetic identities");
    require(appearance.stats().appearance_rejections == 2 && appearance.stats().appearance_matches == 2 &&
                appearance.stats().appearance_updates == 2 && appearance.stats().created_tracks == 2,
            "Appearance candidate/assignment/update counters were not distinct");
    require(initial[0].embedding == std::vector<float>({3.0F, 0.0F}) &&
                crossed[0].embedding == std::vector<float>({0.0F, 14.0F}),
            "Appearance normalization mutated caller embeddings");

    // With a permissive appearance gate, the fused reward itself, rather than
    // a forbidden edge, must resolve the same ambiguous geometry correctly.
    auto fused_config = configuration;
    fused_config.max_cosine_distance = 1.0;
    KalmanTracker fused(fused_config);
    const auto fused_first = fused.update(initial);
    const auto fused_second = fused.update(crossed);
    require(at_x(fused_second, 3.0F).track_id == at_x(fused_first, 0.0F).track_id &&
                at_x(fused_second, 1.0F).track_id == at_x(fused_first, 4.0F).track_id &&
                fused.stats().appearance_rejections == 0,
            "IoU/cosine fused reward did not resolve admitted ambiguous pairs");
}

void appearance_hard_gates_and_lifecycle() {
    const auto configuration = appearance_config();
    for (int condition = 0; condition < 3; ++condition) {
        auto config = configuration;
        if (condition == 2) config.gating_threshold = 0.01;
        KalmanTracker tracker(config);
        const auto initial_id = tracker.update({described(0.0F, {1.0F, 0.0F})}).front().track_id;
        auto incompatible = described(condition == 1 ? 200.0F : condition == 2 ? 1.0F : 0.0F,
                                      {1.0F, 0.0F}, 0.9F, condition == 0 ? "car" : "person");
        const auto result = tracker.update({incompatible});
        require(result.size() == 1 && result.front().track_id != initial_id,
                "Identical appearance bypassed a class, IoU or motion gate");
        require(tracker.stats().appearance_matches == 0 && tracker.stats().appearance_rejections == 0 &&
                    tracker.stats().appearance_updates == 0,
                "Appearance counters included a pair rejected by a preceding hard gate");
        if (condition == 2) require(tracker.stats().gate_rejections == 1, "Motion gate counter lost appearance-rejected pairs");
    }

    KalmanTracker rescue(configuration);
    const auto id = rescue.update({described(0.0F, {1.0F, 0.0F})}).front().track_id;
    require(rescue.update({described(0.0F, {0.0F, 1.0F}, 0.20F)}).empty(),
            "Dissimilar low-confidence appearance rescued an active identity");
    require(rescue.stats().appearance_rejections == 1 && rescue.stats().appearance_matches == 0 &&
                rescue.stats().appearance_updates == 0 && rescue.stats().low_confidence_matches == 0,
            "Dissimilar low observation contaminated appearance/lifecycle counters");
    require(rescue.update({described(0.0F, {2.0F, 0.0F})}).front().track_id == id &&
                rescue.stats().reactivations == 1 && rescue.stats().appearance_matches == 1 &&
                rescue.stats().appearance_updates == 1,
            "Appearance prevented normal lost high-confidence reactivation");
    require(rescue.update({}).empty(), "Appearance returned an unobserved prediction");
    require(rescue.update({described(0.0F, {1.0F, 0.0F}, 0.20F)}).empty(),
            "Appearance allowed low confidence to revive a lost identity");

    auto immediate = configuration;
    immediate.max_missed_frames = 0;
    KalmanTracker expiry(immediate);
    const auto expired_id = expiry.update({described(0.0F, {1.0F, 0.0F})}).front().track_id;
    (void)expiry.update({});
    require(expiry.update({described(0.0F, {1.0F, 0.0F})}).front().track_id != expired_id,
            "Appearance resurrected an expired identity outside the bounded lifecycle");
}

void appearance_high_only_ema() {
    const auto configuration = appearance_config();
    const std::vector<float> tilted{0.90F, std::sqrt(0.19F)};
    const std::vector<float> probe{0.79F, std::sqrt(1.0F - 0.79F * 0.79F)};
    // Probe similarity is .79 to the original prototype (rejected) but about
    // .816 to normalized .90*original + .10*tilted (accepted). Therefore these
    // observations distinguish EMA updates without exposing private state.
    KalmanTracker high(configuration);
    const auto high_id = high.update({described(0.0F, {1.0F, 0.0F})}).front().track_id;
    require(high.update({described(0.0F, tilted)}).front().track_id == high_id,
            "Compatible high appearance did not match before its EMA update");
    require(high.update({described(0.0F, probe)}).front().track_id == high_id &&
                high.stats().appearance_matches == 2 && high.stats().appearance_updates == 2 &&
                high.stats().created_tracks == 1,
            "High-confidence prototype did not receive the normalized frozen-momentum EMA");

    KalmanTracker low(configuration);
    const auto low_id = low.update({described(0.0F, {1.0F, 0.0F})}).front().track_id;
    require(low.update({described(0.0F, tilted, 0.20F)}).front().track_id == low_id,
            "Compatible low appearance did not rescue an active identity");
    const auto rejected = low.update({described(0.0F, probe)});
    require(rejected.size() == 1 && rejected.front().track_id != low_id &&
                low.stats().appearance_matches == 1 && low.stats().appearance_updates == 0 &&
                low.stats().low_confidence_matches == 1 && low.stats().appearance_rejections == 1,
            "Low-confidence matching contaminated the appearance prototype");

    auto exact_config = configuration;
    exact_config.appearance_dimension = 3;
    exact_config.max_cosine_distance = 0.0;
    KalmanTracker exact(exact_config);
    const auto exact_id = exact.update({described(0.0F, {3.0F, 4.0F, 5.0F})}).front().track_id;
    require(exact.update({described(0.0F, {6.0F, 8.0F, 10.0F})}).front().track_id == exact_id,
            "Zero-distance gate rejected equal directions due to normalization rounding");
}

void appearance_validation_and_bounds() {
    for (bool enabled : {false, true}) {
        for (int field = 0; field < 13; ++field) {
            auto config = appearance_config();
            config.use_appearance = enabled;
            switch (field) {
            case 0: config.appearance_dimension = 0; break;
            case 1: config.appearance_dimension = 1025; break;
            case 2: config.max_cosine_distance = std::numeric_limits<double>::quiet_NaN(); break;
            case 3: config.max_cosine_distance = std::numeric_limits<double>::infinity(); break;
            case 4: config.max_cosine_distance = -0.01; break;
            case 5: config.max_cosine_distance = 1.01; break;
            case 6: config.appearance_weight = std::numeric_limits<double>::quiet_NaN(); break;
            case 7: config.appearance_weight = -0.01; break;
            case 8: config.appearance_weight = 1.01; break;
            case 9: config.appearance_momentum = std::numeric_limits<double>::quiet_NaN(); break;
            case 10: config.appearance_momentum = -0.01; break;
            case 11: config.appearance_momentum = 1.0; break;
            default: config.appearance_momentum = std::numeric_limits<double>::infinity(); break;
            }
            rejects([&] { KalmanTracker invalid(config); }, "Invalid appearance configuration was accepted");
        }
    }

    const auto configuration = appearance_config();
    KalmanTracker tracker(configuration);
    const auto before = tracker.update({described(0.0F, {1.0F, 0.0F})}).front();
    const std::vector<std::vector<float>> malformed{
        {}, {1.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 0.0F},
        {std::numeric_limits<float>::quiet_NaN(), 1.0F},
        {1.0F, std::numeric_limits<float>::infinity()}};
    for (const auto score : {0.20F, 0.40F, 0.90F}) {
        for (const auto& invalid : malformed) {
            // A valid high observation first must not update age, IDs, stats or
            // the appearance prototype before the malformed observation fails.
            rejects([&] { (void)tracker.update({described(1.0F, {0.90F, std::sqrt(0.19F)}),
                                              described(0.0F, invalid, score)}); },
                    "Malformed eligible appearance embedding was accepted");
        }
    }
    const std::vector<Detection> excess(KalmanTracker::max_detections + 1, described(0.0F, {1.0F, 0.0F}));
    rejects([&] { (void)tracker.update(excess); }, "Appearance mode bypassed the detection input cap");
    const auto after = tracker.update({described(0.0F, {1.0F, 0.0F})}).front();
    require(after.track_id == before.track_id && after.age == before.age + 1 &&
                tracker.stats().created_tracks == 1 && tracker.stats().appearance_matches == 1 &&
                tracker.stats().appearance_updates == 1 && tracker.stats().appearance_rejections == 0,
            "Rejected appearance input partially aged state or changed appearance counters");
    const auto probe = tracker.update({described(0.0F, {0.79F, std::sqrt(1.0F - 0.79F * 0.79F)})});
    require(probe.size() == 1 && probe.front().track_id == 2 && tracker.stats().appearance_updates == 1,
            "Rejected appearance input mutated a prototype or consumed an identity");

    KalmanTracker below(configuration);
    require(below.update({described(0.0F, {std::numeric_limits<float>::quiet_NaN()}, 0.09F)}).empty(),
            "Sub-low observation required an unused appearance feature");
    for (const auto mode : {KalmanGateMode::FullBox, KalmanGateMode::CenterOnly}) {
        KalmanTracker disabled(gate_config(mode));
        const auto id = disabled.update({box(0.0F)}).front().track_id;
        require(disabled.update({described(0.0F, {std::numeric_limits<float>::quiet_NaN()})}).front().track_id == id &&
                    disabled.stats().appearance_matches == 0 && disabled.stats().appearance_rejections == 0 &&
                    disabled.stats().appearance_updates == 0,
                "Disabled appearance inspected embeddings or changed its geometric behavior");
    }

    auto immediate = configuration;
    immediate.max_missed_frames = 0;
    KalmanTracker capacity(immediate);
    const std::vector<Detection> births(KalmanTracker::max_tracks + 1, described(0.0F, {1.0F, 0.0F}));
    require(capacity.update(births).size() == KalmanTracker::max_tracks &&
                capacity.stats().capacity_rejections == 1 && capacity.stats().appearance_matches == 0 &&
                capacity.stats().appearance_updates == 0,
            "Appearance prototypes bypassed live capacity or counted births as updates");
    (void)capacity.update({});
    require(capacity.stats().expired_tracks == KalmanTracker::max_tracks &&
                capacity.update({described(0.0F, {1.0F, 0.0F})}).front().track_id == KalmanTracker::max_tracks + 1,
            "Capacity-suppressed appearance identity was consumed or revived after expiry");

    auto exact_config = configuration;
    exact_config.max_cosine_distance = 0.0;
    KalmanTracker norms(exact_config);
    const float maximum = std::numeric_limits<float>::max();
    const float minimum = std::numeric_limits<float>::denorm_min();
    const auto norm_id = norms.update({described(0.0F, {maximum, maximum})}).front().track_id;
    require(norms.update({described(0.0F, {minimum, minimum})}).front().track_id == norm_id &&
                norms.stats().numerical_resets == 0,
            "Finite extreme/subnormal appearance normalization overflowed or underflowed");
    for (const auto dimension : {std::size_t{1}, std::size_t{1024}}) {
        auto boundary = configuration;
        boundary.appearance_dimension = dimension;
        KalmanTracker valid(boundary);
        const auto observation = described(0.0F, std::vector<float>(dimension, 1.0F));
        const auto id = valid.update({observation}).front().track_id;
        require(valid.update({observation}).front().track_id == id,
                "Valid appearance dimension boundary was rejected");
    }
}
}  // namespace

int main() {
    try {
        for (const auto mode : {KalmanGateMode::FullBox, KalmanGateMode::CenterOnly}) {
            lifecycle_and_motion(mode); association_order_and_gating(mode); bounds_and_validation(mode);
        }
        center_gate_size_jitter(); center_gate_nearby_crossing();
        appearance_ambiguous_assignment(); appearance_hard_gates_and_lifecycle(); appearance_high_only_ema();
        appearance_validation_and_bounds();
        std::cout << "Kalman motion, active priority, full-box/center gating, appearance association/EMA, recovery and bounded-state tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
