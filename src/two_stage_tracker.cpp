#include "aegisvision/two_stage_tracker.hpp"
#include "aegisvision/assignment.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace aegisvision {
TwoStageTracker::TwoStageTracker(TwoStageConfig config) : config_(config) {
    for (float value : {config.low_threshold, config.high_threshold, config.new_track_threshold, config.match_iou}) {
        if (!std::isfinite(value) || value <= 0 || value > 1) {
            throw std::invalid_argument("Tracker thresholds must be finite and in (0,1]");
        }
    }
    if (config.low_threshold >= config.high_threshold || config.high_threshold > config.new_track_threshold ||
        config.max_missed_frames > 10000) {
        throw std::invalid_argument("Require low < high <= new-track threshold; missed frames <= 10000");
    }
}

std::vector<Track> TwoStageTracker::update(const std::vector<Detection>& detections) {
    // Validate before any state mutation; malformed detector results cannot poison tracks.
    for (const auto& detection : detections) {
        const auto& b = detection.bbox;
        if (detection.label.empty() || !std::isfinite(detection.score) || detection.score < 0 || detection.score > 1 ||
            !std::isfinite(b.x1) || !std::isfinite(b.x2) || !std::isfinite(b.y1) || !std::isfinite(b.y2) ||
            b.x2 <= b.x1 || b.y2 <= b.y1 || !std::isfinite(b.area())) {
            throw std::invalid_argument("Invalid detection for two-stage tracker");
        }
    }
    std::vector<std::size_t> high, low;
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].score >= config_.high_threshold) high.push_back(i);
        else if (detections[i].score >= config_.low_threshold) low.push_back(i);
    }
    // Stable input order resolves score ties; map order resolves track ID ties.
    const auto sort_scores = [&](auto& indices) {
        std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) { return detections[a].score > detections[b].score; });
    };
    sort_scores(high); sort_scores(low);
    std::map<std::uint64_t, BoundingBox> predicted;
    std::vector<std::uint64_t> all_ids;
    for (const auto& [id, state] : states_) {
        all_ids.push_back(id);
        const auto& box = state.track.bbox;
        const float steps = static_cast<float>(state.track.missed_frames + 1);
        const float dx = config_.predict_motion ? state.velocity_x * steps : 0;
        const float dy = config_.predict_motion ? state.velocity_y * steps : 0;
        const bool finite = std::isfinite(box.x1 + dx) && std::isfinite(box.x2 + dx) &&
            std::isfinite(box.y1 + dy) && std::isfinite(box.y2 + dy);
        predicted.emplace(id, finite ? BoundingBox{box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy} : box);
    }
    std::set<std::uint64_t> matched_ids;
    std::set<std::size_t> matched_detections;
    std::vector<Track> visible;
    const auto associate = [&](const std::vector<std::uint64_t>& ids,
                               const std::vector<std::size_t>& indices, bool low_stage) {
        std::vector<std::vector<double>> weights(ids.size(), std::vector<double>(indices.size(), -1.0));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            for (std::size_t j = 0; j < indices.size(); ++j) {
                const auto& d = detections[indices[j]];
                if (states_.at(ids[i]).track.label != d.label) continue;
                const float overlap = predicted.at(ids[i]).iou(d.bbox);
                if (overlap >= config_.match_iou) weights[i][j] = overlap;
            }
        }
        for (const auto& [i, j] : assign_max_weight(weights)) {
            const auto id = ids[i];
            const auto index = indices[j];
            auto& state = states_.at(id);
            const auto& detection = detections[index];
            const auto& old = state.track.bbox;
            const auto& next = detection.bbox;
            const float gap = static_cast<float>(state.track.missed_frames + 1);
            const float vx = ((next.x1 - old.x1) * 0.5F + (next.x2 - old.x2) * 0.5F) / gap;
            const float vy = ((next.y1 - old.y1) * 0.5F + (next.y2 - old.y2) * 0.5F) / gap;
            state.velocity_x = state.track.age == 1 ? vx : 0.7F * vx + 0.3F * state.velocity_x;
            state.velocity_y = state.track.age == 1 ? vy : 0.7F * vy + 0.3F * state.velocity_y;
            if (low_stage) ++stats_.low_confidence_matches;
            if (state.track.missed_frames > 0) ++stats_.reactivations;
            state.track = {id, next, detection.label, detection.score, state.track.age + 1, 0};
            visible.push_back(state.track);
            matched_ids.insert(id);
            matched_detections.insert(index);
        }
    };
    associate(all_ids, high, false);
    std::vector<std::uint64_t> active_unmatched;
    for (auto id : all_ids) {
        if (!matched_ids.contains(id) && states_.at(id).track.missed_frames == 0) active_unmatched.push_back(id);
    }
    associate(active_unmatched, low, true);
    for (auto id : all_ids) {
        if (matched_ids.contains(id)) continue;
        auto& track = states_.at(id).track;
        ++track.age;
        if (++track.missed_frames > config_.max_missed_frames) { states_.erase(id); ++stats_.expired_tracks; }
    }
    for (auto index : high) {
        if (matched_detections.contains(index) || detections[index].score < config_.new_track_threshold) continue;
        const auto& d = detections[index];
        Track track{next_id_++, d.bbox, d.label, d.score, 1, 0};
        states_.emplace(track.track_id, State{track});
        visible.push_back(track);
        ++stats_.created_tracks;
    }
    std::sort(visible.begin(), visible.end(), [](const auto& a, const auto& b) { return a.track_id < b.track_id; });
    return visible;
}
}
