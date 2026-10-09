#include "aegisvision/tracking.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <numeric>

namespace aegisvision {

IoUTracker::IoUTracker(const float iou_threshold, const std::uint32_t max_missed_frames)
    : iou_threshold_(iou_threshold), max_missed_frames_(max_missed_frames) {
    if (!std::isfinite(iou_threshold) || iou_threshold < 0.0F || iou_threshold > 1.0F) {
        throw std::invalid_argument("IoU threshold must be in [0, 1]");
    }
}

std::vector<Track> IoUTracker::update(const std::vector<Detection>& detections) {
    auto output = update_indexed(detections);
    std::ranges::sort(output, std::less{}, &Track::track_id);
    return output;
}

std::vector<Track> IoUTracker::update_indexed(const std::vector<Detection>& detections) {
    std::vector<std::size_t> ordered(detections.size());
    std::iota(ordered.begin(), ordered.end(), 0);
    std::stable_sort(ordered.begin(), ordered.end(), [&](auto a, auto b) {
        return detections[a].score > detections[b].score;
    });

    std::unordered_set<std::uint64_t> unmatched;
    for (const auto& [track_id, track] : tracks_) {
        static_cast<void>(track);
        unmatched.insert(track_id);
    }

    std::vector<Track> output(ordered.size());
    for (const auto index : ordered) {
        const auto& detection = detections[index];
        std::uint64_t best_id = 0;
        float best_iou = 0.0F;
        for (const auto track_id : unmatched) {
            const auto& candidate = tracks_.at(track_id);
            const auto overlap = candidate.label == detection.label
                ? detection.bbox.iou(candidate.bbox)
                : 0.0F;
            if (overlap > best_iou || (overlap > 0 && overlap == best_iou && track_id < best_id)) {
                best_iou = overlap;
                best_id = track_id;
            }
        }

        Track track;
        if (best_id != 0 && best_iou >= iou_threshold_) {
            const auto previous_age = tracks_.at(best_id).age;
            track = Track{best_id, detection.bbox, detection.label, detection.score, previous_age + 1, 0};
            unmatched.erase(best_id);
        } else {
            track = Track{next_id_++, detection.bbox, detection.label, detection.score, 1, 0};
        }
        tracks_.insert_or_assign(track.track_id, track);
        output[index] = std::move(track);
    }

    for (const auto track_id : unmatched) {
        auto& track = tracks_.at(track_id);
        ++track.age;
        ++track.missed_frames;
    }
    std::erase_if(tracks_, [this](const auto& entry) {
        return entry.second.missed_frames > max_missed_frames_;
    });
    return output;
}

}  // namespace aegisvision

