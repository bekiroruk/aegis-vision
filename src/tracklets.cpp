#include "aegisvision/tracklets.hpp"
#include "aegisvision/reid.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace aegisvision {
TrackletCollector::TrackletCollector(std::string camera, std::uint64_t epoch,
    std::string space, std::uint32_t max_missed_frames)
    : camera_(std::move(camera)), space_(std::move(space)), epoch_(epoch), max_missed_(max_missed_frames) {
    if (camera_.empty() || camera_.size() > 64 ||
        camera_.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos ||
        !epoch_ || space_.empty() || space_.size() > 512 || max_missed_ > 10000)
        throw std::invalid_argument("Invalid tracklet camera, epoch, space or timeout");
}
void TrackletCollector::close(std::map<std::uint64_t, Active>::iterator it, const std::string& reason) {
    auto& a = it->second;
    std::vector<float> mean(512);
    for (std::size_t i = 0; i < mean.size(); ++i) mean[i] = static_cast<float>(a.sum[i] / a.count);
    a.tracklet.embedding = normalize_reid(std::move(mean));
    results_.push_back({std::move(a.tracklet), a.count, reason});
    active_.erase(it);
}
void TrackletCollector::update(std::int64_t ms, const std::vector<PersonObservation>& observations) {
    if (finished_ || ms < 0 || ms <= last_ms_ || observations.size() > 512)
        throw std::invalid_argument("Tracklet frames must be strictly increasing and bounded");
    std::set<std::uint64_t> seen;
    std::vector<std::vector<float>> normalized;
    std::size_t added = 0;
    // Validate the entire frame before changing collector state.
    for (const auto& o : observations) {
        if (!o.local_id || !seen.insert(o.local_id).second ||
            std::any_of(results_.begin(), results_.end(), [&](const auto& r) { return r.tracklet.local_id == o.local_id; }))
            throw std::invalid_argument("Duplicate, reused or zero tracklet ID");
        added += !active_.contains(o.local_id);
        normalized.push_back(normalize_reid(o.embedding));
    }
    if (active_.size() + results_.size() + added > 512)
        throw std::invalid_argument("Tracklet batch exceeds 512; split the recording into explicit epochs");
    last_ms_ = ms;
    for (std::size_t j = 0; j < observations.size(); ++j) {
        const auto id = observations[j].local_id;
        auto [it, inserted] = active_.try_emplace(id);
        auto& a = it->second;
        if (inserted) {
            a.tracklet = {camera_, epoch_, id, ms, ms, space_, {}};
            a.sum.assign(512, 0);
        }
        a.tracklet.end_ms = ms;
        a.last_frame = frame_;
        ++a.count;
        for (std::size_t i = 0; i < 512; ++i) a.sum[i] += normalized[j][i];
    }
    for (auto it = active_.begin(); it != active_.end();) {
        auto candidate = it++;
        if (frame_ - candidate->second.last_frame > max_missed_) close(candidate, "missing_timeout");
    }
    ++frame_;
}
void TrackletCollector::finish(const std::string& reason) {
    if (finished_ || (reason != "end_of_input" && reason != "frame_limit"))
        throw std::invalid_argument("Invalid tracklet finish");
    while (!active_.empty()) close(active_.begin(), reason);
    finished_ = true;
}
} // namespace aegisvision
