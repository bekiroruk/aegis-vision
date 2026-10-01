#pragma once
#include "aegisvision/live.hpp"
#include <nlohmann/json.hpp>
#include <mutex>
#include <thread>

namespace aegisvision {
struct LivePreset {
    std::string id, label, url; // URL stays server-side, never returned by the API.
};
struct LiveServiceConfig {
    std::vector<LivePreset> sources;
    vision::LiveConfig stream;
    vision::VideoConfig tracking;
};
using LiveDetectorFactory = std::function<std::unique_ptr<IDetector>()>;
class LiveSessionBusy : public std::runtime_error {
public: LiveSessionBusy() : std::runtime_error("A live session is already active") {}
};
struct LivePreview {
    std::shared_ptr<const std::vector<unsigned char>> jpeg;
    std::uint64_t sequence{}, source_session{};
    int tracking_epoch{};
    double decode_age_ms{};
};
// One bounded, nonpersistent live session, one isolated model instance and one
// immutable latest JPEG. No per-client capture/inference or durable job replay.
class LiveSessions {
public:
    LiveSessions(LiveServiceConfig config = {}, LiveDetectorFactory detector = {}, vision::LiveCaptureFactory capture = {});
    ~LiveSessions();
    nlohmann::json sources() const;
    nlohmann::json current() const; // null before first start; last terminal summary retained.
    nlohmann::json start(const std::string& source_id);
    bool request_stop(const std::string& id);
    bool contains(const std::string& id) const;
    LivePreview preview(const std::string& id) const;
    void shutdown();
private:
    using Clock = std::chrono::steady_clock;
    void run(LivePreset preset);
    nlohmann::json snapshot_locked() const;
    LivePreview preview_locked() const;
    LiveServiceConfig config_;
    LiveDetectorFactory detector_;
    vision::LiveCaptureFactory capture_;
    mutable std::mutex state_mutex_;
    std::mutex operation_mutex_;
    std::thread worker_;
    std::atomic_bool cancel_{false};
    bool active_{false}, closed_{false};
    std::uint64_t next_id_{1};
    std::string id_, source_id_, state_, error_;
    vision::LiveSummary summary_;
    Clock::time_point started_, arrived_, finished_;
    LivePreview preview_;
};
}
