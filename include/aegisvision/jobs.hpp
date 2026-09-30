#pragma once
#include <nlohmann/json.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace aegisvision {
class JobQueueFull : public std::runtime_error {
public:
    JobQueueFull() : std::runtime_error("Job queue is full; retry later") {}
};
// One worker owns model access. Pending work and retained history are bounded.
// History is process-local; indexed Qdrant records persist separately.
class JobQueue {
public:
    using Json = nlohmann::json;
    using Progress = std::function<void(Json)>;
    using Handler = std::function<Json(const Json&, const Progress&, const std::atomic_bool&)>;
    JobQueue(Handler handler, std::size_t max_pending = 8, std::size_t max_retained = 128);
    ~JobQueue();
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;
    [[nodiscard]] std::string submit(Json request);
    [[nodiscard]] Json get(const std::string& id) const;
    [[nodiscard]] Json list() const;
    // Returns false for unknown IDs. A running job acknowledges cancellation at a checkpoint.
    bool cancel(const std::string& id);
private:
    struct Job {
        Json value;
        std::atomic_bool cancelled{false};
    };
    void work();
    Handler handler_;
    std::size_t max_pending_, max_retained_;
    std::string prefix_;
    std::uint64_t sequence_{};
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> pending_, order_;
    bool stopping_{false};
    std::thread worker_;
};
}
