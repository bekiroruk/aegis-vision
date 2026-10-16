#include "aegisvision/redis_worker.hpp"
#include "aegisvision/redis_queue.hpp"
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace aegisvision {
WorkerOutcome run_redis_job(const QueueCall& call, const JobHandler& handler,
    const std::string& owner, int lease_ms, StopCheck stopping) {
    using namespace std::chrono;
    // Leave enough room for the transport's one-second timeout.
    if (lease_ms < 3000 || lease_ms > 300000)
        throw std::invalid_argument("Worker lease must be 3000..300000 ms");
    const nlohmann::json claim{{"op", "claim"}, {"owner", owner}, {"lease_ms", lease_ms}};
    RedisLeaseQueue::validate(claim);
    if (stopping()) return WorkerOutcome::Empty;
    const auto job = call(claim); // An uncertain claim is never immediately retried.
    if (job.is_null()) return WorkerOutcome::Empty;
    const auto id = job.at("id").get<std::string>();
    const auto token = job.at("token").get<std::string>();
    std::atomic<bool> lost{false};
    const StopCheck cancelled = [&] { return lost.load() || stopping(); };
    std::jthread heartbeat([&](std::stop_token stop) {
        auto next = steady_clock::now() + milliseconds(lease_ms / 3);
        while (!stop.stop_requested()) {
            if (stopping()) { lost.store(true); break; }
            if (steady_clock::now() >= next) {
                try {
                    if (call({{"op", "renew"}, {"id", id}, {"owner", owner},
                        {"token", token}, {"lease_ms", lease_ms}}).is_null()) {
                        lost.store(true); break;
                    }
                } catch (...) { lost.store(true); break; }
                next = steady_clock::now() + milliseconds(lease_ms / 3);
            }
            std::this_thread::sleep_for(milliseconds(20));
        }
    });
    nlohmann::json result;
    bool failed = false;
    try {
        if (!cancelled()) result = handler(job.at("payload"), cancelled);
        if (!cancelled()) RedisLeaseQueue::validate({{"op", "complete"}, {"id", id},
            {"owner", owner}, {"token", token}, {"result", result}});
    } catch (...) {
        failed = true;
        result = {{"error", "analysis_failed"}};
    }
    heartbeat.request_stop();
    heartbeat.join(); // Never let a late renew race a successful completion.
    if (cancelled()) return WorkerOutcome::Lost;
    const auto published = call({{"op", failed ? "fail" : "complete"}, {"id", id},
        {"owner", owner}, {"token", token}, {"result", result}});
    if (published.is_null()) return WorkerOutcome::Lost;
    return failed ? WorkerOutcome::Failed : WorkerOutcome::Completed;
}
}
