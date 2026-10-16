#pragma once
#include <functional>
#include <nlohmann/json.hpp>

namespace aegisvision {
enum class WorkerOutcome { Empty, Completed, Failed, Lost };
// Call must support simultaneous claim/renew calls. Handler runs on one thread.
// A handler must not publish external side effects: only the fenced result is safe.
using QueueCall = std::function<nlohmann::json(const nlohmann::json&)>;
using StopCheck = std::function<bool()>;
using JobHandler = std::function<nlohmann::json(const nlohmann::json&, const StopCheck&)>;
WorkerOutcome run_redis_job(const QueueCall& call, const JobHandler& handler,
    const std::string& owner, int lease_ms = 30000, StopCheck stopping = [] { return false; });
}
