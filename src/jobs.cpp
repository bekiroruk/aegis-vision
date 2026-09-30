#include "aegisvision/jobs.hpp"
#include "aegisvision/indexing.hpp"
#include <algorithm>
#include <chrono>

namespace aegisvision {
namespace {
std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
bool terminal(const nlohmann::json& value) {
    const auto state = value.at("state");
    return state == "succeeded" || state == "failed" || state == "cancelled";
}
}
JobQueue::JobQueue(Handler handler, std::size_t max_pending, std::size_t max_retained)
    : handler_(std::move(handler)), max_pending_(max_pending), max_retained_(max_retained),
      prefix_(std::to_string(now_ms()) + "-") {
    if (!handler_ || max_pending < 1 || max_retained < max_pending + 1)
        throw std::invalid_argument("Invalid job queue limits or handler");
    worker_ = std::thread([this] { work(); });
}
JobQueue::~JobQueue() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        for (auto& [id, job] : jobs_) { (void)id; job->cancelled = true; }
    }
    ready_.notify_all();
    worker_.join();
}
std::string JobQueue::submit(Json request) {
    std::lock_guard lock(mutex_);
    if (stopping_ || pending_.size() >= max_pending_) throw JobQueueFull();
    while (jobs_.size() >= max_retained_) {
        const auto old = std::find_if(order_.begin(), order_.end(), [&](const std::string& id) {
            return terminal(jobs_.at(id)->value);
        });
        if (old == order_.end()) throw JobQueueFull();
        jobs_.erase(*old); order_.erase(old);
    }
    const auto id = prefix_ + std::to_string(++sequence_);
    auto job = std::make_shared<Job>();
    job->value = {{"id", id}, {"state", "queued"}, {"request", std::move(request)},
        {"created_ms", now_ms()}, {"updated_ms", now_ms()}, {"cancel_requested", false},
        {"progress", Json::object()}, {"result", nullptr}, {"error", nullptr}};
    jobs_.emplace(id, job); order_.push_back(id); pending_.push_back(id);
    ready_.notify_one();
    return id;
}
JobQueue::Json JobQueue::get(const std::string& id) const {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    return found == jobs_.end() ? Json(nullptr) : found->second->value;
}
JobQueue::Json JobQueue::list() const {
    std::lock_guard lock(mutex_);
    auto result = Json::array();
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) result.push_back(jobs_.at(*it)->value);
    return result;
}
bool JobQueue::cancel(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end()) return false;
    auto& job = *found->second;
    if (terminal(job.value)) return true;
    job.cancelled = true;
    job.value["cancel_requested"] = true;
    job.value["updated_ms"] = now_ms();
    if (job.value["state"] == "queued") {
        job.value["state"] = "cancelled";
        std::erase(pending_, id);
    }
    return true;
}
void JobQueue::work() {
    while (true) {
        std::shared_ptr<Job> job;
        Json request;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) return;
            const auto id = pending_.front(); pending_.pop_front();
            job = jobs_.at(id);
            job->value["state"] = "running";
            job->value["updated_ms"] = now_ms();
            request = job->value.at("request");
        }
        const Progress progress = [this, job](Json value) {
            std::lock_guard lock(mutex_);
            job->value["progress"] = std::move(value);
            job->value["updated_ms"] = now_ms();
        };
        try {
            auto result = handler_(request, progress, job->cancelled);
            std::lock_guard lock(mutex_);
            job->value["state"] = job->cancelled ? "cancelled" : "succeeded";
            job->value["result"] = std::move(result);
            job->value["updated_ms"] = now_ms();
        } catch (const IndexCancelled& error) {
            std::lock_guard lock(mutex_);
            job->value["state"] = "cancelled";
            job->value["error"] = error.what();
            job->value["updated_ms"] = now_ms();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex_);
            job->value["state"] = "failed";
            job->value["error"] = error.what();
            job->value["updated_ms"] = now_ms();
        }
    }
}
}
