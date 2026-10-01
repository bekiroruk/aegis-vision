#include "aegisvision/jobs.hpp"
#include "aegisvision/indexing.hpp"
#include <algorithm>
#include <chrono>
#include <regex>

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
void valid_job(const nlohmann::json& value) {
    try {
        static const std::regex id_pattern("[0-9]+-[0-9]+");
        const auto id = value.at("id").get<std::string>();
        if (id.size() > 64 || !std::regex_match(id, id_pattern) ||
            !value.at("request").is_object() || !value.at("progress").is_object() ||
            !value.at("cancel_requested").is_boolean() || !value.at("created_ms").is_number_integer() ||
            !value.at("updated_ms").is_number_integer() ||
            !(terminal(value) || value.at("state") == "queued" || value.at("state") == "running"))
            throw JobStorageError("Invalid persisted job fields");
        for (const auto* field : {"attempts", "recoveries"}) {
            if (!value.at(field).is_number_integer() || value.at(field).get<std::int64_t>() < 0 ||
                value.at(field).get<std::int64_t>() > 1000000) throw JobStorageError("Invalid persisted attempt count");
        }
        (void)value.at("result"); (void)value.at("error");
    } catch (const nlohmann::json::exception& error) {
        throw JobStorageError("Invalid persisted job: " + std::string(error.what()));
    }
}
}
JobQueue::JobQueue(Handler handler, std::size_t max_pending, std::size_t max_retained, const JobPersistence& persistence)
    : handler_(std::move(handler)), max_pending_(max_pending), max_retained_(max_retained),
      prefix_(std::to_string(now_ms()) + "-") {
    if (!handler_ || max_pending < 1 || max_pending > 1000 || max_retained < max_pending + 1 || max_retained > 10000)
        throw std::invalid_argument("Invalid job queue limits or handler");
    if (!persistence.database.empty()) {
        store_ = std::make_unique<JobStore>(persistence);
        restore();
    }
    worker_ = std::thread([this] { work(); });
}
void JobQueue::restore() {
    const auto values = store_->load();
    if (values.size() > max_retained_) throw JobStorageError("Persisted history exceeds configured capacity");
    for (auto value : values) {
        valid_job(value);
        const auto id = value.at("id").get<std::string>();
        if (jobs_.contains(id)) throw JobStorageError("Duplicate persisted job ID");
        auto job = std::make_shared<Job>();
        if (!terminal(value)) {
            if (value.at("cancel_requested") == true) {
                value["state"] = "cancelled"; value["error"] = "Cancellation retained across restart";
            } else if (value.at("state") == "running") {
                value["recoveries"] = value.at("recoveries").get<int>() + 1;
                value["state"] = value.at("recoveries").get<int>() > 3 ? "failed" : "queued";
                value["error"] = value.at("state") == "failed" ? "Interrupted retry limit exceeded" : "Interrupted; queued for replay";
            }
            value["updated_ms"] = now_ms();
        }
        job->value = std::move(value); jobs_.emplace(id, job); order_.push_back(id);
        if (!terminal(job->value)) pending_.push_back(id);
    }
    if (pending_.size() > max_pending_ + 1) throw JobStorageError("Persisted pending work exceeds configured capacity");
    for (const auto& id : order_) store_->save(jobs_.at(id)->value);
}
JobQueue::~JobQueue() {
    {
        std::lock_guard lock(mutex_); stopping_ = true;
        for (auto& [id, job] : jobs_) { (void)id; job->cancelled = true; }
    }
    ready_.notify_all(); worker_.join();
}
void JobQueue::save_locked(const Json& value, const std::vector<std::string>& evicted) {
    if (!store_) return;
    try { store_->save(value, evicted); }
    catch (const JobStorageError& error) {
        storage_error_ = error.what(); stopping_ = true;
        for (auto& [id, job] : jobs_) { (void)id; job->cancelled = true; }
        ready_.notify_all(); throw;
    }
}
std::string JobQueue::submit(Json request) {
    std::lock_guard lock(mutex_);
    if (!storage_error_.empty()) throw JobStorageError(storage_error_);
    if (stopping_ || pending_.size() >= max_pending_) throw JobQueueFull();
    std::vector<std::string> evicted;
    if (jobs_.size() >= max_retained_) {
        const auto old = std::find_if(order_.begin(), order_.end(), [&](const std::string& id) { return terminal(jobs_.at(id)->value); });
        if (old == order_.end()) throw JobQueueFull();
        evicted.push_back(*old);
    }
    std::string id;
    try { id = store_ ? store_->allocate_id() : prefix_ + std::to_string(++sequence_); }
    catch (const JobStorageError& error) {
        storage_error_ = error.what(); stopping_ = true;
        for (auto& [old, job] : jobs_) { (void)old; job->cancelled = true; }
        ready_.notify_all(); throw;
    }
    auto job = std::make_shared<Job>();
    job->value = {{"id", id}, {"state", "queued"}, {"request", std::move(request)},
        {"created_ms", now_ms()}, {"updated_ms", now_ms()}, {"cancel_requested", false},
        {"attempts", 0}, {"recoveries", 0}, {"progress", Json::object()}, {"result", nullptr}, {"error", nullptr}};
    save_locked(job->value, evicted);
    for (const auto& old : evicted) { jobs_.erase(old); std::erase(order_, old); }
    jobs_.emplace(id, job); order_.push_back(id); pending_.push_back(id);
    ready_.notify_one(); return id;
}
JobQueue::Json JobQueue::get(const std::string& id) const {
    std::lock_guard lock(mutex_); const auto found = jobs_.find(id);
    return found == jobs_.end() ? Json(nullptr) : found->second->value;
}
JobQueue::Json JobQueue::list() const {
    std::lock_guard lock(mutex_); auto result = Json::array();
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) result.push_back(jobs_.at(*it)->value);
    return result;
}
std::string JobQueue::storage_error() const { std::lock_guard lock(mutex_); return storage_error_; }
bool JobQueue::cancel(const std::string& id) {
    std::lock_guard lock(mutex_);
    if (!storage_error_.empty()) throw JobStorageError(storage_error_);
    const auto found = jobs_.find(id); if (found == jobs_.end()) return false;
    auto& job = *found->second; if (terminal(job.value)) return true;
    auto value = job.value; value["cancel_requested"] = true; value["updated_ms"] = now_ms();
    if (value["state"] == "queued") value["state"] = "cancelled";
    save_locked(value); job.value = std::move(value); job.cancelled = true;
    if (job.value["state"] == "cancelled") std::erase(pending_, id);
    return true;
}
void JobQueue::finish_locked(Job& job, const char* state, Json result, Json error) {
    auto value = job.value;
    if (store_ && stopping_ && storage_error_.empty() && value.at("cancel_requested") == false) {
        value["state"] = "queued"; value["error"] = "Service stopped; queued for replay";
    } else { value["state"] = state; value["error"] = std::move(error); }
    value["result"] = std::move(result); value["updated_ms"] = now_ms();
    save_locked(value); job.value = std::move(value);
}
void JobQueue::work() {
    while (true) {
        std::shared_ptr<Job> job; Json request;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_) return;
            job = jobs_.at(pending_.front()); auto value = job->value;
            value["state"] = "running"; value["updated_ms"] = now_ms();
            value["attempts"] = value.at("attempts").get<int>() + 1; value["error"] = nullptr;
            try { save_locked(value); } catch (const JobStorageError&) { return; }
            pending_.pop_front(); job->value = std::move(value);
            job->checkpoint = std::chrono::steady_clock::now(); request = job->value.at("request");
        }
        const Progress progress = [this, job](Json update) {
            std::lock_guard lock(mutex_); auto value = job->value;
            value["progress"] = std::move(update); value["updated_ms"] = now_ms();
            const auto current = std::chrono::steady_clock::now();
            if (current - job->checkpoint >= std::chrono::seconds(1)) {
                save_locked(value); job->checkpoint = current;
            }
            job->value = std::move(value);
        };
        try {
            try {
                auto result = handler_(request, progress, job->cancelled);
                std::lock_guard lock(mutex_);
                finish_locked(*job, job->cancelled ? "cancelled" : "succeeded", std::move(result), nullptr);
            } catch (const JobStorageError&) { throw;
            } catch (const IndexCancelled& error) {
                std::lock_guard lock(mutex_); finish_locked(*job, "cancelled", nullptr, error.what());
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex_); finish_locked(*job, "failed", nullptr, error.what());
            }
        } catch (const JobStorageError&) { return; }
    }
}
}
