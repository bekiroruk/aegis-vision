#pragma once
#include <nlohmann/json.hpp>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aegisvision {
class JobStorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct JobPersistence {
    std::filesystem::path database;
    std::string context;
    std::size_t max_pages{65536};
};
// Single process owner, SQLite WAL/FULL. Calls are serialized by JobQueue.
class JobStore {
public:
    explicit JobStore(const JobPersistence& config);
    ~JobStore();
    JobStore(const JobStore&) = delete;
    JobStore& operator=(const JobStore&) = delete;
    std::vector<nlohmann::json> load() const;
    std::string allocate_id();
    void save(const nlohmann::json& job, const std::vector<std::string>& evicted = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
