#pragma once
#include "aegisvision/jobs.hpp"
#include "aegisvision/indexing.hpp"
#include <httplib.h>
#include <filesystem>

namespace aegisvision {
struct ServiceConfig {
    std::filesystem::path media_root;
    std::filesystem::path web_root;
    std::string detector_signature;
    std::size_t max_pending{8};
    std::size_t max_retained{128};
};
// Loopback-only HTTP API. Model/store references must outlive the service.
// All model operations execute on one queue worker, never HTTP worker threads.
class LocalService {
public:
    LocalService(IDetector& detector, IEmbedder& embedder, IVectorStore& store, ServiceConfig config);
    ~LocalService();
    int bind(int port); // 0 chooses a free port, useful for integration tests.
    bool listen();
    void stop();
private:
    using Json = nlohmann::json;
    std::filesystem::path media_path(const std::string& relative) const;
    Json validate(Json request) const;
    Json execute(const Json&, const JobQueue::Progress&, const std::atomic_bool&);
    Json summary(const IndexSummary&) const;
    void routes();
    IDetector& detector_;
    IEmbedder& embedder_;
    IVectorStore& store_;
    ServiceConfig config_;
    JobQueue jobs_;
    httplib::Server http_;
    int port_{};
};
}
