#pragma once
#include "aegisvision/jobs.hpp"
#include "aegisvision/indexing.hpp"
#include "aegisvision/live_session.hpp"
#include <httplib.h>
#include <filesystem>

namespace aegisvision {
struct ServiceConfig {
    std::filesystem::path media_root;
    std::filesystem::path web_root;
    std::string detector_signature;
    std::size_t max_pending{8};
    std::size_t max_retained{128};
    JobPersistence persistence;
    LiveServiceConfig live;
};
// Loopback-only HTTP API. Model/store references must outlive the service.
// File/search models execute on the queue worker. Live owns a separate model/worker.
class LocalService {
public:
    LocalService(IDetector& detector, IEmbedder& embedder, IVectorStore& store, ServiceConfig config,
        LiveDetectorFactory live_detector = {}, vision::LiveCaptureFactory live_capture = {});
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
    bool accept(const httplib::Request&, httplib::Response&) const;
    IDetector& detector_;
    IEmbedder& embedder_;
    IVectorStore& store_;
    ServiceConfig config_;
    JobQueue jobs_;
    LiveSessions live_;
    httplib::Server http_;
    int port_{};
};
}
