#pragma once
#include "aegisvision/redis_worker.hpp"
#include "aegisvision/yolo.hpp"
#include <filesystem>
#include <memory>

namespace aegisvision {
// Trusted, immutable local media/model files. One instance per inference thread.
class RedisImageWorker {
public:
    RedisImageWorker(const std::filesystem::path& model, const std::filesystem::path& media_root);
    const std::string& context() const { return context_; }
    nlohmann::json request(const std::string& relative_image) const;
    nlohmann::json analyze(const nlohmann::json& payload, const StopCheck& cancelled);
private:
    std::vector<unsigned char> read_image(const std::string& relative) const;
    std::filesystem::path root_;
    std::vector<unsigned char> model_bytes_;
    std::string model_sha_, context_;
    std::unique_ptr<vision::YoloDetector> detector_;
};
}
