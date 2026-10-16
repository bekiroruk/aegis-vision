#pragma once
#include "aegisvision/image_queue_http.hpp"
#include <filesystem>
namespace aegisvision {
std::shared_ptr<ImageQueueBackend> make_redis_image_queue(const std::filesystem::path& model,
    const std::filesystem::path& media, int port, const std::string& queue);
}
