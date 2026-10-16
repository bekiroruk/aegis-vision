#pragma once
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <stdexcept>

namespace aegisvision {
class ImageQueueError : public std::runtime_error {
public:
    ImageQueueError(int status, const std::string& message) : std::runtime_error(message), status(status) {}
    int status;
};
class ImageQueueBackend {
public:
    virtual ~ImageQueueBackend() = default;
    virtual nlohmann::json status() const = 0;
    virtual nlohmann::json submit(const std::string& id, const std::string& image) const = 0;
    virtual nlohmann::json get(const std::string& id) const = 0;
    virtual nlohmann::json cancel(const std::string& id) const = 0;
    virtual std::vector<unsigned char> preview(const std::string& id) const = 0;
};
// Same route implementation in LocalService and the lightweight integration fixture.
// Always enforce loopback Host/Origin and JSON POST, including when backend is disabled.
void register_image_queue_routes(httplib::Server& http_, std::shared_ptr<ImageQueueBackend> backend);
}
