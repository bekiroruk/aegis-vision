#include "aegisvision/image_queue_http.hpp"

namespace aegisvision {
namespace {
using Json = nlohmann::json;
void reply(httplib::Response& r, const Json& value, int status = 200) {
    r.status = status;
    r.set_header("Cache-Control", "no-store");
    r.set_header("X-Content-Type-Options", "nosniff");
    r.set_content(value.dump(), "application/json; charset=utf-8");
}
bool accept(const httplib::Request& req, httplib::Response& res) {
    const auto host = req.get_header_value("Host"), origin = req.get_header_value("Origin");
    const auto port = std::to_string(req.local_port);
    if ((host != "127.0.0.1:" + port && host != "localhost:" + port)
        || (!origin.empty() && origin != "http://" + host)) {
        reply(res, {{"error", "Local same-origin requests only"}}, 403); return false;
    }
    if (req.body.size() > 8192) { reply(res, {{"error", "Request too large"}}, 413); return false; }
    if (req.method == "POST" && !req.get_header_value("Content-Type").starts_with("application/json")) {
        reply(res, {{"error", "Content-Type must be application/json"}}, 415); return false;
    }
    return true;
}
template<class F> auto guarded(F fn) {
    return [fn](const httplib::Request& req, httplib::Response& res) {
        if (!accept(req, res)) return;
        try { fn(req, res); }
        catch (const ImageQueueError& e) { reply(res, {{"error", e.what()}}, e.status); }
        catch (const Json::exception&) { reply(res, {{"error", "Invalid JSON request"}}, 400); }
        catch (const std::invalid_argument&) { reply(res, {{"error", "Invalid image request"}}, 400); }
        catch (const std::exception&) { reply(res, {{"error", "Image queue unavailable; retry with the same job ID"}}, 503); }
    };
}
void enabled(const std::shared_ptr<ImageQueueBackend>& b) {
    if (!b) throw ImageQueueError(503, "Redis image queue is not enabled");
}
}
void register_image_queue_routes(httplib::Server& http_, std::shared_ptr<ImageQueueBackend> b) {
    http_.Get("/api/redis-images", guarded([b](const auto&, auto& res) {
        reply(res, b ? b->status() : Json{{"enabled", false}});
    }));
    http_.Post("/api/redis-images", guarded([b](const auto& req, auto& res) {
        enabled(b);
        const auto body = Json::parse(req.body);
        if (!body.is_object() || body.size() != 2 || !body.contains("id") || !body.at("id").is_string()
            || !body.contains("image") || !body.at("image").is_string())
            throw std::invalid_argument("Expected id and image");
        const auto id=body.at("id").template get<std::string>();
        if(id.empty() || id.size()>64 || id.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")!=std::string::npos)
            throw std::invalid_argument("Invalid job ID");
        const auto image=body.at("image").template get<std::string>();
        if(image.empty() || image.size()>512 || image.find('\0')!=std::string::npos) throw std::invalid_argument("Invalid image path");
        const auto job = b->submit(body.at("id"), body.at("image"));
        res.set_header("Location", "/api/redis-images/" + job.at("id").template get<std::string>());
        reply(res, job, 202);
    }));
    http_.Get(R"(/api/redis-images/([A-Za-z0-9_-]{1,64}))", guarded([b](const auto& req, auto& res) {
        enabled(b); reply(res, b->get(req.matches[1]));
    }));
    http_.Post(R"(/api/redis-images/([A-Za-z0-9_-]{1,64})/cancel)", guarded([b](const auto& req, auto& res) {
        enabled(b);
        const auto body = Json::parse(req.body);
        if (!body.is_object() || !body.empty()) throw std::invalid_argument("Expected empty object");
        reply(res, b->cancel(req.matches[1]));
    }));
    http_.Get(R"(/api/redis-images/([A-Za-z0-9_-]{1,64})/preview\.jpg)", guarded([b](const auto& req, auto& res) {
        enabled(b); const auto bytes = b->preview(req.matches[1]);
        res.set_header("Cache-Control", "no-store");
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/jpeg");
    }));
}
}
