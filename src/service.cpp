#include "aegisvision/service.hpp"
#include "aegisvision/vision.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>

namespace aegisvision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string utf8(const fs::path& path) {
    const auto text = path.generic_u8string(); return {text.begin(), text.end()};
}
void reply(httplib::Response& response, const Json& value, int status = 200) {
    response.status = status; response.set_content(value.dump(), "application/json; charset=utf-8");
}
bool video_extension(const fs::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".mp4" || extension == ".avi" || extension == ".mov" || extension == ".mkv" || extension == ".webm";
}
int integer(const Json& value, const char* key, int fallback, int minimum, int maximum) {
    if (!value.contains(key)) return fallback;
    if (!value.at(key).is_number_integer()) throw std::invalid_argument(std::string(key) + " must be an integer");
    const auto n = value.at(key).get<std::int64_t>();
    if (n < minimum || n > maximum) throw std::invalid_argument(std::string(key) + " out of range");
    return static_cast<int>(n);
}
}
LocalService::LocalService(IDetector& detector, IEmbedder& embedder, IVectorStore& store, ServiceConfig config)
    : detector_(detector), embedder_(embedder), store_(store), config_(std::move(config)),
      jobs_([this](const Json& request, const JobQueue::Progress& progress, const std::atomic_bool& cancel) {
          return execute(request, progress, cancel);
      }, config_.max_pending, config_.max_retained) {
    if (!fs::is_directory(config_.media_root) || !fs::is_directory(config_.web_root) || config_.detector_signature.empty())
        throw std::invalid_argument("Service requires media/web directories and a detector signature");
    config_.media_root = fs::canonical(config_.media_root);
    config_.web_root = fs::canonical(config_.web_root);
    routes();
}
LocalService::~LocalService() { stop(); }
int LocalService::bind(int port) {
    if (port < 0 || port > 65535) throw std::invalid_argument("Invalid HTTP port");
    if (port_) throw std::logic_error("Service already bound");
    port_ = port == 0 ? http_.bind_to_any_port("127.0.0.1") :
        (http_.bind_to_port("127.0.0.1", port) ? port : -1);
    if (port_ <= 0) throw std::runtime_error("Cannot bind local HTTP port");
    return port_;
}
bool LocalService::listen() { return http_.listen_after_bind(); }
void LocalService::stop() { http_.stop(); }
fs::path LocalService::media_path(const std::string& relative) const {
    const auto requested = fs::u8path(relative);
    if (relative.empty() || requested.is_absolute() || requested.has_root_name())
        throw std::invalid_argument("Use a relative video path inside media_root");
    const auto path = fs::weakly_canonical(config_.media_root / requested);
    const auto inside = path.lexically_relative(config_.media_root);
    if (inside.empty() || inside.is_absolute() || *inside.begin() == ".." ||
        !fs::is_regular_file(path) || !video_extension(path))
        throw std::invalid_argument("Video must exist inside media_root with a supported extension");
    return path;
}
LocalService::Json LocalService::validate(Json request) const {
    if (!request.is_object() || !request.contains("type") || !request.at("type").is_string())
        throw std::invalid_argument("Expected an object with a string type");
    const auto type = request.at("type").get<std::string>();
    const std::set<std::string> allowed = type == "index_video" ?
        std::set<std::string>{"type", "path", "stride", "max_frames"} : std::set<std::string>{"type", "query", "limit"};
    for (const auto& [key, value] : request.items()) {
        (void)value; if (!allowed.contains(key)) throw std::invalid_argument("Unknown field: " + key);
    }
    if (type == "index_video") {
        (void)media_path(request.at("path").get<std::string>());
        request["stride"] = integer(request, "stride", 15, 1, 10000);
        request["max_frames"] = integer(request, "max_frames", 0, 0, 1000000);
    } else if (type == "search") {
        const auto query = request.at("query").get<std::string>();
        if (query.empty() || query.size() > 1024 || query.find_first_not_of(" \r\n\t") == std::string::npos)
            throw std::invalid_argument("query must contain 1..1024 bytes of nonblank text");
        request["limit"] = integer(request, "limit", 8, 1, 20);
    } else throw std::invalid_argument("type must be index_video or search");
    return request;
}
LocalService::Json LocalService::summary(const IndexSummary& value) const {
    return {{"source_id", value.source_id}, {"decoded_frames", value.decoded_frames},
        {"sampled_frames", value.sampled_frames}, {"indexed_items", value.indexed_items},
        {"source_fps", value.source_fps}, {"used_fallback_fps", value.used_fallback_fps}, {"stop_reason", value.stop_reason}};
}
LocalService::Json LocalService::execute(const Json& request, const JobQueue::Progress& progress, const std::atomic_bool& cancel) {
    if (cancel) throw IndexCancelled();
    if (request.at("type") == "index_video") {
        VideoIndexConfig options;
        options.detector_signature = config_.detector_signature;
        options.frame_stride = request.at("stride").get<int>();
        options.max_frames = request.at("max_frames").get<int>();
        const auto result = index_video(media_path(request.at("path").get<std::string>()), detector_, embedder_, store_, options,
            [&](const IndexSummary& value) { progress(summary(value)); }, [&] { return cancel.load(); });
        progress(summary(result));
        return summary(result);
    }
    auto vector = embedder_.embed_text(request.at("query").get<std::string>());
    if (cancel) throw IndexCancelled();
    auto results = Json::array();
    for (const auto& match : store_.search(vector, request.at("limit").get<std::size_t>())) {
        Json result{{"id", match.item_id}, {"score", match.score}, {"metadata", match.metadata}};
        // Only videos within this service's media root get a playable URL/preview.
        if (match.metadata.contains("path") && match.metadata.contains("frame_index")) {
            try {
                const auto relative = fs::weakly_canonical(fs::u8path(match.metadata.at("path"))).lexically_relative(config_.media_root);
                (void)media_path(utf8(relative));
                result["media_path"] = utf8(relative);
            } catch (const std::exception&) { }
        }
        results.push_back(std::move(result));
    }
    return {{"query", request.at("query")}, {"results", results}};
}
void LocalService::routes() {
    http_.new_task_queue = [] { return new httplib::ThreadPool(4, 16); };
    http_.set_payload_max_length(8192);
    http_.set_read_timeout(5); http_.set_write_timeout(15);
    http_.set_default_headers({{"X-Content-Type-Options", "nosniff"}, {"Cache-Control", "no-store"},
        {"Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self'; frame-ancestors 'none'"}});
    http_.set_pre_routing_handler([this](const auto& request, auto& response) {
        const auto host = request.get_header_value("Host");
        const auto local = "127.0.0.1:" + std::to_string(port_);
        const auto localhost = "localhost:" + std::to_string(port_);
        const auto origin = request.get_header_value("Origin");
        if ((host != local && host != localhost) || (!origin.empty() && origin != "http://" + host)) {
            reply(response, {{"error", "Local same-origin requests only"}}, 403);
            return httplib::Server::HandlerResponse::Handled;
        }
        if (request.method == "POST" && request.get_header_value("Content-Type").find("application/json") != 0) {
            reply(response, {{"error", "Content-Type must be application/json"}}, 415);
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http_.set_exception_handler([](const auto&, auto& response, std::exception_ptr error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& problem) { reply(response, {{"error", problem.what()}}, 500); }
    });
    http_.Get("/api/health", [this](const auto&, auto& response) {
        reply(response, {{"status", "ready"}, {"worker_count", 1}, {"queue_capacity", config_.max_pending},
            {"history_capacity", config_.max_retained}, {"history_persistent", false}});
    });
    http_.Get("/api/media", [this](const auto&, auto& response) {
        auto files = Json::array();
        std::vector<std::string> names;
        for (const auto& entry : fs::directory_iterator(config_.media_root))
            if (!entry.is_symlink() && entry.is_regular_file() && video_extension(entry.path())) names.push_back(utf8(entry.path().filename()));
        std::sort(names.begin(), names.end());
        for (const auto& name : names) files.push_back({{"path", name}});
        reply(response, {{"videos", files}});
    });
    http_.Post("/api/jobs", [this](const auto& request, auto& response) {
        try {
            const auto id = jobs_.submit(validate(Json::parse(request.body)));
            response.set_header("Location", "/api/jobs/" + id);
            reply(response, jobs_.get(id), 202);
        } catch (const JobQueueFull& error) { response.set_header("Retry-After", "2"); reply(response, {{"error", error.what()}}, 429);
        } catch (const std::exception& error) { reply(response, {{"error", error.what()}}, 400); }
    });
    http_.Get("/api/jobs", [this](const auto&, auto& response) { reply(response, {{"jobs", jobs_.list()}}); });
    http_.Get(R"(/api/jobs/([0-9]+-[0-9]+))", [this](const auto& request, auto& response) {
        const auto job = jobs_.get(request.matches[1]);
        reply(response, job.is_null() ? Json{{"error", "Job not found or expired"}} : job, job.is_null() ? 404 : 200);
    });
    http_.Post(R"(/api/jobs/([0-9]+-[0-9]+)/cancel)", [this](const auto& request, auto& response) {
        const std::string id = request.matches[1];
        if (!jobs_.cancel(id)) reply(response, {{"error", "Job not found"}}, 404);
        else reply(response, jobs_.get(id));
    });
    http_.Get(R"(/media/(.+))", [this](const auto& request, auto& response) {
        try {
            const auto path = media_path(request.matches[1]);
            const auto file = std::make_shared<std::ifstream>(path, std::ios::binary);
            if (!*file) throw std::runtime_error("Cannot open video");
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            const char* mime = extension == ".mp4" ? "video/mp4" : extension == ".webm" ? "video/webm" :
                extension == ".mov" ? "video/quicktime" : extension == ".avi" ? "video/x-msvideo" : "video/x-matroska";
            response.set_header("Accept-Ranges", "bytes");
            // Set length before httplib validates ranges; set_file_content in the
            // pinned version initializes its provider after range validation.
            response.set_content_provider(static_cast<std::size_t>(fs::file_size(path)), mime,
                [file](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                    file->clear(); file->seekg(static_cast<std::streamoff>(offset));
                    std::array<char, 65536> buffer{};
                    while (length > 0) {
                        const auto count = std::min(length, buffer.size());
                        if (!file->read(buffer.data(), static_cast<std::streamsize>(count)) ||
                            !sink.write(buffer.data(), count)) return false;
                        length -= count;
                    }
                    return true;
                });
        }
        catch (const std::exception& error) { reply(response, {{"error", error.what()}}, 404); }
    });
    http_.Get(R"(/api/preview/([0-9]+-[0-9]+)/([0-9]+)\.jpg)", [this](const auto& request, auto& response) {
        try {
            const auto job = jobs_.get(request.matches[1]);
            if (job.is_null() || job.at("state") != "succeeded") throw std::runtime_error("Completed search job required");
            const auto& result = job.at("result").at("results").at(std::stoul(request.matches[2]));
            const auto path = media_path(result.at("media_path").get<std::string>());
            const auto metadata = result.at("metadata");
            const double index = std::stod(metadata.at("frame_index").get<std::string>());
            if (!std::isfinite(index) || index < 0 || index > 1000000) throw std::invalid_argument("Invalid frame index");
            cv::VideoCapture video(utf8(path));
            if (!video.isOpened() || !video.set(cv::CAP_PROP_POS_FRAMES, index)) throw std::runtime_error("Cannot seek video");
            cv::Mat frame;
            if (!video.read(frame)) throw std::runtime_error("Cannot decode result frame");
            const auto b = Json::parse(metadata.at("bbox").get<std::string>());
            const Detection detection{{b.at(0).get<float>(), b.at(1).get<float>(), b.at(2).get<float>(), b.at(3).get<float>()},
                metadata.at("label").get<std::string>(), std::stof(metadata.at("confidence").get<std::string>()), {}, {}};
            std::vector<unsigned char> bytes;
            if (!cv::imencode(".jpg", vision::annotate(frame, {detection}), bytes)) throw std::runtime_error("Cannot encode preview");
            response.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/jpeg");
        } catch (const std::exception& error) { reply(response, {{"error", error.what()}}, 404); }
    });
    if (!http_.set_mount_point("/", utf8(config_.web_root))) throw std::runtime_error("Cannot serve web assets");
}
}
