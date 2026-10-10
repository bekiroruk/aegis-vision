#include "aegisvision/service.hpp"
#include "aegisvision/vision.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <regex>

namespace aegisvision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
std::string jpeg_url(const cv::Mat& image) {
    cv::Mat preview=image;
    const double scale=std::min(1.,960./std::max(image.cols,image.rows));
    if (scale<1.) cv::resize(image,preview,{},scale,scale,cv::INTER_AREA);
    std::vector<unsigned char> bytes;
    if (!cv::imencode(".jpg",preview,bytes,{cv::IMWRITE_JPEG_QUALITY,80}) || bytes.size()>512*1024)
        throw std::runtime_error("Frame preview exceeds JPEG limit");
    constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result="data:image/jpeg;base64,";
    result.reserve(result.size()+4*((bytes.size()+2)/3));
    for (std::size_t i=0;i<bytes.size();i+=3) {
        const unsigned bits=(unsigned(bytes[i])<<16) | (i+1<bytes.size()?unsigned(bytes[i+1])<<8:0) |
            (i+2<bytes.size()?unsigned(bytes[i+2]):0);
        result+=alphabet[(bits>>18)&63];result+=alphabet[(bits>>12)&63];
        result+=i+1<bytes.size()?alphabet[(bits>>6)&63]:'=';
        result+=i+2<bytes.size()?alphabet[bits&63]:'=';
    }
    return result;
}
std::string utf8(const fs::path& path) {
    const auto text = path.generic_u8string(); return {text.begin(), text.end()};
}
ServiceConfig prepare(ServiceConfig config) {
    if (!fs::is_directory(config.media_root) || !fs::is_directory(config.web_root) || config.detector_signature.empty())
        throw std::invalid_argument("Service requires media/web directories and a detector signature");
    config.media_root = fs::canonical(config.media_root);
    config.web_root = fs::canonical(config.web_root);
    if (bool(config.segmenter) != !config.segmentation_signature.empty())
        throw std::invalid_argument("Segmentation requires both model and signature");
    if (bool(config.ocr) != !config.ocr_signature.empty())
        throw std::invalid_argument("OCR requires both model and signature");
    if (config.live.archive.enabled) {
        const auto expected = config.media_root / "live-archive";
        if (!config.live.archive.root.empty() && fs::absolute(config.live.archive.root).lexically_normal() != expected)
            throw std::invalid_argument("Live archive root must be media_root/live-archive");
        if (fs::is_symlink(fs::symlink_status(expected)) || fs::weakly_canonical(expected) != expected)
            throw std::invalid_argument("Live archive root cannot be a symlink or directory alias");
        config.live.archive.root = expected;
    }
    if (!config.persistence.database.empty()) {
        if (config.persistence.context.empty()) throw std::invalid_argument("Persistent service requires a model/store context");
        config.persistence.context = Json::array({"local-service-v1", utf8(config.media_root),
            config.detector_signature, config.persistence.context}).dump();
        if (config.segmenter) config.persistence.context = Json::array({config.persistence.context,
            "segment-frame-v1", config.segmentation_signature}).dump();
        if (config.ocr) config.persistence.context = Json::array({config.persistence.context,
            "ocr-frame-v1",config.ocr_signature}).dump();
    }
    return config;
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
LocalService::LocalService(IDetector& detector, IEmbedder& embedder, IVectorStore& store, ServiceConfig config,
    LiveDetectorFactory live_detector, vision::LiveCaptureFactory live_capture,LiveSegmenterFactory live_segmenter)
    : detector_(detector), embedder_(embedder), store_(store), config_(prepare(std::move(config))),
      archive_owner_(config_.live.archive.enabled ? std::make_unique<vision::ArchiveOwner>(config_.live.archive.root) : nullptr),
      jobs_([this](const Json& request, const JobQueue::Progress& progress, const std::atomic_bool& cancel) {
          return execute(request, progress, cancel);
      }, config_.max_pending, config_.max_retained, config_.persistence),
      live_(config_.live, std::move(live_detector), std::move(live_capture),
        [this](const vision::ArchivedSegment& clip) { (void)archive_submit(clip.session_id,clip.index); },std::move(live_segmenter)) {
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
void LocalService::stop() { http_.stop(); live_.shutdown(); }
fs::path LocalService::media_path(const std::string& relative) const {
    const auto requested = fs::u8path(relative);
    if (relative.empty() || requested.is_absolute() || requested.has_root_name())
        throw std::invalid_argument("Use a relative video path inside media_root");
    const auto path = fs::weakly_canonical(config_.media_root / requested);
    const auto inside = path.lexically_relative(config_.media_root);
    if (inside.empty() || inside.is_absolute() || *inside.begin() == ".." ||
        !fs::is_regular_file(path) || !video_extension(path))
        throw std::invalid_argument("Video must exist inside media_root with a supported extension");
    if (*inside.begin() == "live-archive" && path.filename() != "clip.mp4")
        throw std::invalid_argument("Raw or staging archive files are not public media");
    if (*inside.begin() == "live-archive") {
        verify_archive_media(path);
    }
    return path;
}
LocalService::Json LocalService::validate(Json request) const {
    if (!request.is_object() || !request.contains("type") || !request.at("type").is_string())
        throw std::invalid_argument("Expected an object with a string type");
    const auto type = request.at("type").get<std::string>();
    const std::set<std::string> allowed = type == "index_video" ?
        std::set<std::string>{"type", "path", "stride", "max_frames"} : (type == "segment_frame" || type == "ocr_frame") ?
        std::set<std::string>{"type", "path", "frame_index"} : std::set<std::string>{"type", "query", "limit", "scope"};
    for (const auto& [key, value] : request.items()) {
        (void)value; if (!allowed.contains(key)) throw std::invalid_argument("Unknown field: " + key);
    }
    if (type == "index_video" || type == "segment_frame" || type == "ocr_frame") {
        if (type == "segment_frame" && !config_.segmenter)
            throw std::invalid_argument("Segmentation is not configured");
        if (type == "ocr_frame" && !config_.ocr) throw std::invalid_argument("OCR is not configured");
        const auto path = media_path(request.at("path").get<std::string>());
        if (*path.lexically_relative(config_.media_root).begin() == "live-archive")
            throw std::invalid_argument("Use the archive indexing endpoint for managed live clips");
        request["source_bytes"] = fs::file_size(path);
        request["source_modified"] = std::to_string(fs::last_write_time(path).time_since_epoch().count());
        if (type == "segment_frame" || type == "ocr_frame") request["frame_index"] = integer(request,"frame_index",0,0,10000);
        else {
            request["stride"] = integer(request, "stride", 15, 1, 10000);
            request["max_frames"] = integer(request, "max_frames", 0, 0, 1000000);
        }
    } else if (type == "search") {
        const auto query = request.at("query").get<std::string>();
        if (query.empty() || query.size() > 1024 || query.find_first_not_of(" \r\n\t") == std::string::npos)
            throw std::invalid_argument("query must contain 1..1024 bytes of nonblank text");
        request["limit"] = integer(request, "limit", 8, 1, 20);
        const auto scope = request.value("scope",std::string("all"));
        if (scope != "all" && scope != "live") throw std::invalid_argument("scope must be all or live");
        request["scope"] = scope;
    } else throw std::invalid_argument("type must be index_video, search, segment_frame or ocr_frame");
    return request;
}
LocalService::Json LocalService::segment_frame(const Json& request, const JobQueue::Progress& progress,
    const std::atomic_bool& cancel) {
    const bool ocr=request.at("type")=="ocr_frame";
    if (ocr ? !config_.ocr : !config_.segmenter) throw std::runtime_error("Frame analysis model is not configured");
    const auto path=media_path(request.at("path").get<std::string>());
    const auto unchanged=[&] {
        if (request.at("source_bytes") != fs::file_size(path) || request.at("source_modified") !=
            std::to_string(fs::last_write_time(path).time_since_epoch().count()))
            throw std::runtime_error("Video changed after job acceptance; submit a new job");
    };
    unchanged();
    cv::VideoCapture capture(utf8(path));
    if (!capture.isOpened()) throw std::runtime_error("Cannot open frame analysis video");
    const int index=request.at("frame_index").get<int>();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    cv::Mat frame;
    // Sequential decode avoids backend-dependent random seek/frame-number ambiguity.
    for (int i=0;i<=index;++i) {
        if (cancel) throw IndexCancelled();
        if (std::chrono::steady_clock::now()>deadline) throw std::runtime_error("Frame decode deadline exceeded");
        if (!capture.read(frame) || frame.empty()) throw std::runtime_error("Requested frame unavailable");
        if (frame.cols>1920 || frame.rows>1920) throw std::runtime_error("Analysis frame exceeds 1920 pixels per side");
        if (i%100==0) progress({{"decoded_frames",i+1},{"stage","decoding"}});
    }
    if (cancel) throw IndexCancelled();
    if (ocr) {
        progress({{"decoded_frames",index+1},{"stage","reading_text"}});
        std::vector<vision::TextRegion> regions;
        try { regions=config_.ocr->read(frame,&cancel); }
        catch (...) { if(cancel) throw IndexCancelled();throw; }
        if(cancel) throw IndexCancelled();
        if(regions.size()>64) throw std::runtime_error("OCR result exceeds 64 regions");
        Json texts=Json::array();auto painted=frame.clone();
        for(const auto& r:regions) {
            if(cancel) throw IndexCancelled();
            if(r.recognition.text.size()>256 || r.recognition.text.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyz")!=std::string::npos ||
                !std::isfinite(r.detection_confidence) || r.detection_confidence<0 || r.detection_confidence>1 ||
                !std::isfinite(r.recognition.confidence) || r.recognition.confidence<0 || r.recognition.confidence>1)
                throw std::runtime_error("Invalid OCR region text or confidence");
            // Validates finite, in-bounds, nondegenerate convex geometry before drawing/serialization.
            (void)vision::rectify_text(frame,r.polygon);
            Json polygon=Json::array();std::vector<cv::Point> points;
            for(const auto& p:r.polygon) {polygon.push_back({p.x,p.y});points.emplace_back(p);}
            texts.push_back({{"text",r.recognition.text},{"polygon",std::move(polygon)},
                {"detection_confidence",r.detection_confidence},{"recognition_confidence",r.recognition.confidence}});
            cv::polylines(painted,points,true,{0,255,0},2);
            cv::putText(painted,r.recognition.text,{points[1].x,std::max(16,points[1].y-6)},cv::FONT_HERSHEY_SIMPLEX,.55,{0,0,255},2);
        }
        unchanged();
        Json result{{"path",request.at("path")},{"frame_index",index},{"width",frame.cols},{"height",frame.rows},
            {"model_signature",config_.ocr_signature},{"charset","0123456789abcdefghijklmnopqrstuvwxyz"},
            {"polygon_order","BL,TL,TR,BR"},{"recognition_confidence_kind","mean_emitted_softmax_uncalibrated"},
            {"regions",std::move(texts)},{"preview_data_url",jpeg_url(painted)}};
        if(cancel) throw IndexCancelled();
        if(result.dump().size()>1024*1024) throw std::runtime_error("OCR result exceeds 1 MiB");
        return result;
    }
    progress({{"decoded_frames",index+1},{"stage","segmenting"}});
    vision::SegmentationPipeline pipeline(*config_.segmenter); // One frame; no tracking IDs.
    const auto instances=pipeline.analyze(frame);
    if (cancel) throw IndexCancelled();
    Json masks=Json::array();std::size_t runs=0;
    for (const auto& instance:instances) {
        if (cancel) throw IndexCancelled();
        auto counts=vision::mask_rle(instance,frame.size());runs+=counts.size();
        if (runs>100000) throw std::runtime_error("Mask RLE result exceeds 100000 runs; no partial result returned");
        const auto& box=instance.detection.bbox;
        masks.push_back({{"label",instance.detection.label},{"score",instance.detection.score},
            {"bbox",{box.x1,box.y1,box.x2,box.y2}}, {"mask_pixels",cv::countNonZero(instance.mask)},
            {"segmentation",{{"size",{frame.rows,frame.cols}},{"counts",std::move(counts)}}}});
    }
    unchanged();
    const auto preview=jpeg_url(vision::paint_masks(frame,instances));
    if (cancel) throw IndexCancelled();
    Json result{{"path",request.at("path")},{"frame_index",index},{"width",frame.cols},{"height",frame.rows},
        {"model_signature",config_.segmentation_signature},{"tracking",false},
        {"mask_format","COCO uncompressed column-major RLE"},{"instances",std::move(masks)},
        {"preview_data_url",preview}};
    if (result.dump().size()>1024*1024) throw std::runtime_error("Segmentation result exceeds 1 MiB");
    return result;
}
LocalService::Json LocalService::summary(const IndexSummary& value) const {
    return {{"source_id", value.source_id}, {"decoded_frames", value.decoded_frames},
        {"sampled_frames", value.sampled_frames}, {"indexed_items", value.indexed_items},
        {"source_fps", value.source_fps}, {"used_fallback_fps", value.used_fallback_fps}, {"stop_reason", value.stop_reason}};
}
LocalService::Json LocalService::execute(const Json& request, const JobQueue::Progress& progress, const std::atomic_bool& cancel) {
    if (cancel) throw IndexCancelled();
    if (request.at("type") == "segment_frame" || request.at("type") == "ocr_frame") return segment_frame(request,progress,cancel);
    if (request.at("type") == "index_live_archive") return index_archive(request,progress,cancel);
    if (request.at("type") == "index_video") {
        const auto path = media_path(request.at("path").get<std::string>());
        if (request.at("source_bytes") != fs::file_size(path) || request.at("source_modified") !=
            std::to_string(fs::last_write_time(path).time_since_epoch().count()))
            throw std::runtime_error("Video changed after job acceptance; submit a new job");
        VideoIndexConfig options;
        options.detector_signature = config_.detector_signature;
        options.frame_stride = request.at("stride").get<int>();
        options.max_frames = request.at("max_frames").get<int>();
        const auto result = index_video(path, detector_, embedder_, store_, options,
            [&](const IndexSummary& value) { progress(summary(value)); }, [&] { return cancel.load(); });
        progress(summary(result));
        return summary(result);
    }
    auto vector = embedder_.embed_text(request.at("query").get<std::string>());
    if (cancel) throw IndexCancelled();
    auto results = Json::array();
    const auto filter = request.value("scope",std::string("all")) == "live" ?
        std::map<std::string,std::string>{{"origin","live_archive"}} : std::map<std::string,std::string>{};
    for (const auto& match : store_.search_filtered(vector, request.at("limit").get<std::size_t>(), filter)) {
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
    return {{"query", request.at("query")}, {"scope",request.value("scope",std::string("all"))}, {"results", results}};
}
bool LocalService::accept(const httplib::Request& request, httplib::Response& response) const {
        const auto host = request.get_header_value("Host");
        const auto local = "127.0.0.1:" + std::to_string(port_);
        const auto localhost = "localhost:" + std::to_string(port_);
        const auto origin = request.get_header_value("Origin");
        if ((host != local && host != localhost) || (!origin.empty() && origin != "http://" + host)) {
            reply(response, {{"error", "Local same-origin requests only"}}, 403);
            return false;
        }
        if (request.method == "POST" && request.get_header_value("Content-Type").find("application/json") != 0) {
            reply(response, {{"error", "Content-Type must be application/json"}}, 415);
            return false;
        }
        return true;
}
void LocalService::routes() {
    http_.new_task_queue = [] { return new httplib::ThreadPool(4, 16); };
    http_.set_payload_max_length(8192);
    http_.set_read_timeout(5); http_.set_write_timeout(15);
    http_.set_default_headers({{"X-Content-Type-Options", "nosniff"}, {"Cache-Control", "no-store"},
        {"Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self'; frame-ancestors 'none'"}});
    http_.set_pre_routing_handler([this](const auto& request, auto& response) {
        // Reject POST inside its route after the bounded body has been consumed.
        // Early rejection leaves unread TCP data and can reset responses on Windows.
        if (request.method != "POST" && !accept(request, response)) return httplib::Server::HandlerResponse::Handled;
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http_.set_exception_handler([](const auto&, auto& response, std::exception_ptr error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& problem) { reply(response, {{"error", problem.what()}}, 500); }
    });
    http_.Get("/api/health", [this](const auto&, auto& response) {
        const auto error = jobs_.storage_error();
        reply(response, {{"status", error.empty() ? "ready" : "storage_error"}, {"worker_count", 1}, {"queue_capacity", config_.max_pending},
            {"history_capacity", config_.max_retained}, {"history_persistent", !config_.persistence.database.empty()},
            {"segmentation_enabled", config_.segmenter != nullptr}, {"ocr_enabled",config_.ocr != nullptr}, {"storage_error", error}}, error.empty() ? 200 : 503);
    });
    http_.Get("/api/live/sources", [this](const auto&, auto& response) { reply(response, live_.sources()); });
    http_.Get("/api/live", [this](const auto&, auto& response) { reply(response, {{"session", live_.current()}}); });
    http_.Post("/api/live/start", [this](const httplib::Request& request, httplib::Response& response) {
        if (!accept(request, response)) return;
        try {
            const auto body = Json::parse(request.body);
            if (!body.is_object() || !body.contains("source_id") || !body.at("source_id").is_string() ||
                (body.contains("archive") && !body.at("archive").is_boolean()) ||
                body.size() != (body.contains("archive") ? 2 : 1))
                throw std::invalid_argument("Expected configured source_id and optional boolean archive; URLs are not accepted");
            reply(response, live_.start(body.at("source_id").get<std::string>(),body.value("archive",false)), 202);
        } catch (const LiveSessionBusy& e) { reply(response, {{"error", e.what()}}, 409);
        } catch (const std::exception& e) { reply(response, {{"error", e.what()}}, 400); }
    });
    http_.Get("/api/live/archive", [this](const auto&, auto& response) { reply(response,archive_list()); });
    http_.Post("/api/live/archive/index", [this](const httplib::Request& request, httplib::Response& response) {
        if (!accept(request,response)) return;
        try {
            const auto body = Json::parse(request.body);
            if (!body.is_object() || body.size()!=2 || !body.contains("session_id") || !body.at("session_id").is_string() ||
                !body.contains("segment_index")) throw std::invalid_argument("Expected archive session_id and segment_index");
            const auto id = archive_submit(body.at("session_id").get<std::string>(),integer(body,"segment_index",0,1,8));
            reply(response,jobs_.get(id),202);
        } catch (const ArchiveJobBusy& e) { reply(response,{{"error",e.what()}},409);
        } catch (const JobQueueFull& e) { reply(response,{{"error",e.what()}},429);
        } catch (const JobStorageError& e) { reply(response,{{"error",e.what()}},503);
        } catch (const std::exception&) { reply(response,{{"error","Invalid or unavailable sealed archive segment"}},400); }
    });
    http_.Post(R"(/api/live/(live-[0-9]+-[0-9]+)/stop)", [this](const httplib::Request& request, httplib::Response& response) {
        if (!accept(request, response)) return;
        try {
            const auto body = Json::parse(request.body);
            if (!body.is_object() || !body.empty()) throw std::invalid_argument("Expected empty JSON object");
            if (!live_.request_stop(request.matches[1])) reply(response, {{"error", "Live session not found"}}, 404);
            else reply(response, live_.current(), 202);
        } catch (const std::exception& e) { reply(response, {{"error", e.what()}}, 400); }
    });
    http_.Get(R"(/api/live/(live-[0-9]+-[0-9]+)/preview.jpg)", [this](const httplib::Request& request, httplib::Response& response) {
        const std::string id = request.matches[1];
        if (!live_.contains(id)) { reply(response, {{"error", "Live session not found"}}, 404); return; }
        const auto preview = live_.preview(id);
        if (!preview.jpeg) { response.status = 204; return; }
        response.set_header("X-Live-Sequence", std::to_string(preview.sequence));
        response.set_header("X-Live-Source-Session", std::to_string(preview.source_session));
        response.set_header("X-Live-Epoch", std::to_string(preview.tracking_epoch));
        response.set_header("X-Decode-Age-Ms", std::to_string(preview.decode_age_ms));
        response.set_content(reinterpret_cast<const char*>(preview.jpeg->data()), preview.jpeg->size(), "image/jpeg");
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
        if (!accept(request, response)) return;
        try {
            const auto id = jobs_.submit(validate(Json::parse(request.body)));
            response.set_header("Location", "/api/jobs/" + id);
            reply(response, jobs_.get(id), 202);
        } catch (const JobQueueFull& error) { response.set_header("Retry-After", "2"); reply(response, {{"error", error.what()}}, 429);
        } catch (const JobStorageError& error) { reply(response, {{"error", error.what()}}, 503);
        } catch (const std::exception& error) { reply(response, {{"error", error.what()}}, 400); }
    });
    http_.Get("/api/jobs", [this](const auto&, auto& response) { reply(response, {{"jobs", jobs_.list()}}); });
    http_.Get(R"(/api/jobs/([0-9]+-[0-9]+))", [this](const auto& request, auto& response) {
        const auto job = jobs_.get(request.matches[1]);
        reply(response, job.is_null() ? Json{{"error", "Job not found or expired"}} : job, job.is_null() ? 404 : 200);
    });
    http_.Post(R"(/api/jobs/([0-9]+-[0-9]+)/cancel)", [this](const auto& request, auto& response) {
        if (!accept(request, response)) return;
        const std::string id = request.matches[1];
        try {
            if (!jobs_.cancel(id)) reply(response, {{"error", "Job not found"}}, 404);
            else reply(response, jobs_.get(id));
        } catch (const JobStorageError& error) { reply(response, {{"error", error.what()}}, 503); }
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
    http_.Get(R"(/api/preview/([0-9]+-[0-9]+)/([0-9]+)\.jpg)", [this](const httplib::Request& request, httplib::Response& response) {
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
