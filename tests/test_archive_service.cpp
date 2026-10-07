#include "aegisvision/service.hpp"
#include "aegisvision/vector_store.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <source_location>

namespace {
using namespace aegisvision;
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void until(F predicate, const std::source_location location = std::source_location::current()) {
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Archive HTTP test timed out at " + std::string(location.file_name()) +
                ":" + std::to_string(location.line()));
        std::this_thread::sleep_for(10ms);
    }
}
Json json(const httplib::Result& result, int expected = 200) {
    if (!result) throw std::runtime_error("HTTP request failed: " + httplib::to_string(result.error()));
    if (result->status != expected) throw std::runtime_error("Unexpected HTTP status: " +
        std::to_string(result->status) + ", expected " + std::to_string(expected) + ", body=" + result->body);
    return Json::parse(result->body);
}
void status(const httplib::Result& result, int expected) {
    require(result && result->status == expected, "Unexpected media HTTP status");
}
void jpeg(const httplib::Result& result) {
    status(result,200);
    const std::vector<unsigned char> bytes(result->body.begin(),result->body.end());
    require(result->get_header_value("Content-Type") == "image/jpeg" &&
        !cv::imdecode(bytes,cv::IMREAD_COLOR).empty(), "JPEG is missing or undecodable");
}
struct Control { std::atomic_bool image_released{true}, image_started{false}; };
class Capture final : public vision::ILiveCapture {
public:
    bool open(const std::string&, const vision::LiveConfig&) override { return true; }
    bool read(cv::Mat& image) override {
        std::this_thread::sleep_for(10ms);
        image = cv::Mat(120,160,CV_8UC3,cv::Scalar(25,80,135));
        return true;
    }
    void close() override {}
};
class Detector final : public IDetector {
public:
    std::vector<Detection> detect(const Frame& frame) override {
        require(frame.image && !frame.image->pixels.empty(), "Detector did not receive pixels");
        return {{{10,10,80,90},"person",.9F,{},{}}};
    }
};
class Embedder final : public IEmbedder {
public:
    explicit Embedder(std::shared_ptr<Control> control) : control_(std::move(control)) {}
    std::vector<float> embed_image(const Frame& frame, const Detection&) override {
        require(frame.image && !frame.image->pixels.empty(), "Embedder did not receive archive pixels");
        control_->image_started = true;
        until([&] { return control_->image_released.load(); });
        return {1,0};
    }
    std::vector<float> embed_text(std::string_view) override { return {1,0}; }
private:
    std::shared_ptr<Control> control_;
};
struct Running {
    LocalService& service;
    std::shared_ptr<Control> control;
    std::thread thread;
    int port;
    Running(LocalService& value, std::shared_ptr<Control> state)
        : service(value), control(std::move(state)), port(service.bind(0)) {
        thread = std::thread([this] { service.listen(); });
    }
    ~Running() {
        control->image_released = true; // Release blocked inference before worker joins.
        service.stop(); if (thread.joinable()) thread.join();
    }
};
ServiceConfig configuration(const fs::path& root, bool enabled = true) {
    ServiceConfig result{root / "media", fs::path(__FILE__).parent_path().parent_path() / "web", "archive-fixture-detector"};
    result.persistence = {root / "jobs.sqlite","archive-http-fixture-v1"};
    result.live.sources = {{"test","Archive fixture","rtsp://127.0.0.1/preset-only"}};
    result.live.stream.duration_seconds = 2;
    result.live.stream.read_timeout_ms = 100;
    result.live.stream.open_timeout_ms = 100;
    result.live.archive.enabled = enabled;
    result.live.archive.max_frames = 3;
    result.live.archive.max_segments_per_session = 2;
    result.live.archive.max_total_segments = 8;
    result.archive_encoder.timeout = 10s;
    return result;
}
Json completed(httplib::Client& client, const Json& request) {
    const auto id = json(client.Post("/api/jobs",request.dump(),"application/json"),202).at("id").get<std::string>();
    Json job;
    until([&] {
        job = json(client.Get("/api/jobs/" + id));
        const std::string state = job.at("state");
        return state == "succeeded" || state == "failed" || state == "cancelled";
    });
    if (job.at("state") != "succeeded") throw std::runtime_error("Fixture job failed: " + job.dump());
    return job;
}
void ordinary_fixture(const fs::path& root) {
    fs::create_directories(root / "media");
    const auto path = root / "media/ordinary.avi";
    cv::VideoWriter writer(path.string(),cv::CAP_OPENCV_MJPEG,
        cv::VideoWriter::fourcc('M','J','P','G'),10,{160,120});
    require(writer.isOpened(), "Ordinary video fixture could not be opened");
    for (int i = 0; i < 2; ++i) writer.write(cv::Mat(120,160,CV_8UC3,cv::Scalar(70,40,20)));
    writer.release();
}
std::string http_checks(const fs::path& root, InMemoryVectorStore& store) {
    auto control = std::make_shared<Control>();
    Detector detector; Embedder embedder(control);
    LocalService service(detector,embedder,store,configuration(root),
        [] { return std::make_unique<Detector>(); },[] { return std::make_unique<Capture>(); });
    Running running(service,control);
    httplib::Client client("127.0.0.1",running.port); client.set_connection_timeout(2); client.set_read_timeout(10);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    const auto sources = json(client.Get("/api/live/sources"));
    require(sources.at("archive_available") == true && sources.dump().find("rtsp://") == std::string::npos,
        "Archive availability missing or private source URL leaked");
    require(json(client.Get("/api/live/archive")).at("segments").empty(), "Archive started without opt-in");
    (void)json(client.Post("/api/live/start",R"({"source_id":"test","archive":"true"})","application/json"),400);
    (void)json(client.Post("/api/live/start",R"({"source_id":"test","archive":true,"path":"evil"})","application/json"),400);
    (void)json(client.Post("/api/jobs",R"({"type":"index_live_archive","session_id":"test","segment_index":1})","application/json"),400);
    (void)json(client.Post("/api/jobs",R"({"type":"search","query":"person","scope":"unknown"})","application/json"),400);
    (void)json(client.Post("/api/live/archive/index",R"({"session_id":"../outside","segment_index":1})","application/json"),400);
    (void)json(client.Post("/api/live/archive/index",R"({"session_id":"live-1-1","segment_index":0})","application/json"),400);
    const auto start = json(client.Post("/api/live/start",R"({"source_id":"test","archive":true})","application/json"),202);
    const auto session_id = start.at("id").get<std::string>();
    until([&] { return json(client.Get("/api/live")).at("session").at("has_preview") == true; });
    jpeg(client.Get("/api/live/" + session_id + "/preview.jpg"));
    Json archive;
    until([&] {
        archive = json(client.Get("/api/live/archive"));
        const auto live = json(client.Get("/api/live")).at("session");
        const auto recording = live.at("archive");
        if (recording.at("state") == "error" ||
            (archive.at("segments").size() != 2 && !live.at("active").get<bool>()))
            throw std::runtime_error("Archive recording incomplete: " + live.dump() + " catalog=" + archive.dump());
        for (const auto& segment : archive.at("segments"))
            if (segment.at("index_state") == "failed")
                throw std::runtime_error("Archive indexing failed: " + segment.dump());
        if (archive.at("segments").size() != 2) return false;
        for (const auto& segment : archive.at("segments")) if (segment.at("index_state") != "succeeded") return false;
        return true;
    });
    require(json(client.Get("/api/live")).at("session").at("processed_frames").get<int>() > 6,
        "Archive segment quota stopped live preview analysis");
    for (const auto& segment : archive.at("segments")) {
        require(segment.at("session_id") == session_id && segment.at("frames") == 3,
            "Archive segment boundary or provenance mismatch");
        require(std::abs(segment.at("playback_duration_seconds").get<double>() - .3) < .01,
            "Archive playback duration was incorrectly treated as live arrival span");
        const auto media = segment.at("media_path").get<std::string>();
        status(client.Get("/media/" + media),200);
        const auto range = client.Get("/media/" + media,{{"Range","bytes=0-99"}});
        require(range && range->status == 206 && range->body.size() == 100,
            "Encoded archive byte range missing");
        const auto directory = fs::path(media).parent_path().generic_string();
        status(client.Get("/media/" + directory + "/raw.avi"),404);
        status(client.Get("/media/" + directory + "/clip.partial.mp4"),404);
        (void)json(client.Post("/api/jobs",Json{{"type","index_video"},{"path",media}}.dump(),"application/json"),400);
        const auto manifest = root / "media" / fs::path(media).parent_path() / "manifest.json";
        std::ifstream file(manifest);
        const auto mapping = Json::parse(file);
        require(mapping.at("frame_map").size() == 3 && mapping.at("source_session").get<int>() > 0,
            "Frame/arrival map missing from sealed segment");
    }
    (void)json(client.Post("/api/live/" + session_id + "/stop","{}","application/json"),202);
    until([&] { return json(client.Get("/api/live")).at("session").at("active") == false; });
    (void)completed(client,{{"type","index_video"},{"path","ordinary.avi"},{"stride",1}});
    auto live = completed(client,{{"type","search"},{"query","person"},{"scope","live"},{"limit",20}});
    const auto matches = live.at("result").at("results");
    require(matches.size() == 2, "Live scope lost results or included ordinary file crops");
    for (const auto& match : matches) {
        const auto metadata = match.at("metadata");
        require(metadata.at("origin") == "live_archive" && metadata.at("live_session_id") == session_id &&
            metadata.at("live_source_id") == "test" && std::stoi(metadata.at("source_session").get<std::string>()) > 0 &&
            std::stoi(metadata.at("tracking_epoch").get<std::string>()) > 0,
            "Live search provenance was not retained");
        require(std::stoll(metadata.at("timestamp_ms").get<std::string>()) == 0 && metadata.at("frame_index") == "0",
            "Search seek timestamp is not encoded CFR clip position");
        require(match.at("media_path").get<std::string>().starts_with("live-archive/"),
            "Live result does not point at verified MP4");
    }
    const auto search_id = live.at("id").get<std::string>();
    jpeg(client.Get("/api/preview/" + search_id + "/0.jpg"));
    const auto all = completed(client,{{"type","search"},{"query","person"},{"scope","all"},{"limit",20}});
    require(all.at("result").at("results").size() == 4, "All scope did not include both live and ordinary video");
    const auto first_index = archive.at("segments").at(0).at("segment_index").get<int>();
    control->image_started = false; control->image_released = false;
    const Json retry_body{{"session_id",session_id},{"segment_index",first_index}};
    const auto retry = json(client.Post("/api/live/archive/index",retry_body.dump(),"application/json"),202);
    until([&] { return control->image_started.load(); });
    (void)json(client.Post("/api/live/archive/index",retry_body.dump(),"application/json"),409);
    control->image_released = true;
    const auto retry_id = retry.at("id").get<std::string>();
    until([&] { return json(client.Get("/api/jobs/" + retry_id)).at("state") == "succeeded"; });
    const auto repeated = completed(client,{{"type","search"},{"query","person"},{"scope","live"},{"limit",20}});
    require(repeated.at("result").at("results").size() == 2, "Archive retry duplicated vector points");
    (void)json(client.Post("/api/live/archive/index",Json{{"session_id",session_id},{"segment_index",99}}.dump(),"application/json"),400);
    return search_id;
}
void restart_checks(const fs::path& root, InMemoryVectorStore& store, const std::string& search_id) {
    auto control = std::make_shared<Control>(); Detector detector; Embedder embedder(control);
    LocalService service(detector,embedder,store,configuration(root),
        [] { return std::make_unique<Detector>(); },[] { return std::make_unique<Capture>(); });
    Running running(service,control); httplib::Client client("127.0.0.1",running.port); client.set_read_timeout(10);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    require(json(client.Get("/api/live")).at("session").is_null(), "Restart unexpectedly resumed camera capture");
    const auto archive = json(client.Get("/api/live/archive"));
    require(archive.at("segments").size() == 2, "Sealed archive manifests not restored after restart");
    for (const auto& segment : archive.at("segments")) {
        require(segment.at("index_state") == "succeeded" && !segment.at("media_path").is_null(),
            "Restart lost encoded clip or completed index state");
        status(client.Get("/media/" + segment.at("media_path").get<std::string>()),200);
    }
    require(json(client.Get("/api/jobs/" + search_id)).at("state") == "succeeded", "SQLite search history lost on restart");
    jpeg(client.Get("/api/preview/" + search_id + "/0.jpg"));
}
// Mutates only generated fixture files, and restores both contents and the exact
// original timestamp even when an integrity assertion throws.
class FixtureMutation {
public:
    explicit FixtureMutation(fs::path path) : path_(std::move(path)), modified_(fs::last_write_time(path_)) {
        std::ifstream file(path_,std::ios::binary);
        require(file.is_open(),"Fixture mutation source missing");
        original_ = std::string(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
        require(!original_.empty(),"Fixture mutation source is empty");
    }
    ~FixtureMutation() {
        try { restore(); } catch (...) { /* Keep the original test exception. */ }
    }
    const std::string& original() const { return original_; }
    void write(const std::string& value) {
        std::ofstream file(path_,std::ios::binary | std::ios::trunc);
        file.write(value.data(),static_cast<std::streamsize>(value.size())); file.close();
        require(!file.fail(),"Could not write fixture integrity mutation");
        fs::last_write_time(path_,modified_);
    }
    void restore() { write(original_); }
private:
    fs::path path_;
    fs::file_time_type modified_;
    std::string original_;
};
void integrity_checks(const fs::path& root, InMemoryVectorStore& store) {
    auto control = std::make_shared<Control>(); Detector detector; Embedder embedder(control);
    LocalService service(detector,embedder,store,configuration(root),
        [] { return std::make_unique<Detector>(); },[] { return std::make_unique<Capture>(); });
    Running running(service,control); httplib::Client client("127.0.0.1",running.port); client.set_read_timeout(10);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    const auto initial = json(client.Get("/api/live/archive")).at("segments");
    require(initial.size() == 2,"Integrity fixture requires two sealed segments");
    const auto damaged = initial.at(0), healthy = initial.at(1);
    const auto relative = fs::path(damaged.at("media_path").get<std::string>()).parent_path();
    const auto directory = root / "media" / relative;
    const Json retry{{"session_id",damaged.at("session_id")},{"segment_index",damaged.at("segment_index")}};
    const auto catalog_healthy_and_damaged = [&] {
        const auto catalog = json(client.Get("/api/live/archive")).at("segments");
        require(catalog.size() == 2,"Damaged manifest broke the bounded catalog");
        bool found_healthy = false, found_damaged = false;
        for (const auto& segment : catalog) {
            if (segment.at("segment_index") == healthy.at("segment_index"))
                found_healthy = segment.at("index_state") == "succeeded" && !segment.at("media_path").is_null();
            if (segment.at("segment_index") == damaged.at("segment_index"))
                found_damaged = segment.at("index_state") == "failed" && segment.at("media_path").is_null();
        }
        require(found_healthy && found_damaged,"Catalog did not isolate damaged manifest from healthy segment");
    };
    {
        FixtureMutation manifest(directory / "manifest.json");
        auto changed = manifest.original();
        const auto original_source = Json::parse(changed).at("source_id").get<std::string>();
        std::smatch source;
        require(std::regex_search(changed,source,
            std::regex(R"source("source_id"\s*:\s*"([A-Za-z0-9_-]{1,64})")source")) &&
            source.str(1) == original_source,"Manifest source mutation fixture mismatch");
        // Change one identifier byte in place, preserving arbitrary JSON
        // whitespace/indentation and the exact file length, not a dump style.
        auto& first = changed.at(static_cast<std::size_t>(source.position(1)));
        first = first == 'x' ? 'y' : 'x';
        require(Json::parse(changed).at("source_id") != original_source,
            "Manifest mutation did not change source provenance");
        require(changed.size() == manifest.original().size(),"Manifest mutation changed byte count");
        manifest.write(changed); // Same size and mtime: stat-only validation is insufficient.
        (void)json(client.Post("/api/live/archive/index",retry.dump(),"application/json"),400);
        catalog_healthy_and_damaged();
        manifest.write("{");
        catalog_healthy_and_damaged(); // Parse failure must not turn GET archive into HTTP 500.
        manifest.restore();
    }
    {
        FixtureMutation raw(directory / "raw.avi");
        auto changed = raw.original(); changed.at(0) = static_cast<char>(changed.at(0) ^ 1);
        raw.write(changed); // Exact original size and mtime, but different sealed content.
        (void)json(client.Post("/api/live/archive/index",retry.dump(),"application/json"),400);
        raw.restore();
    }
    {
        const auto media = damaged.at("media_path").get<std::string>();
        status(client.Get("/media/" + media),200);
        FixtureMutation clip(directory / "clip.mp4");
        auto changed = clip.original();
        auto& byte = changed.at(changed.size() / 2); byte = static_cast<char>(byte ^ 1);
        clip.write(changed); // Same size and mtime cannot authenticate published bytes.
        status(client.Get("/media/" + media),404);
        clip.restore();
        status(client.Get("/media/" + media),200);
    }
    const auto restored = json(client.Get("/api/live/archive")).at("segments");
    require(restored.size() == 2,"Restoring fixture changed catalog cardinality");
    for (const auto& segment : restored)
        require(segment.at("index_state") == "succeeded" && !segment.at("media_path").is_null(),
            "Restored immutable fixture did not recover catalog status");
    // Merely having the approved filename does not make an arbitrary MP4 an
    // owned, verified archive. This new path belongs only to this scratch test.
    const auto unowned_relative = fs::path("live-archive/live-1234567890-99/segment-0001/clip.mp4");
    const auto unowned = root / "media" / unowned_relative;
    fs::create_directories(unowned.parent_path());
    { std::ofstream file(unowned,std::ios::binary); file << "not a verified archive"; }
    status(client.Get("/media/" + unowned_relative.generic_string()),404);
}
void disabled_checks(const fs::path& root) {
    auto control = std::make_shared<Control>(); Detector detector; Embedder embedder(control); InMemoryVectorStore store;
    auto config = configuration(root,false); config.persistence = {};
    LocalService service(detector,embedder,store,config,
        [] { return std::make_unique<Detector>(); },[] { return std::make_unique<Capture>(); });
    Running running(service,control); httplib::Client client("127.0.0.1",running.port);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    require(json(client.Get("/api/live/sources")).at("archive_available") == false, "Disabled archive announced as available");
    (void)json(client.Post("/api/live/start",R"({"source_id":"test","archive":true})","application/json"),400);
    const auto start = json(client.Post("/api/live/start",R"({"source_id":"test"})","application/json"),202);
    until([&] { return json(client.Get("/api/live")).at("session").at("has_preview") == true; });
    require(json(client.Get("/api/live")).at("session").at("archive").at("enabled") == false,
        "Absent archive field enabled recording");
    (void)json(client.Post("/api/live/" + start.at("id").get<std::string>() + "/stop","{}","application/json"),202);
}
} // namespace
int main() {
    const auto root = fs::temp_directory_path() / ("aegis-archive-http-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        ordinary_fixture(root); InMemoryVectorStore store;
        const auto search_id = http_checks(root,store); restart_checks(root,store,search_id);
        integrity_checks(root,store); disabled_checks(root);
        fs::remove_all(root); // Only this test's unique, newly created fixture directory.
        std::cout << "Live archive HTTP opt-in, FFmpeg MP4, provenance, filters, retry, confinement, integrity and restart passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << " fixtures=" << root << '\n'; return 1;
    }
}
