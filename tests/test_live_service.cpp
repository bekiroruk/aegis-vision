#include "aegisvision/service.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

namespace {
using namespace aegisvision;
using Json = nlohmann::json;
using namespace std::chrono_literals;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void until(F predicate) {
    const auto end = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= end) throw std::runtime_error("Live test state timed out");
        std::this_thread::sleep_for(5ms);
    }
}
struct Control {
    std::atomic_bool connected{true}, produce{true}, fail_model{false}, wide{false};
    std::atomic_int model_instances{0}, analyses{0};
};
class Capture : public vision::ILiveCapture {
public:
    explicit Capture(std::shared_ptr<Control> state) : state_(std::move(state)) {}
    bool open(const std::string&, const vision::LiveConfig& config) override { timeout_ = config.read_timeout_ms; return state_->connected; }
    bool read(cv::Mat& image) override {
        std::this_thread::sleep_for(10ms);
        if (!state_->connected) return false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_);
        while (!state_->produce && state_->connected && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(2ms);
        if (!state_->connected || !state_->produce) return false;
        image = cv::Mat(120,state_->wide ? 240 : 160,CV_8UC3,cv::Scalar(30,40,50)); return true;
    }
    void close() override {}
private: std::shared_ptr<Control> state_; int timeout_{2000};
};
class LiveDetector : public IDetector {
public:
    explicit LiveDetector(std::shared_ptr<Control> state) : state_(std::move(state)) { ++state_->model_instances; }
    std::vector<Detection> detect(const Frame& frame) override {
        require(frame.image && frame.source_id.starts_with("rtsp-session-"), "Real live pixels missing");
        if (state_->fail_model) throw std::runtime_error("secret model exception: rtsp://private-camera");
        ++state_->analyses; std::this_thread::sleep_for(20ms);
        return {{{10,10,80,90}, "person", .9F, {}, {}}};
    }
private: std::shared_ptr<Control> state_;
};
class FileDetector : public IDetector {
public: std::vector<Detection> detect(const Frame&) override { throw std::runtime_error("Live used file detector"); }
};
class LiveSegmenter : public vision::ISegmenter {
public:
    explicit LiveSegmenter(std::shared_ptr<Control> state):state_(std::move(state)) {++state_->model_instances;}
    std::vector<vision::InstanceMask> segment(const cv::Mat& image) override {
        require((image.cols==160 || image.cols==240) && image.rows==120,"Live segmentation pixels missing");
        if (state_->fail_model) throw std::runtime_error("secret segmentation failure");
        ++state_->analyses;std::this_thread::sleep_for(20ms);
        return {{{{10,10,80,90},"person",.9F,{},{}},{10,10,70,80},cv::Mat(80,70,CV_8UC1,cv::Scalar(255))}};
    }
private:std::shared_ptr<Control> state_;
};
class Embedder : public IEmbedder {
public:
    std::vector<float> embed_image(const Frame&, const Detection&) override { return {1,0}; }
    std::vector<float> embed_text(std::string_view) override { return {1,0}; }
};
class Store : public IVectorStore {
public:
    void upsert(std::string, std::vector<float>, std::map<std::string,std::string>) override {}
    std::vector<SearchResult> search(const std::vector<float>&, std::size_t) const override { return {}; }
};
Json json(const httplib::Result& response, int code = 200) {
    require(response && response->status == code, "Unexpected HTTP status"); return Json::parse(response->body);
}
void code(const httplib::Result& result, int expected) { require(result && result->status == expected, "Unexpected HTTP status"); }
void http_checks(const std::filesystem::path& root,bool masks=false) {
    const auto control = std::make_shared<Control>();
    FileDetector detector; Embedder embedder; Store store;
    ServiceConfig config{root, std::filesystem::path(__FILE__).parent_path().parent_path() / "web", "fixture"};
    config.live.sources = {{"test", "Test source", "rtsp://127.0.0.1/preset-only"}};
    config.live.stream.duration_seconds = 20; config.live.stream.max_outage_ms = 5000;
    config.live.stream.read_timeout_ms = 5000;
    config.live.stream.reconnect_initial_ms = 10; config.live.stream.reconnect_max_ms = 20;
    LocalService service(detector, embedder, store, config,
        [control] { return std::make_unique<LiveDetector>(control); },
        [control] { return std::make_unique<Capture>(control); },
        masks ? LiveSegmenterFactory([control]{return std::make_unique<LiveSegmenter>(control);}) : LiveSegmenterFactory{});
    const auto port = service.bind(0);
    std::thread server([&] { service.listen(); });
    struct Guard { LocalService& s; std::thread& thread; std::shared_ptr<Control> state;
        ~Guard() { state->produce = true; state->connected = true; s.stop(); thread.join(); } } guard{service,server,control};
    httplib::Client client("127.0.0.1",port); client.set_read_timeout(10);
    until([&] { auto r = client.Get("/api/health"); return r && r->status == 200; });
    const auto presets = json(client.Get("/api/live/sources"));
    require(presets.at("analysis_mode")== (masks ? "segmentation" : "detection"),"Wrong advertised live mode");
    require(presets.at("sources").size() == 1 && presets.dump().find("rtsp://") == std::string::npos,
        "Source URL leaked through API");
    require(json(client.Get("/api/live")).at("session").is_null(), "Session started without user action");
    code(client.Post("/api/live/start", {{"Origin","https://evil.example"}}, R"({"source_id":"test"})", "application/json"),403);
    code(client.Post("/api/live/start", R"({"source_id":"test"})", "text/plain"),415);
    (void)json(client.Post("/api/live/start", R"({"source_id":"test","url":"rtsp://evil"})", "application/json"),400);
    (void)json(client.Post("/api/live/start", R"({"source_id":"unknown"})", "application/json"),400);
    (void)json(client.Post("/api/live/start", "[1]", "application/json"),400);
    const auto accepted = json(client.Post("/api/live/start", R"({"source_id":"test"})", "application/json"),202);
    const std::string id = accepted.at("id");
    (void)json(client.Post("/api/live/start", R"({"source_id":"test"})", "application/json"),409);
    const auto current = [&] { return json(client.Get("/api/live")).at("session"); };
    until([&] { return current().at("has_preview") == true; });
    const auto picture = client.Get("/api/live/" + id + "/preview.jpg"); code(picture,200);
    require(picture->get_header_value("Content-Type") == "image/jpeg" && !picture->get_header_value("X-Live-Sequence").empty(),
        "Preview headers missing");
    const std::vector<unsigned char> bytes(picture->body.begin(), picture->body.end());
    require(!cv::imdecode(bytes,cv::IMREAD_COLOR).empty(), "Live JPEG not decodable");
    if (masks) {
        const auto pixel=cv::imdecode(bytes,cv::IMREAD_COLOR).at<cv::Vec3b>(40,40);
        require(pixel[0]>45 && pixel[1]>65,"Preview did not paint mask interior");
        require(current().at("analysis_mode")=="segmentation","Session mode missing");
    }
    require(control->model_instances == 1 && control->analyses > 0, "Isolated model not used");
    // Regular search still executes while live detector/capture run on their own worker.
    const auto search = json(client.Post("/api/jobs", R"({"type":"search","query":"a person"})", "application/json"),202);
    until([&] { return json(client.Get("/api/jobs/" + search.at("id").get<std::string>())).at("state") == "succeeded"; });
    const auto sessions_before = current().at("sessions").get<int>();
    control->connected = false;
    until([&] { return current().at("connection_state") == "reconnecting"; });
    require(current().at("has_preview") == false, "Disconnected image remained live");
    code(client.Get("/api/live/" + id + "/preview.jpg"),204);
    control->connected = true;
    until([&] { const auto s = current(); return s.at("sessions").get<int>() > sessions_before && s.at("has_preview") == true; });
    require(current().at("tracking_epochs").get<int>() >= 2, "Tracker not reset after reconnect");
    const int epoch_before_resize=current().at("tracking_epochs").get<int>();
    control->wide=true;
    until([&]{return current().at("has_preview")==true && current().at("tracking_epochs").get<int>()>epoch_before_resize;});
    control->produce = false;
    const auto previous_sequence = current().at("preview_sequence").get<std::uint64_t>();
    until([&] { return current().at("has_preview") == false; });
    code(client.Get("/api/live/" + id + "/preview.jpg"),204);
    control->produce = true;
    until([&] { const auto s = current(); return s.at("has_preview") == true && s.at("preview_sequence").get<std::uint64_t>() > previous_sequence; });
    code(client.Post("/api/live/" + id + "/stop", "{}", "application/json"),202);
    code(client.Get("/api/live/" + id + "/preview.jpg"),204);
    until([&] { return current().at("state") == "stopped"; });
    require(current().at("queue_high_watermark").get<int>() <= 1, "Live queue exceeded capacity");
    code(client.Post("/api/live/" + id + "/stop", "{}", "application/json"),202);
    control->fail_model = true;
    const auto second = json(client.Post("/api/live/start", R"({"source_id":"test"})", "application/json"),202);
    require(second.at("id") != id, "Live session ID reused");
    code(client.Post("/api/live/" + id + "/stop", "{}", "application/json"),404);
    code(client.Get("/api/live/" + id + "/preview.jpg"),404);
    until([&] { return current().at("state") == "failed"; });
    require(current().at("error").get<std::string>().find("secret") == std::string::npos, "Worker exception leaked to HTTP");
    control->fail_model = false;
    const auto third = json(client.Post("/api/live/start", R"({"source_id":"test"})", "application/json"),202);
    until([&] { return current().at("has_preview") == true; });
    require(control->model_instances == 3, "Completed/failed model reused");
    code(client.Post("/api/live/" + third.at("id").get<std::string>() + "/stop", "{}", "application/json"),202);
    until([&] { return current().at("active") == false; });
}
void lifecycle_checks() {
    auto control = std::make_shared<Control>();
    LiveServiceConfig config;
    config.sources = {{"test", "Test", "rtsp://127.0.0.1/test"}}; config.stream.duration_seconds = 1;
    LiveSessions sessions(config, [control] { return std::make_unique<LiveDetector>(control); },
        [control] { return std::make_unique<Capture>(control); });
    (void)sessions.start("test");
    until([&] { return sessions.current().at("state") == "completed"; });
    require(sessions.preview(sessions.current().at("id")).jpeg == nullptr, "Completed session still serves live image");
    (void)sessions.start("test");
    until([&] { return sessions.current().at("has_preview") == true; });
    const auto before = std::chrono::steady_clock::now(); sessions.shutdown();
    require(std::chrono::steady_clock::now() - before < 2s && sessions.current().at("state") == "stopped", "Shutdown did not join live worker");
}
}
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("aegis-live-service-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        http_checks(root); http_checks(root,true); lifecycle_checks();
        std::filesystem::remove_all(root); // Only this test's newly created scratch directory.
        std::cout << "Live HTTP presets, isolation, preview, reconnect, stop, failure and lifecycle passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << " fixtures=" << root << '\n'; return 1; }
}
