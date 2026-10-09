#include "aegisvision/service.hpp"
#include <opencv2/videoio.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
using namespace aegisvision;
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void until(F predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Timed out waiting for test state");
        std::this_thread::sleep_for(2ms);
    }
}
void queue_checks() {
    std::atomic_int calls{0};
    JobQueue queue([&](const Json& value, const JobQueue::Progress& progress, const std::atomic_bool& cancel) -> Json {
        ++calls;
        if (value == "block") {
            progress({{"indexed_items", 2}});
            until([&] { return cancel.load(); });
            throw IndexCancelled();
        }
        if (value == "fail") throw std::runtime_error("fixture failure");
        return {{"ok", true}};
    }, 1, 3);
    const auto active = queue.submit("block");
    until([&] { return queue.get(active).at("progress").contains("indexed_items"); });
    const auto waiting = queue.submit("unused");
    bool full = false;
    try { (void)queue.submit("overflow"); } catch (const JobQueueFull&) { full = true; }
    require(full, "Queue capacity not enforced");
    require(queue.cancel(waiting) && queue.get(waiting).at("state") == "cancelled", "Queued cancellation failed");
    require(calls == 1, "Cancelled queued job executed");
    require(queue.cancel(active), "Running cancellation missing");
    until([&] { return queue.get(active).at("state") == "cancelled"; });
    require(queue.get(active).at("progress").at("indexed_items") == 2, "Cancellation lost partial progress");
    const auto failure = queue.submit("fail");
    until([&] { return queue.get(failure).at("state") == "failed"; });
    const auto success = queue.submit("ok");
    until([&] { return queue.get(success).at("state") == "succeeded"; });
    require(queue.get(active).is_null() && queue.list().size() == 3, "History eviction failed");
    require(queue.get(success).at("result").at("ok") == true, "Worker did not recover from failure");
}
class Detector : public IDetector {
public:
    std::vector<Detection> detect(const Frame&) override { return {{{10, 10, 80, 90}, "person", .9F, {}, {}}}; }
};
class Embedder : public IEmbedder {
public:
    std::atomic_bool released{true}, started{false};
    std::vector<float> embed_image(const Frame& frame, const Detection&) override {
        require(frame.image && !frame.image->pixels.empty(), "Pixels missing"); return {1, 0};
    }
    std::vector<float> embed_text(std::string_view text) override {
        if (text == "hold") { started = true; until([&] { return released.load(); }); }
        if (text == "fail") throw std::runtime_error("fixture model error"); return {1, 0};
    }
};
class Segmenter : public vision::ISegmenter {
public:
    std::atomic_int mode{0};
    std::vector<vision::InstanceMask> segment(const cv::Mat& image) override {
        if (mode==1) throw std::runtime_error("fixture segmentation error");
        if (mode==2) {
            cv::Mat mask(image.size(),CV_8UC1);
            for(int y=0;y<mask.rows;++y) for(int x=0;x<mask.cols;++x) mask.at<unsigned char>(y,x)=y%2?255:0;
            vision::InstanceMask item{{{0,0,float(image.cols),float(image.rows)},"person",.9F,{},{}},
                {0,0,image.cols,image.rows},mask};
            return std::vector<vision::InstanceMask>(6,item);
        }
        return {{{{10,10,12,12},"person",.9F,{},{}},{10,10,2,2},cv::Mat(2,2,CV_8UC1,cv::Scalar(255))}};
    }
};
class Store : public IVectorStore {
public:
    std::map<std::string, SearchResult> records;
    void upsert(std::string id, std::vector<float>, std::map<std::string, std::string> metadata) override {
        records[id] = {id, 1, std::move(metadata)};
    }
    std::vector<SearchResult> search(const std::vector<float>&, std::size_t limit) const override {
        std::vector<SearchResult> result;
        for (const auto& [id, row] : records) { (void)id; if (result.size() == limit) break; result.push_back(row); }
        return result;
    }
};
Json response(const httplib::Result& result, int status = 200) {
    if (!result) throw std::runtime_error("HTTP request failed: " + httplib::to_string(result.error()));
    if (result->status != status) throw std::runtime_error("Unexpected HTTP status: " + std::to_string(result->status));
    return Json::parse(result->body);
}
void status(const httplib::Result& result, int expected) {
    if (!result) throw std::runtime_error("HTTP request failed: " + httplib::to_string(result.error()));
    require(result->status == expected, "Unexpected HTTP status");
}
Json completed(httplib::Client& client, const Json& request) {
    const auto id = response(client.Post("/api/jobs", request.dump(), "application/json"), 202).at("id").get<std::string>();
    Json job;
    until([&] {
        job = response(client.Get("/api/jobs/" + id));
        return job.at("state") == "succeeded" || job.at("state") == "failed";
    });
    return job;
}
std::string http_checks(const fs::path& root) {
    fs::create_directories(root / "media");
    const auto path = root / "media/test.avi";
    cv::VideoWriter writer(path.string(), cv::CAP_OPENCV_MJPEG, cv::VideoWriter::fourcc('M','J','P','G'), 10, {160,120});
    require(writer.isOpened(), "Video fixture unavailable");
    for (int n = 0; n < 10; ++n) writer.write(cv::Mat(120,160,CV_8UC3,cv::Scalar(10,20,30)));
    writer.release();
    Detector detector; Embedder embedder; Store store; Segmenter segmenter;
    ServiceConfig config{root / "media", fs::path(__FILE__).parent_path().parent_path() / "web", "fixture-detector"};
    config.persistence = {root / "jobs.sqlite", "http-fixture"};
    config.segmenter=&segmenter;config.segmentation_signature="fixture-seg-v1";
    LocalService service(detector, embedder, store, config);
    const int port = service.bind(0);
    std::thread server([&] { service.listen(); });
    struct Guard { LocalService& service; std::thread& server; ~Guard() { service.stop(); server.join(); } } guard{service,server};
    httplib::Client client("127.0.0.1", port); client.set_connection_timeout(2); client.set_read_timeout(10);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    require(response(client.Get("/api/health")).at("history_persistent") == true, "HTTP persistence disabled");
    require(response(client.Get("/api/health")).at("segmentation_enabled") == true,"Segmentation capability missing");
    require(response(client.Get("/api/media")).at("videos").size() == 1, "Media listing failed");
    const auto dashboard = client.Get("/");
    status(dashboard, 200);
    require(dashboard->body.find("AegisVision") != std::string::npos, "Dashboard missing");
    status(client.Get("/api/health", {{"Host", "evil.example"}}), 403);
    status(client.Post("/api/jobs", {{"Origin", "https://evil.example"}}, "{}", "application/json"), 403);
    status(client.Post("/api/jobs", "{}", "text/plain"), 415);
    (void)response(client.Post("/api/jobs", "{", "application/json"), 400);
    (void)response(client.Post("/api/jobs", R"({"type":"search","query":"x","limit":1.5})", "application/json"), 400);
    (void)response(client.Post("/api/jobs", R"({"type":"index_video","path":"../outside.avi"})", "application/json"), 400);
    (void)response(client.Post("/api/jobs", R"({"type":"index_video","path":"test.avi","stride":0})", "application/json"), 400);
    (void)response(client.Get("/api/jobs/1-999"), 404);
    for (const auto& invalid : std::vector<Json>{
        {{"type","segment_frame"},{"path","../outside.avi"}},
        {{"type","segment_frame"},{"path","test.avi"},{"frame_index",-1}},
        {{"type","segment_frame"},{"path","test.avi"},{"frame_index",10001}},
        {{"type","segment_frame"},{"path","test.avi"},{"frame_index",.5}},
        {{"type","segment_frame"},{"path","test.avi"},{"track",true}}})
        (void)response(client.Post("/api/jobs",invalid.dump(),"application/json"),400);
    const Json segmentation{{"type","segment_frame"},{"path","test.avi"},{"frame_index",2}};
    const auto segmented=completed(client,segmentation);
    require(segmented.at("state")=="succeeded","Queued segmentation failed");
    const auto& masks=segmented.at("result").at("instances");
    require(masks.size()==1 && masks[0].at("mask_pixels")==4 &&
        masks[0].at("segmentation").at("counts")==Json::array({1210,2,118,2,17868}),"HTTP mask RLE differs");
    require(segmented.at("result").at("frame_index")==2 && segmented.at("result").at("tracking")==false,
        "Single-frame semantics differ");
    require(completed(client,{{"type","segment_frame"},{"path","test.avi"},{"frame_index",10}}).at("state")=="failed",
        "Out-of-video frame accepted");
    segmenter.mode=1;
    require(completed(client,segmentation).at("state")=="failed","Segmentation model failure hidden");
    segmenter.mode=2;
    require(completed(client,segmentation).at("error").get<std::string>().find("100000 runs")!=std::string::npos,
        "Oversized mask result accepted");
    segmenter.mode=0;
    require(completed(client,segmentation).at("state")=="succeeded","Worker did not recover from segmentation failure");
    { std::ofstream outside(root / "outside.avi"); outside << "outside"; }
    std::error_code error;
    fs::create_symlink(root / "outside.avi", root / "media/escape.avi", error);
    if (!error) (void)response(client.Post("/api/jobs", R"({"type":"index_video","path":"escape.avi"})", "application/json"), 400);

    const Json video{{"type", "index_video"}, {"path", "test.avi"}, {"stride", 3}};
    const auto first = completed(client, video);
    require(first.at("state") == "succeeded" && first.at("result").at("indexed_items") == 4, "Async video indexing failed");
    const auto retry = completed(client, video);
    require(retry.at("state") == "succeeded", "Retry failed");
    const auto found = completed(client, {{"type", "search"}, {"query", "a person"}, {"limit", 8}});
    require(found.at("result").at("results").size() == 4, "Video retry duplicated records");
    const auto preview = client.Get("/api/preview/" + found.at("id").get<std::string>() + "/0.jpg");
    require(preview && preview->status == 200 && !preview->body.empty() && preview->get_header_value("Content-Type") == "image/jpeg" &&
        static_cast<unsigned char>(preview->body[0]) == 0xff, "Result preview missing");
    const auto range = client.Get("/media/test.avi", {{"Range", "bytes=0-99"}});
    require(range && range->status == 206 && range->body.size() == 100 &&
        range->get_header_value("Content-Range").starts_with("bytes 0-99/"), "Video range requests failed");
    const auto invalid_range = client.Get("/media/test.avi", {{"Range", "bytes=999999999-"}});
    require(invalid_range && invalid_range->status == 416, "Invalid video range accepted");
    require(completed(client, {{"type", "search"}, {"query", "fail"}}).at("state") == "failed", "Model error not reported");
    embedder.released = false;
    struct Release { Embedder& embedder; ~Release() { embedder.released = true; } } release{embedder};
    (void)response(client.Post("/api/jobs", R"({"type":"search","query":"hold"})", "application/json"), 202);
    until([&] { return embedder.started.load(); });
    const auto cancelled_mask = response(client.Post("/api/jobs",segmentation.dump(),"application/json"),202).at("id").get<std::string>();
    (void)response(client.Post("/api/jobs/"+cancelled_mask+"/cancel","{}","application/json"));
    require(response(client.Get("/api/jobs/"+cancelled_mask)).at("state")=="cancelled","Queued mask cancellation failed");
    const auto changed = response(client.Post("/api/jobs", video.dump(), "application/json"), 202).at("id").get<std::string>();
    const auto changed_mask = response(client.Post("/api/jobs",segmentation.dump(),"application/json"),202).at("id").get<std::string>();
    fs::last_write_time(path, fs::last_write_time(path) + std::chrono::seconds(1));
    embedder.released = true;
    Json changed_job;
    until([&] { changed_job = response(client.Get("/api/jobs/" + changed)); return changed_job.at("state") == "failed"; });
    require(changed_job.at("error").get<std::string>().find("Video changed") != std::string::npos, "Changed queued input silently replayed");
    until([&] { return response(client.Get("/api/jobs/"+changed_mask)).at("state")=="failed"; });
    require(response(client.Get("/api/jobs/"+changed_mask)).at("error").get<std::string>().find("Video changed")!=std::string::npos,
        "Segmentation read changed queued input");
    return found.at("id").get<std::string>();
}
void http_restart(const fs::path& root, const std::string& search_id) {
    Detector detector; Embedder embedder; Store store; Segmenter segmenter;
    ServiceConfig config{root / "media", fs::path(__FILE__).parent_path().parent_path() / "web", "fixture-detector"};
    config.persistence = {root / "jobs.sqlite", "http-fixture"};
    config.segmenter=&segmenter;config.segmentation_signature="fixture-seg-v1";
    LocalService service(detector, embedder, store, config);
    const int port = service.bind(0);
    std::thread server([&] { service.listen(); });
    struct Guard { LocalService& service; std::thread& server; ~Guard() { service.stop(); server.join(); } } guard{service,server};
    httplib::Client client("127.0.0.1", port);
    until([&] { const auto result = client.Get("/api/health"); return result && result->status == 200; });
    const auto restored = response(client.Get("/api/jobs/" + search_id));
    require(restored.at("state") == "succeeded" && restored.at("result").at("results").size() == 4 &&
        restored.at("attempts") == 1, "HTTP completed search lost or rerun after restart");
    status(client.Get("/api/preview/" + search_id + "/0.jpg"), 200);
    bool restored_mask=false;
    const auto history=response(client.Get("/api/jobs"));
    for (const auto& job:history.at("jobs")) {
        if (job.at("request").at("type")=="segment_frame" && job.at("state")=="succeeded") {
            require(job.at("attempts")==1 && job.at("result").at("instances")[0].at("mask_pixels")==4,
                "Completed mask lost or rerun after restart");
            restored_mask=true;
        }
    }
    require(restored_mask,"Persisted mask job missing");
}
void segmentation_disabled(const fs::path& root) {
    Detector detector; Embedder embedder; Store store;
    ServiceConfig config{root/"media",fs::path(__FILE__).parent_path().parent_path()/"web","fixture-detector"};
    LocalService service(detector,embedder,store,config);
    const int port=service.bind(0);std::thread server([&]{service.listen();});
    struct Guard {LocalService& s;std::thread& t;~Guard(){s.stop();t.join();}} guard{service,server};
    httplib::Client client("127.0.0.1",port);
    until([&]{const auto result=client.Get("/api/health");return result && result->status==200;});
    require(response(client.Get("/api/health")).at("segmentation_enabled")==false,"Disabled segmentation advertised");
    (void)response(client.Post("/api/jobs",R"({"type":"segment_frame","path":"test.avi"})","application/json"),400);
}
}
int main() {
    const auto root = fs::temp_directory_path() / ("aegis-service-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        queue_checks(); const auto search_id = http_checks(root); http_restart(root, search_id);segmentation_disabled(root);
        fs::remove_all(root); // This test's unique fixture directory only.
        std::cout << "Async jobs, cancellation, backpressure, history, path confinement, HTTP and video previews passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << " fixtures=" << root << '\n'; return 1; }
}
