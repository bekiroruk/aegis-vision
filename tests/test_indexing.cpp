#include "aegisvision/indexing.hpp"
#include "aegisvision/vision.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F action, const char* message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
class RecordingStore : public IVectorStore {
public:
    std::map<std::string, std::map<std::string, std::string>> records;
    bool fail{false};
    void upsert(std::string id, std::vector<float> vector, std::map<std::string, std::string> metadata) override {
        if (fail) throw std::runtime_error("Simulated store outage");
        require(vector.size() == 2, "Missing embedding");
        records[id] = std::move(metadata);
    }
    std::vector<SearchResult> search(const std::vector<float>&, std::size_t) const override { return {}; }
};
class PixelEmbedder : public IEmbedder {
public:
    std::vector<float> embed_image(const Frame& frame, const Detection& detection) override {
        require(frame.image && !frame.image->pixels.empty(), "Indexing lost source pixels");
        require(detection.bbox.area() > 0 && detection.bbox.x2 <= frame.image->width &&
            detection.bbox.y2 <= frame.image->height, "Indexing lost crop coordinates");
        return {1, 0};
    }
    std::vector<float> embed_text(std::string_view) override { throw std::runtime_error("Unexpected text query"); }
};
class FrameDetector : public IDetector {
public:
    bool reverse{false}, empty{false}, invalid{false};
    std::vector<int> frames;
    std::vector<Detection> detect(const Frame& frame) override {
        const int index = std::stoi(frame.frame_id);
        frames.push_back(index);
        require(frame.timestamp_ms == index * 100, "Incorrect source timestamp");
        if (empty || index == 3) return {};
        if (invalid) return {{{0, 0, 999, 999}, "bad", 1, {}, {}}};
        std::vector<Detection> detections = {
            {{5, 5, 60, 70}, "person", 0.9F, {}, {}},
            {{70, 10, 140, 110}, "bus", 0.8F, {}, {}}};
        if (reverse) std::reverse(detections.begin(), detections.end());
        return detections;
    }
};

void directory_checks(const fs::path& root) {
    const auto images = root / fs::u8path("görseller");
    fs::create_directories(images / "nested");
    const cv::Mat pixels(32, 32, CV_8UC3, cv::Scalar(10, 20, 30));
    vision::save_image(images / "a.PNG", pixels);
    vision::save_image(images / "nested/b.png", pixels);
    { std::ofstream ignored(images / "note.txt"); ignored << "not an image"; }
    PixelEmbedder embedder;
    RecordingStore store;
    const auto flat = index_directory(images, embedder, store);
    require(flat.indexed_items == 1 && flat.skipped_entries == 1 && store.records.size() == 1,
        "Directory depth or unsupported extension handling failed");
    const auto old_id = store.records.begin()->first;
    const auto old_hash = store.records.begin()->second.at("sha256");
    DirectoryIndexConfig recursive; recursive.recursive = true;
    const auto all = index_directory(images, embedder, store, recursive);
    require(all.indexed_items == 2 && store.records.size() == 2, "Recursive retry duplicated a file");
    vision::save_image(images / "a.PNG", cv::Mat(32, 32, CV_8UC3, cv::Scalar(50, 60, 70)));
    (void)index_directory(images, embedder, store, recursive);
    require(store.records.size() == 2 && store.records.at(old_id).at("sha256") != old_hash,
        "Changed image did not update the existing record");

    RecordingStore limited_store;
    auto limited = recursive; limited.max_images = 1;
    rejects([&] { (void)index_directory(images, embedder, limited_store, limited); }, "Image limit ignored");
    require(limited_store.records.empty(), "Image limit must fail before upserts");
    fs::create_directories(root / "empty");
    rejects([&] { (void)index_directory(root / "empty", embedder, store); }, "Empty directory accepted");

    std::error_code error;
    fs::create_directory_symlink(images, images / "loop", error);
    if (!error) {
        require(index_directory(images, embedder, store, recursive).indexed_items == 2,
            "Directory symlink was followed");
        fs::remove(images / "loop");
    }
    { std::ofstream corrupt(images / "z.jpg"); corrupt << "broken jpeg"; }
    rejects([&] { (void)index_directory(images, embedder, store); }, "Corrupt image silently skipped");
    fs::remove(images / "z.jpg");
    store.fail = true;
    rejects([&] { (void)index_directory(images, embedder, store); }, "Store failure swallowed");
}

void video_checks(const fs::path& root) {
    const auto input = root / "video.avi";
    cv::VideoWriter writer(input.string(), cv::CAP_OPENCV_MJPEG,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, {160, 120});
    require(writer.isOpened(), "Cannot create local video fixture");
    for (int i = 0; i < 7; ++i) writer.write(cv::Mat(120, 160, CV_8UC3, cv::Scalar(10, 20, 30)));
    writer.release();
    PixelEmbedder embedder;
    FrameDetector detector;
    RecordingStore store;
    VideoIndexConfig config; config.frame_stride = 3; config.detector_signature = "fixture-v1";
    const auto summary = index_video(input, detector, embedder, store, config);
    require(summary.decoded_frames == 7 && summary.sampled_frames == 3 && summary.indexed_items == 4 &&
        summary.stop_reason == "end_of_stream_or_decode_stop", "Video sampling counters incorrect");
    require(detector.frames == std::vector<int>({0, 3, 6}), "Detector ran on unsampled frames");
    const auto initial = store.records;
    for (const auto& [id, metadata] : initial) {
        (void)id;
        require(metadata.at("kind") == "video_crop" && metadata.at("source_id") == summary.source_id,
            "Video provenance missing");
        const int frame = std::stoi(metadata.at("frame_index"));
        require(std::stoi(metadata.at("timestamp_ms")) == frame * 100, "Stored timestamp incorrect");
    }
    detector.reverse = true;
    (void)index_video(input, detector, embedder, store, config);
    require(store.records == initial, "Reordered detections changed IDs or metadata on retry");
    config.frame_stride = 2;
    (void)index_video(input, detector, embedder, store, config);
    require(store.records.size() == 8, "Changing stride duplicated overlapping frames");
    RecordingStore limited_store;
    config.frame_stride = 3; config.max_frames = 5;
    const auto partial = index_video(input, detector, embedder, limited_store, config);
    require(partial.decoded_frames == 5 && partial.sampled_frames == 2 && partial.indexed_items == 2 &&
        partial.stop_reason == "frame_limit", "Decoded frame limit incorrect");
    config.max_frames = 0;
    config.detector_signature = "fixture-v2";
    require(index_video(input, detector, embedder, limited_store, config).source_id != summary.source_id,
        "Detector versions must have independent IDs");
    detector.empty = true;
    RecordingStore empty_store;
    require(index_video(input, detector, embedder, empty_store, config).indexed_items == 0,
        "No detections must be a successful empty run");
    detector.empty = false; detector.invalid = true;
    rejects([&] { (void)index_video(input, detector, embedder, empty_store, config); }, "Invalid crop accepted");
    require(empty_store.records.empty(), "Invalid crops must fail before embedding/upsert");
    detector.invalid = false;
    config.source_metadata={{"origin","live_archive"},{"live_session_id","live-1-1"}};
    for(int i=0;i<7;++i) config.frame_metadata.push_back({{"live_arrival_ms",std::to_string(i*150)}});
    RecordingStore archive_store;
    const auto archived=index_video(input,detector,embedder,archive_store,config);
    require(archived.source_id!=summary.source_id && !archive_store.records.empty(),"Archive identity must preserve provenance");
    for(const auto& [id,row]:archive_store.records) {
        (void)id; const auto frame=std::stoi(row.at("frame_index"));
        require(row.at("origin")=="live_archive" && std::stoi(row.at("live_arrival_ms"))==frame*150 &&
            std::stoi(row.at("timestamp_ms"))==frame*100,"CFR seek and live arrival provenance were conflated");
    }
    config.source_metadata["timestamp_ms"]="override";
    rejects([&] { (void)index_video(input,detector,embedder,archive_store,config); },"Reserved provenance override accepted");
    config.source_metadata.clear(); config.frame_metadata.clear();
    bool cancel = false, cancelled = false;
    try {
        (void)index_video(input, detector, embedder, empty_store, config,
            [&](const IndexSummary& progress) { if (progress.indexed_items == 1) cancel = true; }, [&] { return cancel; });
    } catch (const IndexCancelled&) { cancelled = true; }
    require(cancelled && empty_store.records.size() == 1, "Cancellation did not retain exactly the completed writes");
    config.frame_stride = 0;
    rejects([&] { (void)index_video(input, detector, embedder, store, config); }, "Zero stride accepted");
    config.frame_stride = 1;
    rejects([&] { (void)index_video(root / "missing.avi", detector, embedder, store, config); }, "Missing video accepted");
}
}

int main() {
    const auto root = fs::temp_directory_path() / ("aegis-indexing-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root);
        directory_checks(root);
        video_checks(root);
        fs::remove_all(root); // Only this invocation's uniquely created fixture directory.
        std::cout << "Directory/video indexing, sampling, provenance, retries and errors passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << " (fixtures: " << root << ")\n";
        return 1;
    }
}
