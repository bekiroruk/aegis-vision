#include "aegisvision/live_archive.hpp"
#include "../src/archive_scan_retry.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
namespace fs = std::filesystem;
using aegisvision::vision::ArchivedSegment;
using aegisvision::vision::LiveArchive;
using aegisvision::vision::LiveArchiveConfig;
using aegisvision::vision::LiveFrame;
using Json = nlohmann::json;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
Json read_json(const fs::path& path) { std::ifstream input(path); Json value; input >> value; return value; }
void scan_retry_contract(const fs::path& root) {
    using aegisvision::vision::detail::stable_archive_scan;
    const auto staging = root / "metadata.json.partial", published = root / "metadata.json";
    { std::ofstream output(staging); output << "fixture"; }
    int filesystem_calls = 0;
    const auto bytes = stable_archive_scan([&] {
        const auto enumerated = fs::exists(staging) ? staging : published;
        if (++filesystem_calls == 1) fs::rename(staging,published);
        return fs::file_size(enumerated); // First enumeration is now stale.
    });
    require(filesystem_calls == 2 && bytes == 7,"Real renamed file did not restart scan");
    int calls = 0;
    const auto result = stable_archive_scan([&] {
        if (++calls < 3) throw fs::filesystem_error("renamed staging file",
            std::make_error_code(std::errc::no_such_file_or_directory));
        return 42;
    });
    require(calls == 3 && result == 42,"Transient scan did not restart to a complete result");
    calls = 0;
    bool bounded = false;
    try { stable_archive_scan([&]() -> int {
        ++calls; throw fs::filesystem_error("still changing",
            std::make_error_code(std::errc::no_such_file_or_directory));
    }); } catch (const std::runtime_error& e) { bounded = std::string(e.what()) == "archive_scan_unstable"; }
    require(bounded && calls == 3,"Unstable scan must fail closed after three attempts");
    calls = 0;
    bool denied = false;
    try { stable_archive_scan([&]() -> int {
        ++calls; throw fs::filesystem_error("denied",std::make_error_code(std::errc::permission_denied));
    }); } catch (const fs::filesystem_error& e) { denied = e.code() == std::errc::permission_denied; }
    require(denied && calls == 1,"Permission failure must not be retried or ignored");
    calls = 0;
    bool quota = false;
    try { stable_archive_scan([&]() -> int { ++calls; throw std::runtime_error("archive_disk_quota"); }); }
    catch (const std::runtime_error& e) { quota = std::string(e.what()) == "archive_disk_quota"; }
    require(quota && calls == 1,"Quota failure must not be retried or ignored");
}
LiveFrame frame(std::uint64_t sequence, std::int64_t arrival, std::uint64_t session = 1,
    cv::Size size = {160, 120}) {
    LiveFrame value;
    value.image = cv::Mat(size, CV_8UC3, cv::Scalar(30 + static_cast<int>(sequence % 50), 70, 140));
    value.sequence = sequence; value.arrival_ms = arrival; value.session = session;
    value.arrived = std::chrono::steady_clock::now();
    return value;
}
LiveArchiveConfig config(const fs::path& root) {
    LiveArchiveConfig value; value.enabled = true; value.root = root;
    return value;
}
int decode(const fs::path& path, cv::Size expected) {
    cv::VideoCapture capture(path.string(), cv::CAP_OPENCV_MJPEG);
    require(capture.isOpened(), "Raw finite MJPEG AVI could not be opened by OpenCV");
    require(std::abs(capture.get(cv::CAP_PROP_FPS) - 10) < .01, "CFR output FPS changed");
    cv::Mat image; int frames = 0;
    while (capture.read(image)) {
        require(image.type() == CV_8UC3 && image.size() == expected, "Archived frame type/size changed");
        const auto pixel = image.at<cv::Vec3b>(20, 20);
        require(std::abs(pixel[1] - 70) <= 5 && std::abs(pixel[2] - 140) <= 5,
            "Archive contains an annotation or unrelated pixels");
        ++frames;
    }
    return frames;
}
void boundaries(const fs::path& root) {
    auto c = config(root / "boundaries"); c.segment_seconds = 1; c.max_frames = 2;
    std::vector<ArchivedSegment> sealed;
    LiveArchive archive(c, "live-test-1", "local-camera", [&](const auto& value) {
        require(fs::exists(value.directory / "sealed.json"), "Callback ran before atomic integrity seal committed");
        sealed.push_back(value);
    });
    require(!fs::exists(c.root), "Archive eagerly creates directories before any frame");
    archive.accept(frame(1, 10), 1);
    archive.accept(frame(2, 510), 1);
    archive.accept(frame(3, 1010), 1); // Time and frame boundary: closes 2 frames.
    archive.accept(frame(4, 1300, 2), 2); // Source reconnect/epoch boundary.
    archive.accept(frame(5, 1400, 2, {180, 120}), 2); // Format boundary.
    archive.finish(); archive.finish();
    require(sealed.size() == 4 && archive.snapshot()["closed_segments"] == 4, "Segment boundaries/final flush lost data");
    require(archive.snapshot()["state"] == "finished" && archive.snapshot()["queued_segments"] == 4,
        "Finished archive/callback state is incorrect");
    for (std::size_t i = 0; i < sealed.size(); ++i) {
        const auto& segment = sealed[i]; const auto manifest = read_json(segment.manifest_path);
        const auto seal = read_json(segment.directory / "sealed.json");
        for (const auto* key : {"raw_sha256", "manifest_sha256"}) {
            const auto hash = seal.at(key).get<std::string>();
            require(hash.size() == 64 && std::all_of(hash.begin(), hash.end(), [](char value) {
                return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
            }), "Integrity seal omitted a canonical SHA256 digest");
        }
        if (i > 0) require(seal["manifest_sha256"] != read_json(sealed[i - 1].directory / "sealed.json")["manifest_sha256"],
            "Different sealed segment manifests were assigned the same digest");
        require(segment.index == static_cast<int>(i + 1) && manifest["segment_index"] == segment.index,
            "One-based segment IDs changed");
        require(manifest["session_id"] == "live-test-1" && manifest["source_id"] == "local-camera" &&
            manifest["raw_sealed"] == true && manifest["output_fps"] == 10,
            "Manifest omitted archive identity/encoding contract");
        require(manifest["time_basis"] == "decode_arrival_elapsed_ms_not_camera_pts" &&
            manifest["playback_time_basis"] == "analyzed_frame_index_divided_by_output_fps_cfr",
            "Decode-arrival time was misrepresented as camera PTS");
        require(manifest["frame_map"].size() == static_cast<std::size_t>(segment.frames), "Frame mapping incomplete");
        require(fs::exists(segment.directory / "index-queued.json") && !fs::exists(segment.media_path),
            "Recorder incorrectly wrote encoded MP4 or missed admission marker");
        require(decode(segment.raw_path, i == 3 ? cv::Size{180, 120} : cv::Size{160, 120}) == segment.frames,
            "AVI decoded-frame count differs from sealed manifest");
        for (int n = 0; n < segment.frames; ++n) {
            const auto& item = manifest["frame_map"][n];
            require(item["frame_index"] == n && item.contains("arrival_ms") && item.contains("source_sequence") &&
                item["source_session"] == manifest["source_session"] && item["tracking_epoch"] == manifest["tracking_epoch"],
                "Frame arrival/source/epoch mapping is incorrect");
        }
    }
    require(read_json(sealed[0].manifest_path)["arrival_start_ms"] == 10 &&
        read_json(sealed[0].manifest_path)["arrival_end_ms"] == 510, "Arrival span changed to CFR playback duration");
    const auto original_bytes = fs::file_size(sealed[0].raw_path);
    LiveArchive duplicate(c, "live-test-1", "local-camera"); duplicate.accept(frame(1, 0), 1);
    require(duplicate.snapshot()["state"] == "error" && fs::file_size(sealed[0].raw_path) == original_bytes,
        "Existing immutable session output was overwritten");

    auto timed = config(root / "independent-boundaries"); timed.segment_seconds = 1;
    std::vector<ArchivedSegment> split;
    {
        LiveArchive separated(timed, "time-epoch", "camera", [&](const auto& value) { split.push_back(value); });
        separated.accept(frame(1, 0), 1);
        separated.accept(frame(2, 1000), 1); // Time boundary alone, below max_frames.
        separated.accept(frame(3, 1001), 2); // Tracking epoch alone, same source session.
        separated.accept(frame(4, 1002), 2);
        // Destruction flushes the last segment just like explicit stop.
    }
    require(split.size() == 3 && split[0].frames == 1 && split[1].frames == 1 && split[2].frames == 2,
        "Independent time/epoch changes or destructor flush were ignored");
}
void quotas(const fs::path& root) {
    auto c = config(root / "session-limit"); c.max_frames = 1; c.max_segments_per_session = 2;
    LiveArchive archive(c, "first", "camera");
    for (int i = 0; i < 5; ++i) archive.accept(frame(i + 1, i * 10), 1);
    archive.finish();
    require(archive.snapshot()["closed_segments"] == 2 && archive.snapshot()["state"] == "limit_reached" &&
        !fs::exists(c.root / "first/segment-0003"), "Per-session segment quota is not a hard bound");

    auto global = config(root / "global-limit"); global.max_total_segments = 2; global.max_segments_per_session = 2;
    global.max_bytes = 64ULL * 1024 * 1024; global.max_frames = 1;
    LiveArchive initial(global, "first", "camera"); initial.accept(frame(1, 0), 1); initial.accept(frame(2, 1), 1); initial.finish();
    LiveArchive next(global, "second", "camera"); next.accept(frame(3, 0), 1);
    require(next.snapshot()["state"] == "error" && !fs::exists(global.root / "second"),
        "Global slot quota admits extra or empty session directories");

    auto dirty = config(root / "unmanaged"); dirty.max_bytes = 32ULL * 1024 * 1024;
    fs::create_directories(dirty.root);
    { std::ofstream file(dirty.root / "unmanaged.bin"); file << 'x'; }
    LiveArchive unmanaged(dirty, "third", "camera"); unmanaged.accept(frame(1, 0), 1);
    require(unmanaged.snapshot()["state"] == "error" && fs::exists(dirty.root / "unmanaged.bin") &&
        !fs::exists(dirty.root / "third"), "Unmanaged bytes are ignored or destructively removed");

    auto oversized = config(root / "oversized-slot"); oversized.max_total_segments = 3;
    oversized.max_segments_per_session = 2; oversized.max_bytes = 96ULL * 1024 * 1024;
    fs::create_directories(oversized.root / "existing/first");
    fs::create_directories(oversized.root / "existing/second");
    {
        std::ofstream file(oversized.root / "existing/first/unmanaged.bin", std::ios::binary);
        file.seekp(50ULL * 1024 * 1024 - 1); file.put('x');
    }
    LiveArchive over_reserved(oversized, "third", "camera"); over_reserved.accept(frame(1, 0), 1);
    require(over_reserved.snapshot()["state"] == "error" && !fs::exists(oversized.root / "third"),
        "Oversized slot bytes were averaged away against a different empty reservation");

    auto small = config(root / "raw-limit"); small.raw_limit = 64 * 1024; small.max_segments_per_session = 8;
    std::vector<ArchivedSegment> sealed;
    LiveArchive bounded(small, "small", "camera", [&](const auto& value) { sealed.push_back(value); });
    for (int i = 0; i < 30; ++i) {
        auto value = frame(i + 1, i * 10);
        cv::RNG random(100 + i); random.fill(value.image, cv::RNG::UNIFORM, 0, 256);
        bounded.accept(value, 1);
    }
    bounded.finish();
    require(!sealed.empty() && sealed.size() > 1, "Raw byte boundary did not create finite smaller segments");
    for (const auto& segment : sealed) {
        require(fs::file_size(segment.raw_path) <= small.raw_limit, "Raw hard byte cap was exceeded");
        cv::VideoCapture capture(segment.raw_path.string(), cv::CAP_OPENCV_MJPEG);
        require(capture.isOpened(), "Byte-limited segment has corrupt AVI headers");
        cv::Mat image; int count = 0; while (capture.read(image)) ++count;
        require(count == segment.frames, "Byte-limit rotation truncates the final frame/index");
    }
}
void failures(const fs::path& root) {
    auto c = config(root / "callback"); c.max_frames = 1;
    int admissions = 0;
    LiveArchive unavailable(c, "pending", "camera", [&](const auto&) { ++admissions; throw std::runtime_error("private path secret"); });
    unavailable.accept(frame(1, 0), 1); unavailable.accept(frame(2, 10), 1); unavailable.finish();
    require(admissions == 2 && unavailable.snapshot()["closed_segments"] == 2 &&
        unavailable.snapshot()["state"] == "finished" && unavailable.snapshot()["index_queue_failures"] == 2,
        "Queue admission failure poisoned the recording or preview worker");
    const auto pending = c.root / "pending/segment-0001";
    require(read_json(pending / "manifest.json")["index_status"] == "pending" &&
        read_json(pending / "index-pending.json")["error"] == "index_queue_admission_failed" &&
        unavailable.snapshot().dump().find("secret") == std::string::npos,
        "Unadmitted sealed segment is missing a retry marker or leaks internal errors");
    require(decode(pending / "raw.avi", {160, 120}) == 1, "Pending segment is not a playable durable recording");

    LiveArchiveConfig disabled; disabled.root = root / "disabled";
    LiveArchive off(disabled, "unused", "camera"); off.accept(frame(1, 0), 1); off.finish();
    require(off.snapshot()["state"] == "disabled" && !fs::exists(disabled.root), "Disabled archive has disk side effects");

    auto invalid = config(root / "invalid");
    LiveArchive bad_id(invalid, "../escape", "camera"); bad_id.accept(frame(1, 0), 1);
    require(bad_id.snapshot()["state"] == "error" && !fs::exists(invalid.root), "Unsafe session ID escaped archive root");
    LiveArchive bad_frame(invalid, "invalid-frame", "camera");
    auto floating = frame(1, 0); floating.image = cv::Mat(120, 160, CV_32FC3);
    bad_frame.accept(floating, 1);
    require(bad_frame.snapshot()["state"] == "error" && !fs::exists(invalid.root), "Unsupported frame caused disk artifacts");

    auto dimensions = config(root / "oversized-dimensions"); LiveArchive dimension_limit(dimensions, "wide", "camera");
    dimension_limit.accept(frame(1, 0, 1, {1922, 120}), 1);
    require(dimension_limit.snapshot()["error"] == "archive_invalid_frame" && !fs::exists(dimensions.root),
        "Recorder accepted dimensions that the finite encoder/manifest contract cannot index");

    auto too_small = config(root / "large-frame"); too_small.raw_limit = 64 * 1024;
    LiveArchive huge(too_small, "huge", "camera"); auto noisy = frame(1, 0, 1, {640, 480});
    cv::RNG random(42); random.fill(noisy.image, cv::RNG::UNIFORM, 0, 256); huge.accept(noisy, 1);
    require(huge.snapshot()["error"] == "archive_frame_too_large" && !fs::exists(too_small.root),
        "A single oversized JPEG exceeded the raw cap or created empty quota slots");

    auto changed = config(root / "external-failure"); LiveArchive partial(changed, "partial", "camera");
    partial.accept(frame(1, 0), 1);
    auto malformed = frame(2, 1); malformed.image = cv::Mat(120, 160, CV_32FC3);
    partial.accept(malformed, 1); partial.finish();
    require(partial.snapshot()["state"] == "error" && fs::exists(changed.root / "partial/segment-0001/raw.avi") &&
        !fs::exists(changed.root / "partial/segment-0001/manifest.json"),
        "Invalid input deleted material recording or falsely sealed partial output");

    const auto outside = root / "outside"; fs::create_directory(outside);
    const auto links = root / "links"; fs::create_directory(links);
    std::error_code error;
    fs::create_directory_symlink(outside, links / "alias", error);
    if (!error) {
        auto linked = config(links / "alias/live-archive");
        LiveArchive alias(linked, "unsafe", "camera"); alias.accept(frame(1, 0), 1);
        require(alias.snapshot()["state"] == "error" && !fs::exists(outside / "live-archive"),
            "Archive follows a symlink ancestor outside the assigned root");
        linked.root = links;
        LiveArchive tree_alias(linked, "unsafe", "camera"); tree_alias.accept(frame(1, 0), 1);
        require(tree_alias.snapshot()["state"] == "error", "Archive scan follows a symlink tree entry");
    } else std::cout << "Directory symlink creation unavailable; alias test skipped on this host\n";
}
}
int main() {
    const auto root = fs::temp_directory_path() / ("aegis-live-archive-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directory(root);
        scan_retry_contract(root); boundaries(root); quotas(root); failures(root);
        fs::remove_all(root); // Only this owned, uniquely named test scratch root.
        std::cout << "Live archive finite AVI, mapping, quota and failure tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::cerr << "Owned diagnostic artifacts retained at " << root << '\n';
        return 1;
    }
}
