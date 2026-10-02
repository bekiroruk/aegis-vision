#include "aegisvision/tracking.hpp"
#include "aegisvision/video.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char *message) {
    try {
        f();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}
Detection detection(float x = 10) { return {{x, 10, x + 50, 70}, "person", 0.9F, {}, {}}; }
void tracking_checks() {
    IoUTracker tracker(0.3F, 1);
    const auto id = tracker.update({detection()}).front().track_id;
    require(tracker.update({}).empty(), "Missing tracks should not be rendered");
    require(tracker.update({detection(12)}).front().track_id == id, "Short occlusion lost ID");
    (void)tracker.update({});
    (void)tracker.update({});
    require(tracker.update({detection()}).front().track_id != id, "Expired track reused");
    IoUTracker tie;
    const auto two = tie.update({detection(), detection()});
    require(two.size() == 2 && two[0].track_id != two[1].track_id, "Duplicate ID in a frame");
    require(tie.update({detection()}).front().track_id == two[0].track_id,
            "Tie-break not deterministic");
    auto car = detection();
    car.label = "car";
    require(tie.update({car}).front().track_id != two[0].track_id, "Track switched class");
    rejects([] { IoUTracker bad(std::numeric_limits<float>::quiet_NaN()); }, "NaN IoU accepted");
}
class SequenceDetector : public IDetector {
  public:
    bool empty{false};
    std::vector<Detection> detect(const Frame &frame) override {
        require(frame.image && frame.image->width == 160, "Video pixel buffer missing");
        const int index = std::stoi(frame.frame_id);
        require(frame.timestamp_ms == index * 100, "Frame timestamp mismatch");
        if (empty || index == 2)
            return {};
        return {detection(10.0F + index)};
    }
};
int decoded_frames(const fs::path &path) {
    cv::VideoCapture video(path.string(), cv::CAP_OPENCV_MJPEG);
    require(video.isOpened(), "Output video not readable");
    require(std::abs(video.get(cv::CAP_PROP_FPS) - 10) < 0.01, "Output FPS changed");
    cv::Mat frame;
    int count = 0;
    while (video.read(frame)) {
        require(frame.size() == cv::Size(160, 120), "Video dimensions changed");
        ++count;
    }
    return count;
}
} // namespace

int main() {
    const auto root = fs::temp_directory_path() /
                      ("aegis-video-test-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        tracking_checks();
        fs::create_directories(root);
        const auto input = root / "input.avi";
        cv::VideoWriter writer(input.string(), cv::CAP_OPENCV_MJPEG,
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, {160, 120});
        require(writer.isOpened(), "Fixture writer unavailable");
        for (int i = 0; i < 5; ++i)
            writer.write(cv::Mat(120, 160, CV_8UC3, cv::Scalar(30, 40, 50)));
        writer.release();
        SequenceDetector detector;
        const auto summary = vision::process_video(input, root / "full", detector);
        require(summary.processed_frames == 5 && summary.unique_track_ids == 1,
                "Video tracking continuity failed");
        require(decoded_frames(root / "full/tracked.avi") == 5, "Output lost video frames");
        std::ifstream csv(root / "full/tracks.csv");
        int lines = 0;
        std::string line;
        while (std::getline(csv, line))
            ++lines;
        csv.close();
        require(lines == 5, "CSV must have header plus four visible tracks");
        require(fs::exists(root / "full/summary.json"), "Summary missing");
        rejects([&] { (void)vision::process_video(input, root / "full", detector); },
                "Existing results overwritten");
        rejects(
            [&] { (void)vision::process_video(root / "missing.avi", root / "missing", detector); },
            "Missing video accepted");
        vision::VideoConfig limited;
        limited.max_frames = 2;
        const auto partial = vision::process_video(input, root / "limited", detector, limited);
        require(partial.processed_frames == 2 && partial.stop_reason == "frame_limit",
                "Frame limit ignored");
        require(decoded_frames(root / "limited/tracked.avi") == 2, "Limited video count wrong");
        vision::VideoConfig two_stage;
        two_stage.tracker_mode = vision::TrackerMode::TwoStage;
        const auto improved = vision::process_video(input, root / "two-stage", detector, two_stage);
        require(improved.unique_track_ids == 1 && improved.tracking_stats.reactivations == 1,
                "Two-stage video adapter did not preserve/recover the track");
        require(decoded_frames(root / "two-stage/tracked.avi") == 5,
                "Two-stage output video incomplete");
        auto kalman = two_stage;
        kalman.tracker_mode = vision::TrackerMode::Kalman;
        const auto motion = vision::process_video(input, root / "kalman", detector, kalman);
        require(motion.unique_track_ids == 1 && motion.kalman_stats.reactivations == 1,
                "Kalman video adapter did not preserve/recover the track");
        require(decoded_frames(root / "kalman/tracked.avi") == 5, "Kalman output video incomplete");
        detector.empty = true;
        require(vision::process_video(input, root / "empty", detector).unique_track_ids == 0,
                "Empty detections not handled");
        std::cout << "Video decode/encode, ID continuity, expiry, timestamps, limits and error "
                     "tests passed\n";
        fs::remove_all(root); // This test's unique scratch directory only.
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << " (fixtures: " << root << ")\n";
        return 1;
    }
}
