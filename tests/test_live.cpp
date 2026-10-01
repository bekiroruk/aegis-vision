#include "aegisvision/live.hpp"
#include <opencv2/videoio.hpp>
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>

namespace {
using namespace aegisvision;
using namespace aegisvision::vision;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void until(F predicate) {
    const auto limit = Clock::now() + 2s;
    while (!predicate()) {
        if (Clock::now() > limit) throw std::runtime_error("Timed out waiting for fake capture");
        std::this_thread::sleep_for(1ms);
    }
}
template<class F> void rejects(F action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Invalid live configuration accepted");
}
struct State {
    std::atomic<bool> allow_open{true}, disconnect{false}, unlimited{false}, invalid{false};
    std::atomic<int> credits{0}, emitted{0}, opens{0}, closes{0};
};
class FakeCapture final : public ILiveCapture {
public:
    explicit FakeCapture(std::shared_ptr<State> state) : state_(std::move(state)) {}
    bool open(const std::string&, const LiveConfig& c) override {
        timeout_ = c.read_timeout_ms;
        ++state_->opens;
        return state_->allow_open;
    }
    bool read(cv::Mat& image) override {
        const auto end = Clock::now() + std::chrono::milliseconds(timeout_);
        while (Clock::now() < end) {
            if (state_->disconnect.exchange(false)) return false;
            if (state_->unlimited || state_->credits > 0) {
                if (!state_->unlimited) --state_->credits;
                else std::this_thread::sleep_for(5ms);
                const int n = ++state_->emitted;
                buffer_.create(120, 160, state_->invalid ? CV_32FC3 : CV_8UC3);
                buffer_.setTo(cv::Scalar(n % 256, 40, 50));
                image = buffer_; // Deliberate reused storage: source must clone.
                return true;
            }
            std::this_thread::sleep_for(1ms);
        }
        return false;
    }
    void close() override { ++state_->closes; }
private:
    std::shared_ptr<State> state_;
    cv::Mat buffer_;
    int timeout_{200};
};
LiveCaptureFactory factory(const std::shared_ptr<State>& state) {
    return [state] { return std::make_unique<FakeCapture>(state); };
}
LiveConfig fast_config() {
    LiveConfig c;
    c.duration_seconds = 1; c.open_timeout_ms = 50; c.read_timeout_ms = 1000;
    c.reconnect_initial_ms = 2; c.reconnect_max_ms = 10; c.max_outage_ms = 1000;
    c.max_frame_age_ms = 250;
    return c;
}
void source_checks() {
    const auto c = fast_config();
    auto state = std::make_shared<State>();
    LiveSource source("rtsp://127.0.0.1/test", c, factory(state));
    state->credits = 50;
    until([&] { return source.stats().decoded_frames == 50; });
    auto frame = source.next(0ms);
    require(frame && frame->sequence == 50 && frame->session == 1, "Drop-oldest did not keep newest frame");
    require(source.stats().dropped_overflow == 49 && source.stats().queue_high_watermark == 1,
        "Queue capacity/backpressure accounting failed");
    state->credits = 1;
    until([&] { return source.stats().decoded_frames == 51; });
    require(frame->image.at<cv::Vec3b>(0,0)[0] == 50, "Delivered image aliases decoder memory");
    std::this_thread::sleep_for(300ms);
    require(!source.next(0ms) && source.stats().dropped_stale == 1, "Old frame reached analysis");
    state->disconnect = true;
    until([&] { return source.stats().read_failures >= 1; });
    state->credits = 1;
    frame = source.next(1000ms);
    require(frame && frame->session == 2 && frame->sequence == 52, "Disconnected source did not reconnect with new session");
    std::thread stop_a([&] { source.stop(); });
    std::thread stop_b([&] { source.stop(); });
    stop_a.join(); stop_b.join();
    require(source.finished() && source.stats().stop_reason == "stopped", "Reader was not joined");
    auto s = source.stats();
    require(s.decoded_frames == s.delivered_frames + s.dropped_overflow + s.dropped_stale + s.dropped_disconnect,
        "Decoded frame conservation failed");

    auto absent = std::make_shared<State>(); absent->allow_open = false;
    auto outage = c; outage.max_outage_ms = 40;
    LiveSource missing("rtsp://127.0.0.1/missing", outage, factory(absent));
    until([&] { return missing.finished(); });
    require(missing.stats().stop_reason == "outage_limit" && missing.stats().connection_attempts > 1,
        "Unavailable source did not stop after bounded retries");

    auto invalid = std::make_shared<State>(); invalid->invalid = true; invalid->credits = 1;
    LiveSource bad("rtsp://127.0.0.1/bad", c, factory(invalid));
    until([&] { return bad.finished(); });
    require(bad.stats().stop_reason == "unsupported_frame", "Invalid frame was published");
    rejects([&] { LiveSource wrong("file://test", c); });
    rejects([&] { validate_rtsp_url("rtsp://user:secret@host/test"); });
    rejects([&] { validate_rtsp_url("rtsp://host/test?token=secret"); });
    rejects([&] { auto wrong = c; wrong.queue_capacity = 0; validate_live_config(wrong); });
    rejects([&] { auto wrong = c; wrong.reconnect_max_ms = 1; validate_live_config(wrong); });
}
class SlowDetector : public IDetector {
public:
    bool fail{};
    std::vector<Detection> detect(const Frame& frame) override {
        require(frame.image && frame.timestamp_ms >= 0 && frame.source_id.starts_with("rtsp-session-"),
            "Live frame contract not passed to detector");
        if (fail) throw std::runtime_error("Intentional detector failure");
        std::this_thread::sleep_for(20ms);
        return {{{10,10,80,90}, "person", .9F, {}, {}}};
    }
};
}
int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("aegis-live-" + std::to_string(Clock::now().time_since_epoch().count()));
    try {
        if (argc == 3 && std::string(argv[1]) == "--rtsp") {
            // Network transport integration only; not a detector accuracy test.
            LiveSource source(argv[2]);
            const auto end = Clock::now() + 45s;
            std::uint64_t previous_session = 0;
            int received = 0;
            while (Clock::now() < end && !source.finished()) {
                if (auto frame = source.next()) {
                    require(frame->image.type() == CV_8UC3, "RTSP pixels were not decoded");
                    if (frame->session != previous_session) {
                        previous_session = frame->session;
                        std::cout << "RTSP_SESSION=" << previous_session << std::endl;
                    }
                    ++received;
                }
            }
            source.stop();
            const auto s = source.stats();
            std::cout << "RTSP stats: attempts=" << s.connection_attempts << " sessions=" << s.sessions <<
                " decoded=" << s.decoded_frames << " received=" << received << " reason=" << s.stop_reason << std::endl;
            require(received >= 20 && previous_session >= 2 && s.read_failures >= 1 && s.queue_high_watermark <= 1,
                "Real RTSP decode/disconnect/reconnect integration failed");
            std::cout << "RTSP recovery passed; decoded=" << s.decoded_frames << " received=" << received << '\n';
            return 0;
        }
        require(argc == 1, "Use --rtsp URL for real transport integration");
        source_checks();
        auto state = std::make_shared<State>(); state->unlimited = true;
        SlowDetector detector;
        auto c = fast_config();
        const auto summary = process_stream("rtsp://127.0.0.1/test", root / "ok", detector, {}, c, factory(state));
        require(summary.processed_frames > 10 && summary.source.dropped_overflow > 0 &&
            summary.source.queue_high_watermark == 1 && summary.stop_reason == "duration_limit",
            "Live pipeline did not apply backpressure / duration limit");
        cv::VideoCapture video((root / "ok/tracked.avi").string(), cv::CAP_OPENCV_MJPEG);
        require(video.isOpened(), "Live recording not playable");
        cv::Mat image; int count = 0;
        while (video.read(image)) ++count;
        video.release();
        require(count == summary.processed_frames && fs::exists(root / "ok/summary.json") &&
            fs::exists(root / "ok/latest.jpg"), "Live output/report incomplete");
        std::ifstream frames(root / "ok/frames.csv");
        std::string line; int lines = 0; while (std::getline(frames, line)) ++lines; frames.close();
        require(lines == count + 1, "Output frame timeline is incomplete");
        rejects([&] { (void)process_stream("rtsp://127.0.0.1/test", root / "ok", detector, {}, c, factory(state)); });
        detector.fail = true;
        const auto before = Clock::now();
        rejects([&] { (void)process_stream("rtsp://127.0.0.1/test", root / "fail", detector, {}, c, factory(state)); });
        require(Clock::now() - before < 2s, "Exception did not join source promptly");
        fs::remove_all(root); // This test's unique scratch directory only.
        std::cout << "Live queue, owned frames, staleness, reconnect, outage, recording and failure tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << " (fixtures " << root << ")\n"; return 1; }
}
