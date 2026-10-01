#include "aegisvision/application_config.hpp"
#include "aegisvision/live.hpp"
#include <iostream>
#include <vector>

namespace {
int run(const std::vector<std::filesystem::path>& args) {
    if (args.size() == 2 && args[1] == "--help") {
        std::cout << "Usage: aegisvision_stream CONFIG.toml RTSP_URL NEW_OUTPUT_DIR\n"
            "Bounded CPU YOLO/track recording; pipeline.mode=stream. Reconnects and drops old frames.\n"
            "No URL credentials/query tokens. Writes tracked.avi, preview/latest.jpg and CSV/JSON reports.\n"
            "Arrival timestamps are not camera PTS; recorded playback omits dropped frames/outages.\n";
        return 0;
    }
    if (args.size() != 4) { std::cerr << "Use --help for stream usage\n"; return 2; }
    try {
        const auto raw_url = args[2].u8string();
        const std::string url(raw_url.begin(), raw_url.end());
        aegisvision::vision::validate_rtsp_url(url);
        const auto settings = aegisvision::vision::load_application_settings(args[1]);
        if (settings.mode != aegisvision::vision::ApplicationMode::Stream)
            throw std::invalid_argument("Stream CLI requires pipeline.mode=stream");
        auto detector = aegisvision::vision::make_configured_detector(settings);
        const auto result = aegisvision::vision::process_stream(url, args[3], *detector, settings.video, settings.live);
        std::cout << "frames=" << result.processed_frames << " sessions=" << result.source.sessions <<
            " attempts=" << result.source.connection_attempts << " reason=" << result.stop_reason << '\n';
        return result.processed_frames > 0 && result.stop_reason == "duration_limit" ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
