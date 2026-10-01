#include "aegisvision/application_config.hpp"
#include "aegisvision/clip.hpp"
#include "aegisvision/service.hpp"
#include <charconv>
#include <iostream>

namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "Usage: aegisvision_server SEARCH.toml DETECTOR.toml MEDIA_DIR WEB_DIR [PORT [JOB_DB]]\n"
                "Local HTTP worker + dashboard. Defaults: port 8090, artifacts/service/jobs.sqlite. Creates Qdrant collection if missing.\n";
            return 0;
        }
        if (args.size() < 5 || args.size() > 7) throw std::invalid_argument("Missing arguments; run --help");
        int port = 8090;
        if (args.size() >= 6) {
            const auto text = args[5].string();
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
            if (error != std::errc{} || end != text.data() + text.size() || port < 1 || port > 65535)
                throw std::invalid_argument("PORT must be 1..65535");
        }
        auto search = aegisvision::vision::load_application_settings(args[1]);
        auto detection = aegisvision::vision::load_application_settings(args[2]);
        if (search.mode != aegisvision::vision::ApplicationMode::Search || detection.mode != aegisvision::vision::ApplicationMode::Image)
            throw std::invalid_argument("Expected search-mode and image-mode TOML configurations");
        std::cout << "Loading CLIP and YOLO models..." << std::endl;
        aegisvision::ClipEmbedder clip(search.clip_bundle);
        search.qdrant.embedding_space = clip.space_id();
        aegisvision::QdrantVectorStore store(search.qdrant, true);
        auto detector = aegisvision::vision::make_configured_detector(detection);
        aegisvision::ServiceConfig config{args[3], args[4],
            aegisvision::yolo_index_signature(detection.detector_model, detection.detector)};
        config.persistence.database = args.size() == 7 ? args[6] : std::filesystem::path("artifacts/service/jobs.sqlite");
        config.persistence.context = nlohmann::json::array({"clip-qdrant-v1", clip.space_id(),
            search.qdrant.host, search.qdrant.port, search.qdrant.collection, search.qdrant.dimension}).dump();
        aegisvision::LocalService service(*detector, clip, store, config);
        const auto bound = service.bind(port);
        std::cout << "AegisVision ready: http://127.0.0.1:" << bound << " / collection=" << search.qdrant.collection << std::endl;
        return service.listen() ? 0 : 1;
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
