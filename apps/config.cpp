#include "aegisvision/application_config.hpp"
#include <iostream>
#include <vector>

namespace {
int run(const std::vector<std::filesystem::path>& args) {
    if (args.size() == 2 && args[1] == "--help") {
        std::cout << "Usage: aegisvision_config validate CONFIG.toml\n"
            "Checks TOML schema, supported adapters, ranges and local model paths.\n"
            "It does not load model graphs or connect to Qdrant.\n";
        return 0;
    }
    if (args.size() != 3 || args[1] != "validate") {
        std::cerr << "Usage: aegisvision_config validate CONFIG.toml\n";
        return 2;
    }
    try {
        const auto settings = aegisvision::vision::load_application_settings(args[2]);
        const char* mode = settings.mode == aegisvision::vision::ApplicationMode::Image ? "image" :
            settings.mode == aegisvision::vision::ApplicationMode::Video ? "video" :
            settings.mode == aegisvision::vision::ApplicationMode::Stream ? "stream" : "search";
        std::cout << "Valid configuration: mode=" << mode << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Invalid configuration: " << error.what() << '\n';
        return 1;
    }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
