#pragma once
#include "aegisvision/qdrant.hpp"
#include "aegisvision/video.hpp"
#include "aegisvision/live.hpp"
#include "aegisvision/yolo.hpp"
#include <filesystem>
#include <memory>

namespace aegisvision::vision {
enum class ApplicationMode { Image, Video, Search, Stream };

// Resolved, validated settings. Model paths are absolute and relative to the TOML file.
struct ApplicationSettings {
    ApplicationMode mode{ApplicationMode::Image};
    std::filesystem::path detector_model;
    YoloConfig detector;
    VideoConfig video;
    LiveConfig live;
    std::filesystem::path clip_bundle;
    QdrantConfig qdrant;
    std::filesystem::path reid_bundle;
};

[[nodiscard]] ApplicationSettings load_application_settings(const std::filesystem::path& file);
// Image/video adapter factory. Model graph validity is checked by YoloDetector on load.
[[nodiscard]] std::unique_ptr<IDetector> make_configured_detector(const ApplicationSettings& settings);
} // namespace aegisvision::vision
