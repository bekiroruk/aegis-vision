#include "aegisvision/application_config.hpp"
#include <toml++/toml.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>

namespace aegisvision::vision {
namespace {
using Table = toml::table;
using Node = toml::node;

void keys(const Table& table, std::string_view section, std::initializer_list<std::string_view> allowed) {
    for (const auto& [key, value] : table) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key.str()) == allowed.end()) {
            throw std::invalid_argument("Unknown TOML key " + std::string(section) + "." + std::string(key.str()));
        }
    }
}

const Table& section(const Table& root, std::string_view name, bool required = true) {
    const auto* value = root.get(name);
    if (!value && !required) {
        static const Table empty;
        return empty;
    }
    if (!value || !value->is_table()) throw std::invalid_argument("Missing or non-table TOML section [" + std::string(name) + "]");
    return *value->as_table();
}

const Node* field(const Table& table, std::string_view section_name, std::string_view name, bool required) {
    const auto* value = table.get(name);
    if (!value && required) throw std::invalid_argument("Missing TOML field " + std::string(section_name) + "." + std::string(name));
    return value;
}

std::string string(const Table& table, std::string_view section_name, std::string_view name,
                   bool required = true, std::string fallback = {}) {
    const auto* value = field(table, section_name, name, required);
    if (!value) return fallback;
    auto parsed = value->value<std::string>();
    if (!value->is_string() || !parsed || parsed->empty())
        throw std::invalid_argument("Expected nonempty TOML string " + std::string(section_name) + "." + std::string(name));
    return *parsed;
}

std::int64_t integer(const Table& table, std::string_view section_name, std::string_view name,
                     std::int64_t fallback, std::int64_t minimum, std::int64_t maximum, bool required = false) {
    const auto* value = field(table, section_name, name, required);
    if (!value) return fallback;
    const auto parsed = value->value<std::int64_t>();
    if (!value->is_integer() || !parsed || *parsed < minimum || *parsed > maximum)
        throw std::invalid_argument("Invalid TOML integer " + std::string(section_name) + "." + std::string(name));
    return *parsed;
}

double real(const Table& table, std::string_view section_name, std::string_view name,
            double fallback, double minimum, double maximum) {
    const auto* value = field(table, section_name, name, false);
    if (!value) return fallback;
    // TOML integers are accepted for numeric settings, but strings/bools are not.
    const auto floating = value->value<double>();
    const auto integral = value->value<std::int64_t>();
    const double number = floating ? *floating : (integral ? static_cast<double>(*integral) : std::numeric_limits<double>::quiet_NaN());
    if ((!value->is_floating_point() && !value->is_integer()) ||
        !std::isfinite(number) || number < minimum || number > maximum)
        throw std::invalid_argument("Invalid TOML number " + std::string(section_name) + "." + std::string(name));
    return number;
}

std::filesystem::path model_path(const Table& table, std::string_view section_name, std::string_view name,
                                 const std::filesystem::path& directory, bool is_directory = false) {
    auto path = std::filesystem::u8path(string(table, section_name, name));
    if (path.is_relative()) path = directory / path;
    path = std::filesystem::weakly_canonical(path);
    if (is_directory ? !std::filesystem::is_directory(path) : !std::filesystem::is_regular_file(path))
        throw std::invalid_argument("Missing " + std::string(section_name) + "." + std::string(name) + ": " + path.string());
    return path;
}

void runtime(const Table& root, ApplicationMode mode) {
    const auto& table = section(root, "runtime", false);
    keys(table, "runtime", {"device", "precision", "engine"});
    const auto device = string(table, "runtime", "device", false, "cpu");
    const auto precision = string(table, "runtime", "precision", false, "fp32");
    const auto engine = string(table, "runtime", "engine", false,
        mode == ApplicationMode::Search ? "onnxruntime" : "opencv-dnn");
    if (device != "cpu" || precision != "fp32" ||
        engine != (mode == ApplicationMode::Search ? "onnxruntime" : "opencv-dnn"))
        throw std::invalid_argument("Unsupported runtime: expected cpu/fp32 and mode's inference engine");
}
} // namespace

ApplicationSettings load_application_settings(const std::filesystem::path& file) {
    if (!std::filesystem::is_regular_file(file)) throw std::invalid_argument("TOML config file does not exist: " + file.string());
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read TOML config: " + file.string());
    const auto root = toml::parse(input);
    keys(root, "root", {"version", "pipeline", "detector", "tracking", "video", "stream", "embedding", "vector_store", "appearance", "runtime"});
    (void)integer(root, "root", "version", 0, 1, 1, true);
    const auto& pipeline = section(root, "pipeline");
    keys(pipeline, "pipeline", {"mode"});
    const auto mode = string(pipeline, "pipeline", "mode");
    ApplicationSettings settings;
    if (mode == "image") settings.mode = ApplicationMode::Image;
    else if (mode == "video") settings.mode = ApplicationMode::Video;
    else if (mode == "search") settings.mode = ApplicationMode::Search;
    else if (mode == "stream") settings.mode = ApplicationMode::Stream;
    else throw std::invalid_argument("pipeline.mode must be image, video, stream or search");
    runtime(root, settings.mode);
    if (settings.mode != ApplicationMode::Stream && root.contains("stream"))
        throw std::invalid_argument("[stream] requires stream mode");
    if (settings.mode == ApplicationMode::Stream && root.contains("video"))
        throw std::invalid_argument("[video] is incompatible with stream mode");
    if (settings.mode != ApplicationMode::Video && root.contains("appearance"))
        throw std::invalid_argument("[appearance] requires local video mode with tracking.backend=kalman-reid");

    const auto directory = std::filesystem::absolute(file).parent_path();
    if (settings.mode == ApplicationMode::Search) {
        for (auto name : {"detector", "tracking", "video"})
            if (root.contains(name)) throw std::invalid_argument("[" + std::string(name) + "] is incompatible with search mode");
        const auto& embedding = section(root, "embedding");
        keys(embedding, "embedding", {"backend", "bundle", "dimension"});
        if (string(embedding, "embedding", "backend") != "clip_onnx")
            throw std::invalid_argument("embedding.backend must be clip_onnx");
        if (integer(embedding, "embedding", "dimension", 0, 512, 512, true) != 512)
            throw std::invalid_argument("embedding.dimension must be 512");
        settings.clip_bundle = model_path(embedding, "embedding", "bundle", directory, true);
        for (auto name : {"manifest.json", "vision.onnx", "text.onnx", "tokenizer.json"})
            if (!std::filesystem::is_regular_file(settings.clip_bundle / name))
                throw std::invalid_argument("CLIP bundle is missing " + std::string(name));
        const auto& store = section(root, "vector_store");
        keys(store, "vector_store", {"backend", "host", "port", "collection", "dimension", "timeout_seconds"});
        if (string(store, "vector_store", "backend") != "qdrant")
            throw std::invalid_argument("vector_store.backend must be qdrant");
        settings.qdrant.host = string(store, "vector_store", "host", false, "127.0.0.1");
        if (settings.qdrant.host != "127.0.0.1" && settings.qdrant.host != "localhost")
            throw std::invalid_argument("vector_store.host must be local");
        settings.qdrant.port = static_cast<int>(integer(store, "vector_store", "port", 6333, 1, 65535));
        settings.qdrant.collection = string(store, "vector_store", "collection");
        if (!std::regex_match(settings.qdrant.collection, std::regex("[A-Za-z0-9_-]{1,64}")))
            throw std::invalid_argument("Invalid vector_store.collection");
        settings.qdrant.dimension = static_cast<std::size_t>(integer(store, "vector_store", "dimension", 0, 512, 512, true));
        settings.qdrant.timeout_seconds = static_cast<int>(integer(store, "vector_store", "timeout_seconds", 10, 1, 120));
        return settings;
    }

    for (auto name : {"embedding", "vector_store"})
        if (root.contains(name)) throw std::invalid_argument("[" + std::string(name) + "] is incompatible with image/video mode");
    const auto& detector = section(root, "detector");
    keys(detector, "detector", {"backend", "model", "confidence_threshold", "nms_iou_threshold", "input_size", "max_detections"});
    if (string(detector, "detector", "backend") != "yolo_onnx")
        throw std::invalid_argument("detector.backend must be yolo_onnx");
    settings.detector_model = model_path(detector, "detector", "model", directory);
    if (settings.detector_model.extension() != ".onnx") throw std::invalid_argument("detector.model must be an ONNX file");
    settings.detector.confidence_threshold = static_cast<float>(real(detector, "detector", "confidence_threshold", 0.35, 0.000001, 1));
    settings.detector.nms_iou_threshold = static_cast<float>(real(detector, "detector", "nms_iou_threshold", 0.45, 0.000001, 1));
    settings.detector.input_size = static_cast<int>(integer(detector, "detector", "input_size", 640, 32, 2048));
    if (settings.detector.input_size % 32 != 0) throw std::invalid_argument("detector.input_size must be divisible by 32");
    settings.detector.max_detections = static_cast<std::size_t>(integer(detector, "detector", "max_detections", 300, 1, 3000));
    if (settings.mode == ApplicationMode::Image) {
        for (auto name : {"tracking", "video"})
            if (root.contains(name)) throw std::invalid_argument("[" + std::string(name) + "] requires video mode");
        return settings;
    }
    const auto& tracking = section(root, "tracking");
    keys(tracking, "tracking", {"backend", "iou_threshold", "max_missed_frames", "low_confidence", "new_track_confidence", "mahalanobis_gate", "gating_mode"});
    const auto tracker = string(tracking, "tracking", "backend");
    if (tracker == "iou") settings.video.tracker_mode = TrackerMode::IoU;
    else if (tracker == "two-stage") settings.video.tracker_mode = TrackerMode::TwoStage;
    else if (tracker == "kalman") settings.video.tracker_mode = TrackerMode::Kalman;
    else if (tracker == "kalman-reid") {
        settings.video.tracker_mode = TrackerMode::Kalman;
        settings.video.use_appearance = true;
        settings.video.kalman_gate_mode = KalmanGateMode::CenterOnly;
    }
    else throw std::invalid_argument("tracking.backend must be iou, two-stage, kalman or kalman-reid");
    if (root.contains("appearance") && !settings.video.use_appearance)
        throw std::invalid_argument("[appearance] requires tracking.backend=kalman-reid");
    if (settings.video.tracker_mode == TrackerMode::Kalman) {
        if (settings.mode == ApplicationMode::Stream)
            throw std::invalid_argument("Kalman requires consecutive decoded frames; timestamp-aware stream tracking is not implemented");
        if (settings.detector.max_detections > KalmanTracker::max_detections)
            throw std::invalid_argument("Kalman detector.max_detections exceeds bounded input capacity");
        if (tracking.contains("gating_mode")) {
            const auto mode = string(tracking, "tracking", "gating_mode");
            if (mode == "center") settings.video.kalman_gate_mode = KalmanGateMode::CenterOnly;
            else if (mode == "full-box") settings.video.kalman_gate_mode = KalmanGateMode::FullBox;
            else
                throw std::invalid_argument("tracking.gating_mode must be full-box or center");
        }
        const double default_gate = settings.video.kalman_gate_mode == KalmanGateMode::CenterOnly
            ? kalman_center_gate99 : kalman_box_gate99;
        settings.video.kalman_gating_threshold = real(tracking, "tracking", "mahalanobis_gate", default_gate, 0.000001, 100);
    } else if (tracking.contains("mahalanobis_gate") || tracking.contains("gating_mode")) {
        throw std::invalid_argument("mahalanobis_gate/gating_mode require tracking.backend=kalman");
    }
    if (settings.video.use_appearance) {
        const auto& appearance = section(root, "appearance");
        keys(appearance, "appearance", {"backend", "bundle", "max_cosine_distance", "weight", "momentum"});
        if (string(appearance, "appearance", "backend") != "osnet_onnx")
            throw std::invalid_argument("appearance.backend must be osnet_onnx");
        settings.reid_bundle = model_path(appearance, "appearance", "bundle", directory, true);
        for (auto name : {"manifest.json", "model.onnx"})
            if (!std::filesystem::is_regular_file(settings.reid_bundle / name))
                throw std::invalid_argument("OSNet appearance bundle is missing " + std::string(name));
        settings.video.max_cosine_distance = real(appearance, "appearance", "max_cosine_distance", 0.20, 0, 1);
        settings.video.appearance_weight = real(appearance, "appearance", "weight", 0.50, 0, 1);
        settings.video.appearance_momentum = real(appearance, "appearance", "momentum", 0.90, 0, 1);
        if (settings.video.appearance_momentum >= 1.0)
            throw std::invalid_argument("appearance.momentum must be in [0,1)");
    }
    settings.video.tracking_iou = static_cast<float>(real(tracking, "tracking", "iou_threshold", 0.30, 0.000001, 1));
    settings.video.max_missed_frames = static_cast<std::uint32_t>(integer(tracking, "tracking", "max_missed_frames", 20, 0, 10000));
    settings.video.high_confidence = settings.detector.confidence_threshold;
    if (settings.video.tracker_mode != TrackerMode::IoU) {
        settings.video.low_confidence = static_cast<float>(real(tracking, "tracking", "low_confidence", 0.10, 0.000001, 1));
        settings.video.new_track_confidence = static_cast<float>(real(tracking, "tracking", "new_track_confidence", 0.50, 0.000001, 1));
        if (settings.video.low_confidence >= settings.video.high_confidence ||
            settings.video.high_confidence > settings.video.new_track_confidence)
            throw std::invalid_argument("Require tracking.low_confidence < detector.confidence_threshold <= tracking.new_track_confidence");
        settings.detector.confidence_threshold = settings.video.low_confidence;
    } else if (tracking.contains("low_confidence") || tracking.contains("new_track_confidence")) {
        throw std::invalid_argument("Low/new track confidence fields require tracking.backend=two-stage or kalman");
    }
    if (settings.mode == ApplicationMode::Stream) {
        const auto& stream = section(root, "stream");
        keys(stream, "stream", {"duration_seconds", "open_timeout_ms", "read_timeout_ms", "reconnect_initial_ms",
            "reconnect_max_ms", "max_outage_ms", "max_frame_age_ms", "tracking_gap_ms", "queue_capacity", "output_fps"});
        auto& c = settings.live;
        c.duration_seconds = static_cast<int>(integer(stream, "stream", "duration_seconds", 30, 1, 86400));
        c.open_timeout_ms = static_cast<int>(integer(stream, "stream", "open_timeout_ms", 8000, 1, 15000));
        c.read_timeout_ms = static_cast<int>(integer(stream, "stream", "read_timeout_ms", 2000, 1, 5000));
        c.reconnect_initial_ms = static_cast<int>(integer(stream, "stream", "reconnect_initial_ms", 250, 1, 5000));
        c.reconnect_max_ms = static_cast<int>(integer(stream, "stream", "reconnect_max_ms", 2000, 1, 5000));
        c.max_outage_ms = static_cast<int>(integer(stream, "stream", "max_outage_ms", 15000, 1, 600000));
        c.max_frame_age_ms = static_cast<int>(integer(stream, "stream", "max_frame_age_ms", 1000, 1, 10000));
        c.tracking_gap_ms = static_cast<int>(integer(stream, "stream", "tracking_gap_ms", 1000, 1, 10000));
        c.queue_capacity = static_cast<int>(integer(stream, "stream", "queue_capacity", 1, 1, 16));
        c.output_fps = real(stream, "stream", "output_fps", 10, 1, 120);
        validate_live_config(c);
        return settings;
    }
    const auto& video = section(root, "video", false);
    keys(video, "video", {"max_frames", "fallback_fps"});
    settings.video.max_frames = static_cast<int>(integer(video, "video", "max_frames", 0, 0, std::numeric_limits<int>::max()));
    settings.video.fallback_fps = real(video, "video", "fallback_fps", 25.0, 0.000001, 1000);
    return settings;
}

std::unique_ptr<IDetector> make_configured_detector(const ApplicationSettings& settings) {
    if (settings.mode == ApplicationMode::Search || settings.detector_model.empty())
        throw std::invalid_argument("Configured detector requires image or video mode");
    return std::make_unique<YoloDetector>(settings.detector_model, settings.detector);
}
} // namespace aegisvision::vision
