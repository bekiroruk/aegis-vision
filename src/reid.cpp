#include "aegisvision/reid.hpp"
#include "aegisvision/vision.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <picosha2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace aegisvision {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
constexpr std::size_t max_model_bytes = 32ULL * 1024 * 1024;
constexpr std::size_t max_frame_bytes = 256ULL * 1024 * 1024;
constexpr int height = 256, width = 128, dimension = 512;
constexpr float mean[]{.485F, .456F, .406F};
constexpr float standard_deviation[]{.229F, .224F, .225F};

std::vector<unsigned char> bounded_file(const fs::path& path, std::size_t maximum) {
    if (!fs::is_regular_file(path)) throw std::invalid_argument("Re-ID requires a regular bundle file");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot read Re-ID bundle file");
    const auto length = input.tellg();
    if (length <= 0 || length > static_cast<std::streamoff>(maximum))
        throw std::invalid_argument("Re-ID bundle file exceeds its size limit");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Re-ID bundle file changed or could not be read completely");
    return bytes;
}

void exact_keys(const Json& object, const std::set<std::string>& expected) {
    if (!object.is_object() || object.size() != expected.size())
        throw std::invalid_argument("Unsupported Re-ID manifest fields");
    for (auto it = object.begin(); it != object.end(); ++it)
        if (!expected.contains(it.key())) throw std::invalid_argument("Unknown Re-ID manifest field");
}

bool hash_string(const Json& value) {
    if (!value.is_string()) return false;
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool integral_shape(const Json& value, const Json& expected) {
    return value.is_array() && value == expected &&
           std::all_of(value.begin(), value.end(), [](const auto& item) { return item.is_number_integer(); });
}

bool channel_values(const Json& value, const float* expected) {
    if (!value.is_array() || value.size() != 3) return false;
    for (std::size_t i = 0; i < 3; ++i) {
        if (!value[i].is_number()) return false;
        const double actual = value[i].get<double>();
        if (!std::isfinite(actual) || std::abs(actual - expected[i]) > 1e-7) return false;
    }
    return true;
}

Json manifest(const std::vector<unsigned char>& bytes) {
    std::vector<std::set<std::string>> keys;
    const auto callback = [&](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        else if (event == Json::parse_event_t::key) {
            if (!keys.back().insert(parsed.get<std::string>()).second)
                throw std::invalid_argument("Duplicate Re-ID manifest field");
        } else if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    auto result = Json::parse(bytes.begin(), bytes.end(), callback);
    exact_keys(result, {"version", "architecture", "training_dataset", "model_file", "model_sha256",
                        "model_size_bytes", "input_name", "input_shape", "output_name", "output_shape",
                        "dimension", "preprocessing", "mean", "std", "normalize_output", "source", "license"});
    if (!result.at("version").is_number_integer() || result.at("version") != 1 ||
        result.at("architecture") != "osnet_x0_25" || result.at("training_dataset") != "MSMT17-combineall" ||
        result.at("model_file") != "model.onnx" || !hash_string(result.at("model_sha256")) ||
        !result.at("model_size_bytes").is_number_unsigned() ||
        result.at("model_size_bytes").get<std::uint64_t>() == 0 ||
        result.at("model_size_bytes").get<std::uint64_t>() > max_model_bytes ||
        result.at("input_name") != "images" || !integral_shape(result.at("input_shape"), {1, 3, height, width}) ||
        result.at("output_name") != "features" || !integral_shape(result.at("output_shape"), {1, dimension}) ||
        !result.at("dimension").is_number_integer() || result.at("dimension") != dimension ||
        result.at("preprocessing") != "opencv-linear-rgb-imagenet-v1" ||
        !channel_values(result.at("mean"), mean) || !channel_values(result.at("std"), standard_deviation) ||
        !result.at("normalize_output").is_boolean() || result.at("normalize_output") != true ||
        result.at("license") != "MIT")
        throw std::invalid_argument("Unsupported Re-ID model/preprocessing contract");
    const auto& source = result.at("source");
    exact_keys(source, {"hf_repository", "hf_revision", "checkpoint_filename", "checkpoint_sha256",
                        "checkpoint_size_bytes", "architecture_revision", "source_sha256"});
    if (source.at("hf_repository") != "kaiyangzhou/osnet" ||
        source.at("hf_revision") != "a5c5cc037c24235cda3b21085b93ad77c9616224" ||
        source.at("checkpoint_sha256") != "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18" ||
        source.at("checkpoint_filename") != "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth" ||
        !source.at("checkpoint_size_bytes").is_number_integer() || source.at("checkpoint_size_bytes") != 9'336'983 ||
        source.at("architecture_revision") != "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50" ||
        source.at("source_sha256") != "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae")
        throw std::invalid_argument("Unsupported Re-ID checkpoint provenance");
    return result;
}

void output_contract(const cv::Mat& output) {
    if (output.type() != CV_32F || output.dims != 2 || output.rows != 1 || output.cols != dimension ||
        !output.isContinuous() || !cv::checkRange(output))
        throw std::runtime_error("Re-ID output must be finite float32 [1,512]");
}
}  // namespace

cv::Mat prepare_reid(const cv::Mat& cropped) {
    if (cropped.empty() || cropped.dims != 2 || cropped.type() != CV_8UC3 ||
        cropped.total() > max_frame_bytes / 3)
        throw std::invalid_argument("Re-ID requires a bounded nonempty 8-bit BGR crop");
    cv::Mat resized;
    cv::resize(cropped, resized, {width, height}, 0, 0, cv::INTER_LINEAR);
    const int shape[]{1, 3, height, width};
    cv::Mat result(4, shape, CV_32F);
    auto* data = result.ptr<float>();
    for (int y = 0; y < height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                data[channel * height * width + y * width + x] =
                    (row[x][2 - channel] / 255.0F - mean[channel]) / standard_deviation[channel];
            }
        }
    }
    return result;
}

cv::Mat prepare_reid(const Frame& frame, const Detection& detection) {
    if (detection.label != "person" || !std::isfinite(detection.score) ||
        detection.score < 0 || detection.score > 1)
        throw std::invalid_argument("Re-ID requires a valid person detection");
    if (!frame.image) throw std::invalid_argument("Re-ID requires owned image pixels");
    const auto& image = *frame.image;
    if (image.width <= 0 || image.height <= 0 ||
        static_cast<std::size_t>(image.width) > std::numeric_limits<std::size_t>::max() / 3 ||
        image.stride < static_cast<std::size_t>(image.width) * 3 ||
        image.stride > max_frame_bytes / static_cast<std::size_t>(image.height) ||
        image.pixels.size() < image.stride * static_cast<std::size_t>(image.height))
        throw std::invalid_argument("Invalid or oversized owned Re-ID BGR image buffer");
    // OpenCV's view is read-only here; crop returns an owned clone before resize.
    const cv::Mat view(image.height, image.width, CV_8UC3,
                       const_cast<std::uint8_t*>(image.pixels.data()), image.stride);
    return prepare_reid(vision::crop(view, detection.bbox));
}

std::vector<float> normalize_reid(std::vector<float> embedding) {
    if (embedding.size() != dimension) throw std::invalid_argument("Re-ID embedding must have 512 components");
    double norm = 0;
    for (const float value : embedding) {
        if (!std::isfinite(value)) throw std::invalid_argument("Re-ID embedding contains non-finite values");
        norm += static_cast<double>(value) * value;
    }
    if (!std::isfinite(norm) || norm <= 0) throw std::invalid_argument("Re-ID embedding has zero/invalid norm");
    const double length = std::sqrt(norm);
    for (float& value : embedding) value = static_cast<float>(value / length);
    return embedding;
}

struct ReIdEmbedder::Impl {
    cv::dnn::Net network;
    std::string space;
    explicit Impl(const fs::path& bundle) {
        if (!fs::is_directory(bundle)) throw std::invalid_argument("Missing Re-ID bundle directory");
        const auto root = fs::canonical(bundle);
        const auto manifest_path = root / "manifest.json", model_path = root / "model.onnx";
        // Fixed filenames and resolved paths forbid model redirects outside the bundle.
        for (const auto& path : {manifest_path, model_path})
            if (fs::canonical(path).parent_path() != root)
                throw std::invalid_argument("Re-ID bundle file escapes its directory");
        const auto settings = manifest(bounded_file(manifest_path, 64 * 1024));
        const auto model = bounded_file(model_path, max_model_bytes);
        if (model.size() != settings.at("model_size_bytes").get<std::uint64_t>())
            throw std::invalid_argument("Re-ID model size differs from manifest");
        if (picosha2::hash256_hex_string(model) != settings.at("model_sha256").get<std::string>())
            throw std::invalid_argument("Re-ID model SHA256 differs from manifest");
        const Json identity = {{"model_sha256", settings.at("model_sha256")},
                               {"architecture", settings.at("architecture")}, {"dimension", dimension},
                               {"preprocessing", settings.at("preprocessing")}, {"normalize_output", true}};
        space = "reid:" + picosha2::hash256_hex_string(identity.dump());
        network = cv::dnn::readNetFromONNX(model);
        network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        if (network.empty() || network.getUnconnectedOutLayersNames() != std::vector<std::string>{"features"})
            throw std::invalid_argument("Unsupported Re-ID graph outputs");
        const int shape[]{1, 3, height, width};
        network.setInput(cv::Mat::zeros(4, shape, CV_32F), "images");
        output_contract(network.forward("features"));
    }
};

ReIdEmbedder::ReIdEmbedder(const fs::path& bundle) : impl_(std::make_unique<Impl>(bundle)) {}
ReIdEmbedder::~ReIdEmbedder() = default;
const std::string& ReIdEmbedder::space_id() const noexcept { return impl_->space; }
std::vector<float> ReIdEmbedder::embed_text(std::string_view) {
    throw std::invalid_argument("Person Re-ID has no text embedding space");
}
std::vector<float> ReIdEmbedder::embed_image(const Frame& frame, const Detection& detection) {
    const auto input = prepare_reid(frame, detection);
    impl_->network.setInput(input, "images");
    const auto output = impl_->network.forward("features");
    output_contract(output);
    const auto* data = output.ptr<float>();
    return normalize_reid({data, data + dimension});
}

}  // namespace aegisvision
