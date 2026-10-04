#include "aegisvision/reid.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <picosha2.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;
using Json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message, const char* expected = "") {
    try { function(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "Input rejected for an unexpected reason");
        return;
    }
    throw std::runtime_error(message);
}
void save(const fs::path& path, const Json& value) {
    std::ofstream output(path);
    output << value.dump(2);
    require(static_cast<bool>(output), "Cannot write test manifest");
}
Json load(const fs::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "Cannot read test JSON");
    return Json::parse(input);
}
std::string file_hash(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Cannot read golden fixture file");
    return picosha2::hash256_hex_string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
Detection person(BoundingBox box = {0, 0, 2, 2}) {
    return {box, "person", .9F, {}, {}};
}
void preprocessing() {
    const cv::Mat color(1, 1, CV_8UC3, cv::Scalar(10, 64, 255));
    const auto blob = prepare_reid(color);
    require(blob.type() == CV_32F && blob.dims == 4 && blob.size[0] == 1 && blob.size[1] == 3 &&
                blob.size[2] == 256 && blob.size[3] == 128 && blob.isContinuous(),
            "Re-ID blob is not contiguous float32 NCHW [1,3,256,128]");
    const float expected[]{(1.0F - .485F) / .229F,
                           (64.0F / 255.0F - .456F) / .224F,
                           (10.0F / 255.0F - .406F) / .225F};
    for (int c = 0; c < 3; ++c)
        for (int pixel = 0; pixel < 256 * 128; ++pixel)
            require(std::abs(blob.ptr<float>()[c * 256 * 128 + pixel] - expected[c]) < 1e-6,
                    "RGB order, ImageNet normalization or NCHW channel plane is wrong");
    cv::Mat patterned(3, 5, CV_8UC3);
    for (int y = 0; y < patterned.rows; ++y)
        for (int x = 0; x < patterned.cols; ++x)
            patterned.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<unsigned char>(x * 30),
                static_cast<unsigned char>(y * 70), static_cast<unsigned char>((x + y) * 35));
    cv::Mat linear;
    cv::resize(patterned, linear, {128, 256}, 0, 0, cv::INTER_LINEAR);
    const auto actual = prepare_reid(patterned);
    const float mean[]{.485F, .456F, .406F}, stddev[]{.229F, .224F, .225F};
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 128; ++x)
            for (int c = 0; c < 3; ++c)
                require(std::abs(actual.ptr<float>()[c * 256 * 128 + y * 128 + x] -
                    (linear.at<cv::Vec3b>(y, x)[2 - c] / 255.0F - mean[c]) / stddev[c]) < 1e-6,
                    "Linear spatial resize differs from its reference");
    const auto region = patterned(cv::Rect(1, 0, 3, 3));
    require(!region.isContinuous(), "Test crop accidentally contiguous");
    const auto strided = prepare_reid(region), copied = prepare_reid(region.clone());
    require(cv::norm(strided, copied, cv::NORM_INF) == 0, "Non-contiguous Mat preprocessing changed pixels");
    rejects([] { (void)prepare_reid(cv::Mat{}); }, "Empty crop accepted");
    rejects([] { (void)prepare_reid(cv::Mat(1, 1, CV_32FC3)); }, "Floating crop accepted");
    rejects([] { (void)prepare_reid(cv::Mat(1, 1, CV_8UC1)); }, "Grayscale crop accepted");
}
void frame_validation() {
    auto buffer = std::make_shared<ImageBuffer>();
    buffer->width = 2; buffer->height = 2; buffer->stride = 8;
    buffer->pixels = {10, 64, 255, 10, 64, 255, 99, 99, 10, 64, 255, 10, 64, 255, 88, 88};
    const Frame frame{"fixture", "local", 0, {}, buffer};
    const auto whole = prepare_reid(frame, person());
    require(cv::norm(whole, prepare_reid(cv::Mat(1, 1, CV_8UC3, cv::Scalar(10, 64, 255))),
                     cv::NORM_INF) == 0, "Owned padded row stride was ignored");
    require(cv::norm(whole, prepare_reid(frame, person({-1, -1, 3, 3})), cv::NORM_INF) == 0,
            "Clipping outside edges changed a uniform crop");
    require(cv::norm(whole, prepare_reid(frame, person({.0001F, .0001F, .0002F, .0002F})),
                     cv::NORM_INF) == 0, "Tiny intersecting box did not get a positive pixel crop");
    auto invalid = person(); invalid.label = "car";
    rejects([&] { (void)prepare_reid(frame, invalid); }, "Non-person label accepted");
    invalid = person(); invalid.score = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)prepare_reid(frame, invalid); }, "NaN score accepted");
    invalid = person(); invalid.bbox.x1 = std::numeric_limits<float>::infinity();
    rejects([&] { (void)prepare_reid(frame, invalid); }, "Infinite crop coordinate accepted");
    invalid = person(); invalid.bbox.x2 = invalid.bbox.x1;
    rejects([&] { (void)prepare_reid(frame, invalid); }, "Zero-width crop accepted");
    rejects([&] { (void)prepare_reid(frame, person({3, 3, 4, 4})); }, "Non-intersecting box accepted");
    rejects([] { (void)prepare_reid(Frame{}, person()); }, "Missing pixels accepted");
    for (int field = 0; field < 6; ++field) {
        auto malformed = std::make_shared<ImageBuffer>(*buffer);
        switch (field) {
        case 0: malformed->width = 0; break;
        case 1: malformed->height = -1; break;
        case 2: malformed->stride = 5; break;
        case 3: malformed->pixels.resize(15); break;
        case 4: malformed->stride = std::numeric_limits<std::size_t>::max(); break;
        default: malformed->height = std::numeric_limits<int>::max(); break;
        }
        Frame bad = frame; bad.image = malformed;
        rejects([&] { (void)prepare_reid(bad, person()); }, "Malformed image buffer accepted");
    }
}
void normalization() {
    std::vector<float> vector(512); vector[0] = 3; vector[1] = 4;
    const auto normalized = normalize_reid(vector);
    require(std::abs(normalized[0] - .6F) < 1e-6 && std::abs(normalized[1] - .8F) < 1e-6,
            "Re-ID L2 normalization failed");
    require(vector[0] == 3 && vector[1] == 4, "Normalization mutated its input copy");
    rejects([] { (void)normalize_reid({3, 4}); }, "Wrong embedding dimension accepted");
    rejects([] { (void)normalize_reid(std::vector<float>(512)); }, "Zero embedding accepted");
    vector[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)normalize_reid(vector); }, "NaN embedding accepted");
    vector[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)normalize_reid(vector); }, "Infinite embedding accepted");
    vector.assign(512, std::numeric_limits<float>::max());
    const auto large = normalize_reid(vector);
    require(std::isfinite(large[0]) && std::abs(large[0] - 1.0 / std::sqrt(512.0)) < 1e-6,
            "Large finite embeddings overflowed normalization");
}
Json fixture_manifest() {
    const std::vector<unsigned char> fake{1, 2, 3};
    return {{"version", 1}, {"architecture", "osnet_x0_25"}, {"training_dataset", "MSMT17-combineall"},
            {"model_file", "model.onnx"}, {"model_sha256", picosha2::hash256_hex_string(fake)},
            {"model_size_bytes", 3}, {"input_name", "images"}, {"input_shape", {1, 3, 256, 128}},
            {"output_name", "features"}, {"output_shape", {1, 512}}, {"dimension", 512},
            {"preprocessing", "opencv-linear-rgb-imagenet-v1"}, {"mean", {.485, .456, .406}},
            {"std", {.229, .224, .225}}, {"normalize_output", true}, {"license", "MIT"},
            {"source", {{"hf_repository", "kaiyangzhou/osnet"},
                        {"hf_revision", "a5c5cc037c24235cda3b21085b93ad77c9616224"},
                        {"checkpoint_filename", "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"},
                        {"checkpoint_sha256", "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"},
                        {"checkpoint_size_bytes", 9'336'983},
                        {"architecture_revision", "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"},
                        {"source_sha256", "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae"}}}};
}
void bundle_validation() {
    const auto root = fs::temp_directory_path() / ("aegis-reid-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        rejects([&] { ReIdEmbedder bad(root / "absent"); }, "Missing bundle accepted");
        std::ofstream model(root / "model.onnx", std::ios::binary);
        model.write("\1\2\3", 3); model.close();
        const auto valid = fixture_manifest();
        save(root / "manifest.json", valid);
        // Correct metadata and checksum reach graph parsing; random bytes cannot be a model.
        rejects([&] { ReIdEmbedder bad(root); }, "Invalid ONNX accepted");
        auto bad = valid;
        bad["model_sha256"] = std::string(64, '0'); save(root / "manifest.json", bad);
        rejects([&] { ReIdEmbedder rejected(root); }, "Model hash mismatch accepted", "SHA256");
        bad = valid; bad["model_size_bytes"] = 4; save(root / "manifest.json", bad);
        rejects([&] { ReIdEmbedder rejected(root); }, "Model size mismatch accepted", "size");
        for (int field = 0; field < 14; ++field) {
            bad = valid;
            switch (field) {
            case 0: bad["version"] = 2; break;
            case 1: bad["version"] = 1.0; break;
            case 2: bad["architecture"] = "other"; break;
            case 3: bad["model_file"] = "../outside.onnx"; break;
            case 4: bad["input_shape"] = {1, 3, 128, 256}; break;
            case 5: bad["output_name"] = "embedding"; break;
            case 6: bad["mean"][0] = .1; break;
            case 7: bad["normalize_output"] = false; break;
            case 8: bad["source"]["hf_revision"] = "main"; break;
            case 9: bad["extra"] = true; break;
            case 10: bad["model_sha256"] = "invalid"; break;
            case 11: bad["source"]["source_sha256"] = std::string(64, 'a'); break;
            case 12: bad["source"]["checkpoint_filename"] = "other.pth"; break;
            default: bad.erase("license"); break;
            }
            save(root / "manifest.json", bad);
            rejects([&] { ReIdEmbedder rejected(root); }, "Unsupported model contract accepted", "Re-ID");
        }
        std::ofstream duplicate(root / "manifest.json");
        auto text = valid.dump(); text.insert(1, "\"version\":1,");
        duplicate << text; duplicate.close();
        rejects([&] { ReIdEmbedder rejected(root); }, "Duplicate manifest key accepted", "Duplicate");
        std::ofstream huge_manifest(root / "manifest.json");
        huge_manifest << std::string(64 * 1024 + 1, ' '); huge_manifest.close();
        rejects([&] { ReIdEmbedder rejected(root); }, "Oversized manifest accepted", "size limit");
        save(root / "manifest.json", valid);
        std::ofstream huge_model(root / "model.onnx", std::ios::binary);
        huge_model.seekp(32LL * 1024 * 1024); huge_model.put('\0'); huge_model.close();
        rejects([&] { ReIdEmbedder rejected(root); }, "Oversized model accepted", "size limit");
        fs::remove_all(root);
    } catch (...) {
        fs::remove_all(root);
        throw;
    }
}
void reference(const fs::path& bundle) {
    require(std::endian::native == std::endian::little, "Golden input tensors use float32 little endian");
    const auto fixtures = load(bundle / "reference.json");
    require(fixtures.at("version") == 1 && fixtures.at("fixtures").is_array() &&
                !fixtures.at("fixtures").empty(), "Invalid OSNet reference fixtures");
    ReIdEmbedder encoder(bundle);
    ReIdEmbedder second(bundle);
    require(encoder.space_id() == second.space_id() && encoder.space_id().starts_with("reid:"),
            "Re-ID model space is unstable");
    double minimum_cosine = 1, maximum_error = 0, maximum_blob_error = 0;
    for (const auto& fixture : fixtures.at("fixtures")) {
        const auto image_file = bundle / fixture.at("image_file").get<std::string>();
        const auto blob_file = bundle / fixture.at("input_blob_file").get<std::string>();
        require(file_hash(image_file) == fixture.at("image_sha256").get<std::string>() &&
                    file_hash(blob_file) == fixture.at("input_blob_sha256").get<std::string>(),
                "Golden input fixture checksum differs");
        const auto image = vision::load_image(image_file);
        const auto frame = vision::image_frame(image, "reference", "local");
        const auto observation = person({0, 0, static_cast<float>(image.cols), static_cast<float>(image.rows)});
        const auto actual = encoder.embed_image(frame, observation);
        const auto blob = prepare_reid(frame, observation);
        std::ifstream input(blob_file, std::ios::binary | std::ios::ate);
        require(input && input.tellg() == static_cast<std::streamoff>(blob.total() * sizeof(float)),
                "Golden input tensor length differs");
        std::vector<float> golden(blob.total()); input.seekg(0);
        input.read(reinterpret_cast<char*>(golden.data()), static_cast<std::streamsize>(golden.size() * sizeof(float)));
        require(static_cast<bool>(input), "Cannot read golden input tensor");
        for (std::size_t i = 0; i < golden.size(); ++i) {
            require(std::isfinite(golden[i]), "Golden tensor is non-finite");
            maximum_blob_error = std::max(maximum_blob_error,
                std::abs(static_cast<double>(golden[i]) - blob.ptr<float>()[i]));
        }
        for (const char* name : {"pytorch_embedding", "opencv_embedding"}) {
            const auto expected = normalize_reid(fixture.at(name).get<std::vector<float>>());
            double cosine = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                cosine += static_cast<double>(actual[i]) * expected[i];
                maximum_error = std::max(maximum_error,
                    std::abs(static_cast<double>(actual[i]) - expected[i]));
            }
            minimum_cosine = std::min(minimum_cosine, cosine);
        }
        const auto repeated = encoder.embed_image(frame, observation);
        require(std::equal(actual.begin(), actual.end(), repeated.begin()), "Re-ID inference changed for identical input");
        rejects([&] { (void)encoder.embed_text("a person"); }, "Re-ID accepted text query");
        auto wrong_class = observation; wrong_class.label = "car";
        rejects([&] { (void)encoder.embed_image(frame, wrong_class); }, "Encoder accepted non-person crop");
    }
    require(maximum_blob_error < 1e-6, "C++ preprocessing differs from independent Python blob");
    require(minimum_cosine > .99999 && maximum_error < 1e-4,
            "C++ OSNet embeddings differ from independent references");
    std::cout << "OSNet reference fixtures=" << fixtures.at("fixtures").size()
              << " min_cosine=" << minimum_cosine << " max_embedding_error=" << maximum_error
              << " max_blob_error=" << maximum_blob_error << '\n';
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        cv::setNumThreads(1);
        preprocessing(); frame_validation(); normalization(); bundle_validation();
        if (argc == 3 && std::string(argv[1]) == "--bundle") reference(argv[2]);
        else if (argc != 1) throw std::invalid_argument("Usage: aegisvision_reid_tests [--bundle BUNDLE]");
        std::cout << "Re-ID preprocessing, normalization, malformed-input and bundle validation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
