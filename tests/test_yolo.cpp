#include "aegisvision/yolo.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/pipeline.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace aegisvision;
namespace v = aegisvision::vision;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message) {
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
void near(float a, float b) { require(std::abs(a - b) < 0.01F, "Unexpected mapped coordinate"); }

void unit_tests() {
    const cv::Mat image(360, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto input = v::prepare_yolo(image);
    require(input.pad_top == 140 && input.pad_left == 0, "Letterbox padding incorrect");
    require(input.blob.dims == 4 && input.blob.size[1] == 3 && input.blob.size[2] == 640,
        "Expected NCHW blob");
    near(input.blob.ptr<float>(0, 0, 200)[100], 30.0F / 255.0F);
    near(input.blob.ptr<float>(0, 2, 200)[100], 10.0F / 255.0F);
    near(input.blob.ptr<float>(0, 0, 0)[0], 114.0F / 255.0F);
    const auto portrait = v::prepare_yolo(cv::Mat(640, 360, CV_8UC3, cv::Scalar::all(0)));
    require(portrait.pad_left == 140 && portrait.pad_top == 0, "Portrait padding incorrect");
    const auto odd = v::prepare_yolo(cv::Mat(333, 777, CV_8UC3, cv::Scalar::all(0)));
    require(odd.scale_x != odd.scale_y, "Rounded resize scales not preserved");

    const int shape[] = {1, 6, 8};
    cv::Mat output(3, shape, CV_32F, cv::Scalar::all(0));
    auto set = [&](int anchor, float cx, float cy, float w, float h, float c0, float c1) {
        const float values[] = {cx, cy, w, h, c0, c1};
        for (int c = 0; c < 6; ++c) output.ptr<float>(0, c)[anchor] = values[c];
    };
    set(0, 100, 240, 100, 100, 0.9F, 0.1F);
    set(1, 101, 241, 100, 100, 0.8F, 0.1F); // Same class: suppress.
    set(2, 100, 240, 100, 100, 0.1F, 0.85F); // Other class: retain.
    set(3, 400, 400, 100, 100, 0.2F, 0.1F); // Low confidence.
    set(4, 20, 20, 10, 10, 0.99F, 0.1F); // Entirely in padding.
    set(5, 20, 160, 100, 100, 0.7F, 0.1F); // Clip to image.
    set(6, std::numeric_limits<float>::quiet_NaN(), 240, 50, 50, 0.99F, 0.1F);
    set(7, 300, 300, -10, 10, 0.99F, 0.1F);
    const std::vector<std::string> labels{"person", "bus"};
    v::YoloConfig config;
    const auto detections = v::decode_yolo(output, input, labels, config);
    require(detections.size() == 3, "Filtering or class-aware NMS failed");
    require(detections[0].label == "person" && detections[1].label == "bus", "Class mapping incorrect");
    near(detections[0].bbox.x1, 50); near(detections[0].bbox.y1, 50);
    near(detections[0].bbox.x2, 150); near(detections[0].bbox.y2, 150);
    near(detections[2].bbox.x1, 0); near(detections[2].bbox.y1, 0);
    const auto half_scale = v::prepare_yolo(cv::Mat(720, 1280, CV_8UC3, cv::Scalar::all(0)));
    const auto scaled = v::decode_yolo(output, half_scale, labels, config);
    near(scaled[0].bbox.x1, 100); near(scaled[0].bbox.y1, 100);
    near(scaled[0].bbox.x2, 300); near(scaled[0].bbox.y2, 300);
    cv::Mat zeros(3, shape, CV_32F, cv::Scalar::all(0));
    require(v::decode_yolo(zeros, input, labels, config).empty(), "Zero-confidence output not empty");
    rejects([&] { v::YoloDetector missing(std::filesystem::path{}); }, "Missing model accepted");
    config.max_detections = 1;
    require(v::decode_yolo(output, input, labels, config).size() == 1, "Detection limit ignored");
    config.confidence_threshold = 1;
    require(v::decode_yolo(output, input, labels, config).empty(), "Empty result handling failed");
    rejects([&] { (void)v::decode_yolo(output, input, {"one"}, {}); }, "Wrong class layout accepted");
    rejects([&] { (void)v::decode_yolo(cv::Mat::zeros(8, 6, CV_32F), input, labels, {}); },
        "Unsupported output layout accepted");
    rejects([&] { (void)v::prepare_yolo(image, 639); }, "Invalid input size accepted");
    config.confidence_threshold = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)v::decode_yolo(output, input, labels, config); }, "NaN threshold accepted");
    const auto frame = v::image_frame(image(cv::Rect(10, 10, 100, 100)), "test", "camera");
    require(frame.image->pixels.size() == 30000 && frame.image->stride == 300, "ROI copy size incorrect");
    require(frame.image->pixels[0] == 10, "Frame pixel ownership/copy failed");
    require(v::coco_labels().size() == 80 && v::coco_labels()[5] == "bus", "COCO classes incorrect");
    std::cout << "YOLO preprocessing, decoding, clipping, NMS and error tests passed\n";
}

void smoke_test(const std::filesystem::path& model, const std::filesystem::path& path) {
    const auto image = v::load_image(path);
    v::YoloDetector detector(model);
    PipelineConfig config;
    config.enable_tracking = false; config.enable_embeddings = false; config.index_embeddings = false;
    AnalysisPipeline pipeline(config, detector, nullptr, nullptr, nullptr, nullptr);
    const auto detections = pipeline.analyze(v::image_frame(image, "1", "photo")).detections;
    int people = 0, buses = 0;
    for (const auto& d : detections) {
        if (d.label == "person") ++people;
        if (d.label == "bus") ++buses;
        require(d.bbox.x1 >= 0 && d.bbox.y1 >= 0 && d.bbox.x2 <= image.cols &&
            d.bbox.y2 <= image.rows && d.bbox.area() > 0, "Inference coordinates outside original image");
    }
    require(people >= 3 && buses >= 1, "Bus fixture should contain at least 3 people and a bus");
    rejects([&] { (void)detector.detect(Frame{}); }, "Missing pixel buffer accepted");
    Frame invalid;
    invalid.image = std::make_shared<ImageBuffer>(ImageBuffer{640, 640, 1920, {1, 2, 3}});
    rejects([&] { (void)detector.detect(invalid); }, "Truncated pixel buffer accepted");
    std::cout << "Real model smoke test passed: people=" << people << " buses=" << buses << '\n';
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    try {
        unit_tests();
        if (argc == 3) smoke_test(argv[1], argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
