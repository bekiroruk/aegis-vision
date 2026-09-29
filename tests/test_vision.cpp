#include "aegisvision/vision.hpp"
#include "../apps/sample_scene.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message) {
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
}

int main() {
    namespace vision = aegisvision::vision;
    namespace fs = std::filesystem;
    const auto temporary = fs::temp_directory_path() / ("aegisvision-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(temporary);
    try {
        const auto image = sample_scene();
        const auto original = image.clone();
        const auto unicode_path = temporary / fs::path(u8"görüntü-ışık.png");
        vision::save_image(unicode_path, image);
        require(cv::norm(image, vision::load_image(unicode_path), cv::NORM_INF) == 0,
            "Lossless Unicode-path image roundtrip failed");

        auto cropped = vision::crop(image, {-10, -20, 30, 40});
        require(cropped.cols == 30 && cropped.rows == 40, "Clipped crop has wrong dimensions");
        cropped.setTo(cv::Scalar::all(0));
        require(cv::norm(image, original, cv::NORM_INF) == 0, "Crop aliases original pixels");
        const auto annotated = vision::annotate(image, {
            {{5, 10, 100, 120}, "test", 0.9F, {}, {}},
        });
        require(cv::norm(image, annotated, cv::NORM_INF) > 0, "Annotation did not draw");
        require(cv::norm(image, original, cv::NORM_INF) == 0, "Annotation mutated input");

        rejects([&] { (void)vision::load_image(temporary / "missing.png"); }, "Missing file accepted");
        { std::ofstream file(temporary / "corrupt.png"); file << "not a PNG"; }
        rejects([&] { (void)vision::load_image(temporary / "corrupt.png"); }, "Corrupt image accepted");
        rejects([&] { (void)vision::crop(image, {900, 900, 910, 910}); }, "Outside crop accepted");
        rejects([&] { (void)vision::crop(image, {1, 1, 1, 1}); }, "Zero-area crop accepted");
        rejects([&] { (void)vision::crop(image, {1.5F, 1.5F, 1.5F, 2.5F}); }, "Fractional zero-width crop accepted");
        rejects([&] { (void)vision::crop(image, {0, 0, std::numeric_limits<float>::infinity(), 20}); },
            "Nonfinite crop accepted");
        rejects([&] { (void)vision::annotate(image, {{{0, 0, 30, 30}, "bad", 2, {}, {}}}); },
            "Invalid confidence accepted");
        rejects([&] { (void)vision::align({}, image); }, "Empty image accepted");
        const cv::Mat blank(image.size(), CV_8UC3, cv::Scalar::all(0));
        rejects([&] { (void)vision::align(blank, blank); }, "Featureless images accepted");
        vision::AlignmentConfig invalid;
        invalid.ratio_threshold = 1.0F;
        rejects([&] { (void)vision::align(image, image, invalid); }, "Invalid config accepted");
        vision::AlignmentConfig impossible;
        impossible.min_inliers = 4000;
        rejects([&] { (void)vision::align(image, image, impossible); }, "Insufficient matches accepted");

        const auto known = sample_transform();
        cv::Mat target;
        cv::warpPerspective(image, target, known, image.size());
        const auto result = vision::align(image, target);
        require(result.inliers >= 50, "Too few inliers on known transform");
        require(result.reprojection_rmse_px < 2.0, "High reprojection error");
        const std::vector<cv::Point2f> corners{{60, 80}, {700, 80}, {700, 470}, {60, 470}};
        std::vector<cv::Point2f> expected, actual;
        cv::perspectiveTransform(corners, expected, known);
        cv::perspectiveTransform(corners, actual, result.source_to_target);
        for (std::size_t i = 0; i < corners.size(); ++i) {
            require(cv::norm(expected[i] - actual[i]) < 2.0, "Estimated homography has wrong direction or geometry");
        }
        require(result.aligned.size() == target.size(), "Warp output is not in target coordinates");
        vision::save_alignment(temporary / "result", result);
        require(!vision::load_image(temporary / "result" / "matches.png").empty(), "Match output not readable");
        require(fs::file_size(temporary / "result" / "report.yml") > 0, "Report missing");
        std::cout << "Image I/O, Unicode paths, crop, annotation, errors and known-transform alignment passed\n"
            << "inliers=" << result.inliers << " RMSE=" << result.reprojection_rmse_px << " px\n";
        fs::remove_all(temporary);  // Only this test's uniquely created scratch directory.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << " (fixtures retained at " << temporary << ")\n";
        return 1;
    }
}
