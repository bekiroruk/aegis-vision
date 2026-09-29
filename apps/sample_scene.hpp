#pragma once

#include <opencv2/imgproc.hpp>

// Reproducible geometric fixture, not an AI-generated photograph or model output.
inline cv::Mat sample_scene() {
    cv::Mat image(520, 760, CV_8UC3, cv::Scalar(26, 32, 42));
    cv::RNG random(20260930);
    for (int i = 0; i < 240; ++i) {
        const cv::Point center(random.uniform(30, 730), random.uniform(90, 490));
        const cv::Scalar color(random.uniform(60, 245), random.uniform(60, 245), random.uniform(60, 245));
        cv::circle(image, center, random.uniform(3, 13), color, i % 3 ? -1 : 2, cv::LINE_AA);
    }
    cv::putText(image, "AEGIS VISION / GEOMETRY LAB", {25, 40},
        cv::FONT_HERSHEY_SIMPLEX, 0.8, {230, 230, 230}, 2, cv::LINE_AA);
    cv::putText(image, "Synthetic test scene - known perspective transform", {25, 68},
        cv::FONT_HERSHEY_SIMPLEX, 0.45, {190, 190, 190}, 1, cv::LINE_AA);
    cv::rectangle(image, {70, 130, 190, 170}, {230, 190, 60}, 3);
    cv::putText(image, "A-17", {100, 220}, cv::FONT_HERSHEY_SIMPLEX, 1.4, {255, 255, 255}, 3);
    cv::rectangle(image, {430, 300, 210, 140}, {100, 220, 240}, 3);
    cv::putText(image, "B-42", {455, 385}, cv::FONT_HERSHEY_SIMPLEX, 1.4, {255, 255, 255}, 3);
    return image;
}

inline cv::Mat sample_transform() {
    return (cv::Mat_<double>(3, 3) <<
        0.94, -0.035, 35.0, 0.025, 0.93, 18.0, 0.000045, -0.000025, 1.0);
}
