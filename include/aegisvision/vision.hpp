#pragma once

#include "aegisvision/domain.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>

namespace aegisvision::vision {

// All image functions accept nonempty 8-bit BGR images. Coordinates are pixels.
// Filesystem paths support native Windows Unicode paths through binary streams.
[[nodiscard]] cv::Mat load_image(const std::filesystem::path& path);
void save_image(const std::filesystem::path& path, const cv::Mat& image);
[[nodiscard]] cv::Mat crop(const cv::Mat& image, const BoundingBox& box);
[[nodiscard]] cv::Mat annotate(
    const cv::Mat& image, const std::vector<Detection>& detections);

struct AlignmentConfig {
    int max_features{3000};
    float ratio_threshold{0.75F};
    double ransac_threshold_px{3.0};
    int min_inliers{12};
    double min_inlier_ratio{0.35};
};

struct AlignmentResult {
    cv::Mat source_to_target;  // 3x3 CV_64F; maps source pixel coordinates to target.
    cv::Mat aligned;
    cv::Mat overlay;
    cv::Mat matches;
    int candidate_matches{};
    int inliers{};
    double inlier_ratio{};
    double reprojection_rmse_px{};
};

// Intended for planar scenes / camera rotation, not arbitrary 3D scene alignment.
// Throws on insufficient features or unreliable geometry; never returns identity as fallback.
[[nodiscard]] AlignmentResult align(
    const cv::Mat& source, const cv::Mat& target, const AlignmentConfig& config = {});
void save_alignment(const std::filesystem::path& directory, const AlignmentResult& result);

}  // namespace aegisvision::vision
