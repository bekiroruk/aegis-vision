#include "aegisvision/vision.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace aegisvision::vision {
namespace {
void require_image(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("Expected a nonempty 8-bit BGR image");
    }
}

cv::Rect clipped_rect(const BoundingBox& box, const cv::Size size) {
    for (const float value : {box.x1, box.y1, box.x2, box.y2}) {
        if (!std::isfinite(value)) throw std::invalid_argument("Box coordinates must be finite");
    }
    if (box.x2 <= box.x1 || box.y2 <= box.y1) {
        throw std::invalid_argument("Box must have positive width and height");
    }
    const auto clip = [](float value, int maximum) {
        return std::clamp(static_cast<double>(value), 0.0, static_cast<double>(maximum));
    };
    const int left = static_cast<int>(std::floor(clip(box.x1, size.width)));
    const int top = static_cast<int>(std::floor(clip(box.y1, size.height)));
    const int right = static_cast<int>(std::ceil(clip(box.x2, size.width)));
    const int bottom = static_cast<int>(std::ceil(clip(box.y2, size.height)));
    if (right <= left || bottom <= top) {
        throw std::invalid_argument("Box does not intersect image with positive area");
    }
    return {left, top, right - left, bottom - top};
}
}  // namespace

cv::Mat load_image(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open input image");
    const auto length = input.tellg();
    if (length <= 0 || length > 128 * 1024 * 1024) {
        throw std::runtime_error("Encoded image must be between 1 byte and 128 MiB");
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("Cannot read input image");
    auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    require_image(image);
    return image;
}

void save_image(const std::filesystem::path& path, const cv::Mat& image) {
    require_image(image);
    std::vector<unsigned char> bytes;
    if (!cv::imencode(path.extension().string(), image, bytes)) {
        throw std::runtime_error("Cannot encode output image");
    }
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Cannot write output image");
}

cv::Mat crop(const cv::Mat& image, const BoundingBox& box) {
    require_image(image);
    return image(clipped_rect(box, image.size())).clone();
}

cv::Mat annotate(const cv::Mat& image, const std::vector<Detection>& detections) {
    require_image(image);
    auto output = image.clone();
    for (const auto& detection : detections) {
        if (!std::isfinite(detection.score) || detection.score < 0 || detection.score > 1) {
            throw std::invalid_argument("Detection score must be finite and in [0, 1]");
        }
        const auto rect = clipped_rect(detection.bbox, image.size());
        cv::rectangle(output, rect, {70, 220, 110}, 2);
        std::ostringstream label;
        label << detection.label << ' ' << std::fixed << std::setprecision(2) << detection.score;
        cv::putText(output, label.str(), {rect.x, std::max(15, rect.y - 5)},
            cv::FONT_HERSHEY_SIMPLEX, 0.5, {70, 220, 110}, 1, cv::LINE_AA);
    }
    return output;
}

AlignmentResult align(const cv::Mat& source, const cv::Mat& target, const AlignmentConfig& config) {
    require_image(source);
    require_image(target);
    if (config.max_features < 4 || !std::isfinite(config.ratio_threshold) ||
        config.ratio_threshold <= 0 || config.ratio_threshold >= 1 ||
        !std::isfinite(config.ransac_threshold_px) || config.ransac_threshold_px <= 0 ||
        config.min_inliers < 4 || !std::isfinite(config.min_inlier_ratio) ||
        config.min_inlier_ratio <= 0 || config.min_inlier_ratio > 1) {
        throw std::invalid_argument("Invalid alignment configuration");
    }
    cv::Mat gray_source, gray_target, descriptors_source, descriptors_target;
    cv::cvtColor(source, gray_source, cv::COLOR_BGR2GRAY);
    cv::cvtColor(target, gray_target, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> points_source, points_target;
    auto orb = cv::ORB::create(config.max_features);
    orb->detectAndCompute(gray_source, cv::noArray(), points_source, descriptors_source);
    orb->detectAndCompute(gray_target, cv::noArray(), points_target, descriptors_target);
    if (descriptors_source.rows < 2 || descriptors_target.rows < 2) {
        throw std::runtime_error("Not enough visual features to align images");
    }
    std::vector<std::vector<cv::DMatch>> pairs;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(descriptors_source, descriptors_target, pairs, 2);
    std::vector<cv::DMatch> candidates;
    for (const auto& pair : pairs) {
        if (pair.size() == 2 && pair[0].distance < config.ratio_threshold * pair[1].distance) {
            candidates.push_back(pair[0]);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance;
    });
    std::unordered_set<int> used_targets;
    std::vector<cv::DMatch> unique_matches;
    std::vector<cv::Point2f> from, to;
    for (const auto& match : candidates) {
        if (used_targets.insert(match.trainIdx).second) {
            unique_matches.push_back(match);
            from.push_back(points_source[match.queryIdx].pt);
            to.push_back(points_target[match.trainIdx].pt);
        }
    }
    if (from.size() < static_cast<std::size_t>(config.min_inliers)) {
        throw std::runtime_error("Not enough unambiguous feature matches");
    }
    cv::Mat mask;
    auto homography = cv::findHomography(from, to, cv::RANSAC, config.ransac_threshold_px, mask);
    if (homography.empty() || !cv::checkRange(homography) ||
        std::abs(cv::determinant(homography)) < 1e-12) {
        throw std::runtime_error("Could not estimate a finite, invertible homography");
    }
    AlignmentResult result;
    result.candidate_matches = static_cast<int>(from.size());
    result.inliers = cv::countNonZero(mask);
    result.inlier_ratio = static_cast<double>(result.inliers) / result.candidate_matches;
    if (result.inliers < config.min_inliers || result.inlier_ratio < config.min_inlier_ratio) {
        throw std::runtime_error("Alignment rejected: insufficient RANSAC support");
    }
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(from, projected, homography);
    double squared_error = 0;
    std::vector<cv::DMatch> inlier_matches;
    for (std::size_t i = 0; i < from.size(); ++i) {
        if (mask.at<unsigned char>(static_cast<int>(i))) {
            const auto delta = projected[i] - to[i];
            squared_error += delta.dot(delta);
            inlier_matches.push_back(unique_matches[i]);
        }
    }
    result.reprojection_rmse_px = std::sqrt(squared_error / result.inliers);
    if (!std::isfinite(result.reprojection_rmse_px) ||
        result.reprojection_rmse_px > config.ransac_threshold_px) {
        throw std::runtime_error("Alignment rejected: high reprojection error");
    }
    result.source_to_target = homography;
    cv::warpPerspective(source, result.aligned, homography, target.size());
    cv::addWeighted(result.aligned, 0.5, target, 0.5, 0, result.overlay);
    if (inlier_matches.size() > 80) inlier_matches.resize(80);
    cv::drawMatches(source, points_source, target, points_target, inlier_matches, result.matches,
        {70, 220, 110}, cv::Scalar::all(-1), {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return result;
}

void save_alignment(const std::filesystem::path& directory, const AlignmentResult& result) {
    save_image(directory / "aligned.png", result.aligned);
    save_image(directory / "overlay.png", result.overlay);
    save_image(directory / "matches.png", result.matches);
    cv::FileStorage report("report.yml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
    report << "direction" << "source_to_target" << "homography" << result.source_to_target;
    report << "candidate_matches" << result.candidate_matches << "inliers" << result.inliers;
    report << "inlier_ratio" << result.inlier_ratio << "reprojection_rmse_px" << result.reprojection_rmse_px;
    std::ofstream output(directory / "report.yml");
    output << report.releaseAndGetString();
    if (!output) throw std::runtime_error("Cannot write alignment report");
}
}  // namespace aegisvision::vision
