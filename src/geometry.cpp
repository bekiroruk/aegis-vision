#include "aegisvision/geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace aegisvision {

std::pair<double, double> transform_point(
    const std::pair<double, double> point, const Homography& homography) {
    const auto [x, y] = point;
    const auto denominator =
        homography[2][0] * x + homography[2][1] * y + homography[2][2];
    if (std::abs(denominator) < 1e-12) {
        throw std::invalid_argument("Point maps to infinity");
    }
    return {
        (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denominator,
        (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denominator,
    };
}

}  // namespace aegisvision

