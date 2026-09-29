#pragma once

#include <array>
#include <utility>

namespace aegisvision {

using Homography = std::array<std::array<double, 3>, 3>;

[[nodiscard]] std::pair<double, double> transform_point(
    std::pair<double, double> point, const Homography& homography);

}  // namespace aegisvision

