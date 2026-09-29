#include "aegisvision/domain.hpp"

#include <algorithm>
#include <stdexcept>

namespace aegisvision {

BoundingBox::BoundingBox(const float left, const float top, const float right, const float bottom)
    : x1(left), y1(top), x2(right), y2(bottom) {
    if (x2 < x1 || y2 < y1) {
        throw std::invalid_argument("Bounding box must satisfy x2 >= x1 and y2 >= y1");
    }
}

float BoundingBox::area() const noexcept {
    return std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
}

float BoundingBox::iou(const BoundingBox& other) const noexcept {
    const auto width = std::max(0.0F, std::min(x2, other.x2) - std::max(x1, other.x1));
    const auto height = std::max(0.0F, std::min(y2, other.y2) - std::max(y1, other.y1));
    const auto intersection = width * height;
    const auto union_area = area() + other.area() - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

}  // namespace aegisvision

