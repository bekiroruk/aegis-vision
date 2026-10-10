#pragma once
#include "aegisvision/ocr.hpp"

namespace aegisvision::vision {
struct OcrLabel {
    cv::Rect2f box; // Axis-aligned continuous x,y,width,height; not polygon IoU.
    std::string text;
};
struct OcrEvaluation {
    std::size_t references{}, predictions{}, matches{}, exact_matches{};
    std::size_t reference_characters{}, matched_characters{}, matched_edits{}, spatial_edits{};
    OcrEvaluation& operator+=(const OcrEvaluation& other);
};
// Geometry-only maximum-cardinality assignment, then maximum IoU; no text-based matching.
// Unsupported alphabet/invalid boxes are rejected rather than silently normalized away.
OcrEvaluation evaluate_ocr(const std::vector<OcrLabel>& reference,
                           const std::vector<OcrLabel>& prediction, double minimum_iou=.5);
} // namespace aegisvision::vision
