#include "aegisvision/ocr_evaluation.hpp"
#include "aegisvision/assignment.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
void validate(const std::vector<OcrLabel>& labels,bool reference) {
    if(labels.size()>64) throw std::invalid_argument("Evaluation supports at most 64 regions per image");
    for(const auto& label:labels) {
        const auto& b=label.box;
        if(!std::isfinite(b.x)||!std::isfinite(b.y)||!std::isfinite(b.width)||!std::isfinite(b.height)||
           b.x<0||b.y<0||b.width<=0||b.height<=0||b.x+b.width>1920||b.y+b.height>1920)
            throw std::invalid_argument("Invalid evaluation box");
        if(label.text.size()>256||(reference&&label.text.empty())) throw std::invalid_argument("Invalid evaluation text length");
        for(char c:label.text) if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')))
            throw std::invalid_argument("Evaluation labels require lowercase a-z/0-9; unsupported characters must not be dropped");
    }
}
double iou(const cv::Rect2f& a,const cv::Rect2f& b) {
    const double intersection=static_cast<double>(std::max(0.F,std::min(a.x+a.width,b.x+b.width)-std::max(a.x,b.x)))*
        std::max(0.F,std::min(a.y+a.height,b.y+b.height)-std::max(a.y,b.y));
    return intersection/(static_cast<double>(a.width)*a.height+static_cast<double>(b.width)*b.height-intersection);
}
}
OcrEvaluation& OcrEvaluation::operator+=(const OcrEvaluation& b) {
    references+=b.references;predictions+=b.predictions;matches+=b.matches;exact_matches+=b.exact_matches;
    reference_characters+=b.reference_characters;matched_characters+=b.matched_characters;
    matched_edits+=b.matched_edits;spatial_edits+=b.spatial_edits;return *this;
}
OcrEvaluation evaluate_ocr(const std::vector<OcrLabel>& reference,const std::vector<OcrLabel>& prediction,double minimum_iou) {
    if(!std::isfinite(minimum_iou)||minimum_iou<=0||minimum_iou>1) throw std::invalid_argument("IoU must be in (0,1]");
    validate(reference,true);validate(prediction,false);
    OcrEvaluation result;result.references=reference.size();result.predictions=prediction.size();
    std::vector<std::vector<double>> weights(reference.size(),std::vector<double>(prediction.size(),-1));
    const double bonus=static_cast<double>(std::min(reference.size(),prediction.size())+1);
    for(std::size_t r=0;r<reference.size();++r) {
        result.reference_characters+=reference[r].text.size();
        for(std::size_t p=0;p<prediction.size();++p) {
            const auto overlap=iou(reference[r].box,prediction[p].box);
            if(overlap>=minimum_iou) weights[r][p]=(bonus+overlap)/(bonus+1);
        }
    }
    std::vector<bool> used_reference(reference.size()),used_prediction(prediction.size());
    for(const auto& [r,p]:assign_max_weight(weights)) {
        used_reference[r]=true;used_prediction[p]=true;++result.matches;
        if(reference[r].text==prediction[p].text) ++result.exact_matches;
        result.matched_characters+=reference[r].text.size();
        result.matched_edits+=ascii_edit_distance(reference[r].text,prediction[p].text);
    }
    result.spatial_edits=result.matched_edits;
    for(std::size_t r=0;r<reference.size();++r) if(!used_reference[r]) result.spatial_edits+=reference[r].text.size();
    for(std::size_t p=0;p<prediction.size();++p) if(!used_prediction[p]) result.spatial_edits+=prediction[p].text.size();
    return result;
}
} // namespace aegisvision::vision
