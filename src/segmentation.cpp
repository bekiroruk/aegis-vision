#include "aegisvision/segmentation.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
void validate_mask(const InstanceMask& item,cv::Size size) {
    if (size.width<1 || size.height<1 || size.width>1920 || size.height>1920 ||
        item.mask.type()!=CV_8UC1 || item.mask.size()!=item.region.size() || item.region.empty() ||
        (item.region & cv::Rect(0,0,size.width,size.height))!=item.region)
        throw std::invalid_argument("Invalid ROI mask");
    for (int y=0;y<item.mask.rows;++y) for (int x=0;x<item.mask.cols;++x) {
        const auto value=item.mask.at<unsigned char>(y,x);
        if (value!=0 && value!=255) throw std::invalid_argument("Mask must contain only 0/255");
    }
}
}
std::vector<InstanceMask> SegmentationPipeline::analyze(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type()!=CV_8UC3 || bgr.cols>1920 || bgr.rows>1920 ||
        (!size_.empty() && bgr.size()!=size_)) throw std::invalid_argument("Invalid or changing pipeline frame size");
    auto instances=segmenter_.segment(bgr);
    if (instances.size()>100) throw std::invalid_argument("Too many segmentation instances");
    std::vector<Detection> detections;
    for (auto& item:instances) {
        validate_mask(item,bgr.size());
        const auto& d=item.detection;
        if (!std::isfinite(d.score) || d.score<0 || d.score>1 || d.label.empty() ||
            !std::isfinite(d.bbox.area()) || d.bbox.area()<=0)
            throw std::invalid_argument("Invalid segmentation detection");
        const auto& b=d.bbox;
        for (const float coordinate:{b.x1,b.y1,b.x2,b.y2})
            if (!std::isfinite(coordinate)) throw std::invalid_argument("Nonfinite segmentation box");
        if (b.x1<0 || b.y1<0 || b.x2>bgr.cols || b.y2>bgr.rows ||
            item.region!=cv::Rect(static_cast<int>(std::ceil(b.x1)),static_cast<int>(std::ceil(b.y1)),
                static_cast<int>(std::ceil(b.x2))-static_cast<int>(std::ceil(b.x1)),
                static_cast<int>(std::ceil(b.y2))-static_cast<int>(std::ceil(b.y1))))
            throw std::invalid_argument("Mask ROI does not match its detection box");
        item.track_id=0;detections.push_back(d);
    }
    if (track_) {
        const auto tracks=tracker_.update_indexed(detections);
        for (std::size_t i=0;i<instances.size();++i) instances[i].track_id=tracks[i].track_id;
    }
    size_=bgr.size();return instances;
}
std::vector<std::uint32_t> mask_rle(const InstanceMask& instance,cv::Size image_size) {
    validate_mask(instance,image_size);
    std::vector<std::uint32_t> counts;
    bool previous=false;std::uint32_t run=0;
    for (int x=0;x<image_size.width;++x) for (int y=0;y<image_size.height;++y) {
        const bool value=instance.region.contains(cv::Point(x,y)) &&
            instance.mask.at<unsigned char>(y-instance.region.y,x-instance.region.x)!=0;
        if (value!=previous) {counts.push_back(run);run=0;previous=value;}
        ++run;
    }
    counts.push_back(run);return counts;
}
std::vector<InstanceMask> decode_segmentation(const cv::Mat& predictions, const cv::Mat& prototypes,
    const Letterbox& transform, const std::vector<std::string>& labels, const YoloConfig& config) {
    const int channels = static_cast<int>(labels.size()) + 4;
    if (labels.empty() || labels.size() > 10000 || predictions.type() != CV_32F || predictions.dims != 3 ||
        predictions.size[0] != 1 || predictions.size[1] != channels + 32 ||
        predictions.size[2] < 1 || predictions.size[2] > 100000 || !predictions.isContinuous() ||
        prototypes.type() != CV_32F || prototypes.dims != 4 || prototypes.size[0] != 1 ||
        prototypes.size[1] != 32 || prototypes.size[2] != config.input_size/4 ||
        prototypes.size[3] != config.input_size/4 || !prototypes.isContinuous())
        throw std::invalid_argument("Expected static YOLOv8-seg predictions and 32 prototypes");
    if (transform.original_size.width < 1 || transform.original_size.height < 1 ||
        transform.original_size.width > 1920 || transform.original_size.height > 1920 ||
        config.max_detections > 100)
        throw std::invalid_argument("Segmentation limit: image sides <=1920, instances <=100");
    const int size[] = {1,channels,predictions.size[2]};
    cv::Mat boxes(3,size,CV_32F,const_cast<float*>(predictions.ptr<float>()));
    auto detections = decode_yolo(boxes,transform,labels,config,true);
    if (!cv::checkRange(prototypes)) throw std::invalid_argument("Nonfinite mask prototypes");
    const double content_width=transform.original_size.width*transform.scale_x;
    const double content_height=transform.original_size.height*transform.scale_y;
    if (!std::isfinite(content_width) || !std::isfinite(content_height) ||
        content_width<1 || content_height<1 || content_width>config.input_size || content_height>config.input_size ||
        transform.pad_left<0 || transform.pad_top<0 ||
        transform.pad_left+content_width>config.input_size || transform.pad_top+content_height>config.input_size)
        throw std::invalid_argument("Invalid segmentation content dimensions");
    const int side = prototypes.size[2];
    const int left = static_cast<int>(std::round(transform.pad_left / 4.0));
    const int top = static_cast<int>(std::round(transform.pad_top / 4.0));
    const int width = static_cast<int>(std::round(transform.original_size.width * transform.scale_x / 4.0));
    const int height = static_cast<int>(std::round(transform.original_size.height * transform.scale_y / 4.0));
    if (left < 0 || top < 0 || width < 1 || height < 1 || left+width > side || top+height > side)
        throw std::invalid_argument("Invalid segmentation letterbox crop");
    const cv::Mat bases(32,side*side,CV_32F,const_cast<float*>(prototypes.ptr<float>()));
    std::vector<InstanceMask> result;
    for (auto& detection : detections) {
        const int anchor = std::stoi(detection.attributes.at("anchor_index"));
        detection.attributes.erase("anchor_index");
        cv::Mat coefficients(1,32,CV_32F);
        for (int c=0;c<32;++c) coefficients.at<float>(0,c) = predictions.ptr<float>(0,channels+c)[anchor];
        if (!cv::checkRange(coefficients)) throw std::invalid_argument("Nonfinite mask coefficients");
        cv::Mat logits = coefficients * bases;
        if (!cv::checkRange(logits)) throw std::invalid_argument("Nonfinite mask logits");
        cv::Mat full;
        cv::resize(logits.reshape(1,side)(cv::Rect(left,top,width,height)),full,
            transform.original_size,0,0,cv::INTER_LINEAR);
        const auto& b = detection.bbox;
        // A pixel belongs to the box iff its integer coordinate is in [x1,x2).
        const int x1=static_cast<int>(std::ceil(b.x1)), y1=static_cast<int>(std::ceil(b.y1));
        const int x2=static_cast<int>(std::ceil(b.x2)), y2=static_cast<int>(std::ceil(b.y2));
        const cv::Rect region(x1,y1,x2-x1,y2-y1);
        if (region.empty()) continue;
        cv::Mat binary;
        cv::compare(full(region),0,binary,cv::CMP_GT); // sigmoid(logit) > .5
        result.push_back({std::move(detection),region,std::move(binary)});
    }
    return result;
}
cv::Mat paint_masks(const cv::Mat& bgr,const std::vector<InstanceMask>& instances) {
    if (bgr.empty() || bgr.type()!=CV_8UC3) throw std::invalid_argument("Mask painting needs BGR image");
    auto output=bgr.clone();
    for (std::size_t i=0;i<instances.size();++i) {
        const auto& item=instances[i];
        if (item.mask.type()!=CV_8UC1 || item.mask.size()!=item.region.size() || item.region.empty() ||
            (item.region & cv::Rect(0,0,bgr.cols,bgr.rows))!=item.region)
            throw std::invalid_argument("Invalid ROI mask");
        const auto color_id=item.track_id ? item.track_id : i;
        const cv::Scalar color(static_cast<double>(60+(color_id*73)%180),
            static_cast<double>(70+(color_id*131)%170),static_cast<double>(70+(color_id*47)%170));
        auto roi=output(item.region);
        cv::Mat blended;
        cv::addWeighted(roi,.55,cv::Mat(roi.size(),CV_8UC3,color),.45,0,blended);
        blended.copyTo(roi,item.mask);
        cv::rectangle(output,item.region,color,1);
        const auto label=item.detection.label+(item.track_id ? " #"+std::to_string(item.track_id) : "");
        cv::putText(output,label, {item.region.x,std::max(14,item.region.y-3)},
            cv::FONT_HERSHEY_SIMPLEX,.45,color,1,cv::LINE_AA);
    }
    return output;
}
YoloSegmenter::YoloSegmenter(const std::filesystem::path& model,YoloConfig config):config_(config) {
    if (config_.max_detections<1 || config_.max_detections>100 ||
        !std::isfinite(config_.confidence_threshold) || config_.confidence_threshold<=0 || config_.confidence_threshold>1 ||
        !std::isfinite(config_.nms_iou_threshold) || config_.nms_iou_threshold<=0 || config_.nms_iou_threshold>1)
        throw std::invalid_argument("Invalid segmentation configuration");
    (void)prepare_yolo(cv::Mat(32,32,CV_8UC3,cv::Scalar::all(0)),config_.input_size);
    std::ifstream file(model,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()<=0 || file.tellg()>256LL*1024*1024) throw std::runtime_error("Invalid segmentation ONNX file");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Cannot read segmentation ONNX");
    net_=cv::dnn::readNetFromONNX(bytes);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}
std::vector<InstanceMask> YoloSegmenter::segment(const cv::Mat& bgr) {
    if (bgr.cols>1920 || bgr.rows>1920) throw std::invalid_argument("Segmentation image side exceeds 1920");
    const auto input=prepare_yolo(bgr,config_.input_size);
    net_.setInput(input.blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs,net_.getUnconnectedOutLayersNames());
    if (outputs.size()!=2) throw std::runtime_error("Segmentation requires two ONNX outputs");
    if (outputs[0].dims==4) std::swap(outputs[0],outputs[1]);
    return decode_segmentation(outputs[0],outputs[1],input,coco_labels(),config_);
}
} // namespace aegisvision::vision
