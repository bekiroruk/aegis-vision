#include "aegisvision/ocr.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
void validate_image(const cv::Mat& image) {
    if(image.empty() || image.dims!=2 || image.type()!=CV_8UC3 || image.cols>1920 || image.rows>1920)
        throw std::invalid_argument("OCR requires nonempty BGR8 image with sides <=1920");
}
bool valid_quad(const TextQuad& points,cv::Size size) {
    for(const auto& p:points)
        if(!std::isfinite(p.x) || !std::isfinite(p.y) || p.x<0 || p.y<0 || p.x>size.width-1 || p.y>size.height-1) return false;
    const std::vector<cv::Point2f> polygon(points.begin(),points.end());
    return cv::isContourConvex(polygon) && std::abs(cv::contourArea(polygon))>=1;
}
cv::dnn::Net load_net(const std::filesystem::path& path) {
    if(!std::filesystem::is_regular_file(path)) throw std::invalid_argument("OCR model must be a local file");
    const auto size=std::filesystem::file_size(path);
    if(size==0 || size>128*1024*1024) throw std::invalid_argument("OCR model exceeds 128 MiB or is empty");
    std::ifstream input(path,std::ios::binary);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if(!input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(size)))
        throw std::runtime_error("Cannot read OCR model");
    auto net=cv::dnn::readNetFromONNX(bytes);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
}
OcrConfig validate_config(OcrConfig config) {
    if(config.max_regions<1 || config.max_regions>100 || config.timeout_ms<1 || config.timeout_ms>60000)
        throw std::invalid_argument("OCR limits require 1..100 regions and 1..60000 ms");
    return config;
}
}
RecognizedText decode_english_ctc(const cv::Mat& logits) {
    if(logits.type()!=CV_32F || logits.dims!=3 || logits.size[0]<1 || logits.size[0]>256 ||
       logits.size[1]!=1 || logits.size[2]!=37 || !logits.isContinuous())
        throw std::invalid_argument("CRNN EN expects finite float logits [T,1,37], T<=256");
    constexpr char alphabet[]="0123456789abcdefghijklmnopqrstuvwxyz";
    RecognizedText result;int previous=-1;double confidence=0;
    for(int t=0;t<logits.size[0];++t) {
        const float* row=logits.ptr<float>(t);
        for(int c=0;c<37;++c) if(!std::isfinite(row[c])) throw std::invalid_argument("Nonfinite CRNN logit");
        const int best=static_cast<int>(std::max_element(row,row+37)-row);
        if(best!=0 && best!=previous) {
            double denominator=0;
            for(int c=0;c<37;++c) denominator+=std::exp(static_cast<double>(row[c])-row[best]);
            confidence+=1.0/denominator;result.text+=alphabet[best-1];
        }
        previous=best; // A blank separates repeated characters: a,blank,a => aa.
    }
    if(!result.text.empty()) result.confidence=static_cast<float>(confidence/result.text.size());
    return result;
}
cv::Mat rectify_text(const cv::Mat& bgr,const TextQuad& polygon) {
    validate_image(bgr);
    if(!valid_quad(polygon,bgr.size())) throw std::invalid_argument("OCR quadrilateral must be convex and inside image");
    const TextQuad destination={cv::Point2f{0,31},{0,0},{99,0},{99,31}};
    cv::Mat crop;
    cv::warpPerspective(bgr,crop,cv::getPerspectiveTransform(polygon.data(),destination.data()),{100,32});
    return crop;
}
std::size_t ascii_edit_distance(const std::string& reference,const std::string& prediction) {
    for(const auto* text:{&reference,&prediction}) {
        if(text->size()>1024) throw std::invalid_argument("OCR reference exceeds 1024 bytes");
        for(unsigned char c:*text) if(c>127) throw std::invalid_argument("ASCII metric does not accept Unicode");
    }
    std::vector<std::size_t> row(prediction.size()+1);std::iota(row.begin(),row.end(),0);
    for(std::size_t i=0;i<reference.size();++i) {
        auto diagonal=row[0];row[0]=i+1;
        for(std::size_t j=0;j<prediction.size();++j) {
            const auto above=row[j+1];
            row[j+1]=std::min({above+1,row[j]+1,diagonal+(reference[i]!=prediction[j])});diagonal=above;
        }
    }
    return row.back();
}
PpocrCrnn::PpocrCrnn(const std::filesystem::path& detector,const std::filesystem::path& recognizer,OcrConfig config)
    :config_(validate_config(config)),detector_(load_net(detector)),recognizer_(load_net(recognizer)) {
    // OpenCV Zoo ppocr_det.py preprocessing, pinned by prepare_ocr.ps1.
    // Normalize explicitly: older OpenCV 4.x only exposes a scalar model scale.
    detector_.setInputParams(1.0,{736,736},cv::Scalar(),false,false);
    detector_.setBinaryThreshold(.3F);detector_.setPolygonThreshold(.5F);
    detector_.setUnclipRatio(2.0);detector_.setMaxCandidates(200);
}
std::vector<TextRegion> PpocrCrnn::read(const cv::Mat& bgr,const std::atomic_bool* cancel) {
    validate_image(bgr);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(config_.timeout_ms);
    const auto checkpoint=[&] {
        if(cancel && cancel->load()) throw std::runtime_error("OCR cancelled");
        if(std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("OCR deadline exceeded");
    };
    checkpoint();
    cv::Mat resized;cv::resize(bgr,resized,{736,736});resized.convertTo(resized,CV_32F);
    std::vector<cv::Mat> channels;cv::split(resized,channels);
    constexpr double means[]={123.675,116.28,103.53}, deviations[]={.229,.224,.225};
    for(int c=0;c<3;++c) channels[c]=(channels[c]-means[c])/(255.0*deviations[c]);
    cv::merge(channels,resized);
    std::vector<std::vector<cv::Point>> boxes;std::vector<float> scores;
    detector_.detect(resized,boxes,scores);checkpoint();
    if(boxes.size()!=scores.size()) throw std::runtime_error("OCR detector result mismatch");
    std::vector<std::size_t> order(boxes.size());std::iota(order.begin(),order.end(),0);
    for(float s:scores) if(!std::isfinite(s) || s<0 || s>1) throw std::runtime_error("Invalid OCR detection score");
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[a]>scores[b];});
    std::vector<TextRegion> results;
    for(auto index:order) {
        checkpoint();if(results.size()>=config_.max_regions) break;
        if(boxes[index].size()!=4) throw std::runtime_error("OCR detector expected quadrilateral");
        TextQuad polygon;
        for(std::size_t i=0;i<4;++i) polygon[i]={
            std::clamp(boxes[index][i].x*bgr.cols/736.F,0.F,static_cast<float>(bgr.cols-1)),
            std::clamp(boxes[index][i].y*bgr.rows/736.F,0.F,static_cast<float>(bgr.rows-1))};
        if(!valid_quad(polygon,bgr.size())) continue; // Clipped degenerate detection, never warp it.
        auto crop=rectify_text(bgr,polygon);cv::cvtColor(crop,crop,cv::COLOR_BGR2GRAY);
        recognizer_.setInput(cv::dnn::blobFromImage(crop,1.0/127.5,{100,32},cv::Scalar::all(127.5)));
        auto recognition=decode_english_ctc(recognizer_.forward());checkpoint();
        results.push_back({polygon,scores[index],std::move(recognition)});
    }
    checkpoint();return results; // No partial result published after failure/cancellation.
}
} // namespace aegisvision::vision
