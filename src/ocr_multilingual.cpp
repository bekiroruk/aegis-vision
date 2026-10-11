#include "aegisvision/ocr_multilingual.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <picosha2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
constexpr auto detector_name="text_detection_en_ppocrv3_2023may.onnx";
constexpr auto detector_hash="03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587";
constexpr auto recognizer_hash="9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6";
constexpr auto dictionary_hash="66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1";
std::vector<unsigned char> verified_file(const std::filesystem::path& path,const char* hash) {
    if(!std::filesystem::is_regular_file(path)) throw std::invalid_argument("Missing multilingual OCR bundle file");
    const auto size=std::filesystem::file_size(path);
    if(!size||size>32*1024*1024) throw std::invalid_argument("Invalid multilingual OCR bundle file size");
    std::ifstream file(path,std::ios::binary);std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if(!file.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(size))||file.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("Cannot read OCR bundle file completely");
    if(picosha2::hash256_hex_string(bytes)!=hash) throw std::invalid_argument("Multilingual OCR SHA256 mismatch");
    return bytes;
}
std::filesystem::path verified_detector(const std::filesystem::path& bundle) {
    (void)verified_file(bundle/detector_name,detector_hash);return bundle/detector_name;
}
std::vector<std::string> dictionary(const std::filesystem::path& bundle) {
    const auto bytes=verified_file(bundle/"inference.yml",dictionary_hash);
    std::istringstream input(std::string(bytes.begin(),bytes.end()));std::string line;bool started=false;
    std::vector<std::string> result;
    // This is a reader for the exact hash-pinned scalar-list artifact, NOT a general YAML parser.
    while(std::getline(input,line)) {
        if(!line.empty()&&line.back()=='\r') line.pop_back();
        if(!started) {if(line=="  character_dict:") started=true;continue;}
        if(!line.starts_with("  - ")) throw std::invalid_argument("Unexpected pinned dictionary format");
        auto symbol=line.substr(4);
        if(symbol.size()>=2&&symbol.front()=='\''&&symbol.back()=='\'') {
            symbol=symbol.substr(1,symbol.size()-2);
            std::size_t pos=0;while((pos=symbol.find("''",pos))!=std::string::npos) {symbol.erase(pos,1);++pos;}
        }
        if(utf8_codepoints(symbol).size()!=1) throw std::invalid_argument("Dictionary entry must be one Unicode scalar");
        result.push_back(std::move(symbol));
    }
    if(!started||result.empty()||result.size()>10000) throw std::invalid_argument("Invalid dictionary size");
    result.push_back(" "); // PaddleX CTCLabelDecode: blank=0, dictionary, final space.
    return result;
}
Ort::SessionOptions options() {
    Ort::SessionOptions opts;opts.SetIntraOpNumThreads(1);opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);return opts;
}
}
RecognizedText decode_unicode_ctc(std::span<const float> probabilities,std::size_t steps,const std::vector<std::string>& dict) {
    if(!steps||steps>800||dict.empty()||dict.size()>10000||probabilities.size()!=steps*(dict.size()+1))
        throw std::invalid_argument("Invalid multilingual CTC shape");
    for(const auto& symbol:dict) if(utf8_codepoints(symbol).size()!=1) throw std::invalid_argument("Invalid CTC dictionary symbol");
    const auto classes=dict.size()+1;RecognizedText result;std::size_t previous=0,count=0;double confidence=0;
    for(std::size_t t=0;t<steps;++t) {
        const float* row=probabilities.data()+t*classes;double sum=0;
        for(std::size_t c=0;c<classes;++c) {if(!std::isfinite(row[c])||row[c]<0||row[c]>1) throw std::runtime_error("Invalid CTC probability");sum+=row[c];}
        if(std::abs(sum-1)>.005) throw std::runtime_error("CTC output is not normalized probabilities");
        const auto best=static_cast<std::size_t>(std::max_element(row,row+classes)-row);
        if(best&&best!=previous) {result.text+=dict[best-1];confidence+=row[best];++count;}
        previous=best;
    }
    if(result.text.size()>1024) throw std::runtime_error("Multilingual text exceeds 1024 bytes");
    if(count) result.confidence=static_cast<float>(confidence/count);
    return result;
}
struct PpocrMultilingual::Impl {
    OcrConfig config;PpocrDetector detector;std::vector<std::string> dict;
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"aegis-ocr"};
    Ort::Session session{nullptr};std::string input_name,output_name;
    std::string id=std::string("ppocr3-det-ppocr6-tiny-rec-ort-cpu-v1:")+detector_hash+":"+recognizer_hash+":"+dictionary_hash;
    Impl(const std::filesystem::path& bundle,OcrConfig settings):config(settings),detector(verified_detector(bundle),settings),dict(dictionary(bundle)) {
        const auto bytes=verified_file(bundle/"inference.onnx",recognizer_hash);
        session=Ort::Session(environment,bytes.data(),bytes.size(),options());
        if(session.GetInputCount()!=1||session.GetOutputCount()!=1) throw std::runtime_error("Unexpected OCR model IO count");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name=session.GetInputNameAllocated(0,allocator).get();output_name=session.GetOutputNameAllocated(0,allocator).get();
        const auto input_type=session.GetInputTypeInfo(0),output_type=session.GetOutputTypeInfo(0);
        const auto input=input_type.GetTensorTypeAndShapeInfo(),output=output_type.GetTensorTypeAndShapeInfo();
        const auto shape=input.GetShape(),out=output.GetShape();
        if(input.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.size()!=4||shape[1]!=3||shape[2]!=48||
           output.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||out.size()!=3||out[2]!=static_cast<int64_t>(dict.size()+1))
            throw std::runtime_error("Unexpected OCR model shape or dictionary class count");
        id+=":"+std::to_string(config.max_regions)+":"+std::to_string(config.timeout_ms);
    }
};
PpocrMultilingual::PpocrMultilingual(const std::filesystem::path& bundle,OcrConfig config):impl_(std::make_unique<Impl>(bundle,config)) {}
PpocrMultilingual::~PpocrMultilingual()=default;
const std::string& PpocrMultilingual::signature() const {return impl_->id;}
std::vector<TextRegion> PpocrMultilingual::read(const cv::Mat& bgr,const std::atomic_bool* cancel) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(impl_->config.timeout_ms);
    const auto checkpoint=[&] {if(cancel&&cancel->load()) throw std::runtime_error("OCR cancelled");
        if(std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("OCR deadline exceeded");};
    checkpoint();auto regions=impl_->detector.detect(bgr,cancel);checkpoint();
    for(auto& region:regions) {
        checkpoint();const auto& p=region.polygon;
        const double width=std::max(cv::norm(p[0]-p[3]),cv::norm(p[1]-p[2]));
        const double height=std::max(cv::norm(p[0]-p[1]),cv::norm(p[2]-p[3]));
        const int resized_width=std::clamp(static_cast<int>(std::ceil(48*width/std::max(1.0,height))),2,3200);
        const int padded_width=std::max(320,resized_width);
        const TextQuad target={cv::Point2f{0,47},{0,0},{float(resized_width-1),0},{float(resized_width-1),47}};
        cv::Mat crop;cv::warpPerspective(bgr,crop,cv::getPerspectiveTransform(p.data(),target.data()),{resized_width,48});
        crop.convertTo(crop,CV_32FC3,1.0/127.5,-1);
        cv::Mat padded=cv::Mat::zeros(48,padded_width,CV_32FC3);crop.copyTo(padded(cv::Rect(0,0,resized_width,48)));
        auto blob=cv::dnn::blobFromImage(padded);const std::array<int64_t,4> shape{1,3,48,padded_width};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor=Ort::Value::CreateTensor<float>(memory,blob.ptr<float>(),blob.total(),shape.data(),shape.size());
        const char* inputs[]={impl_->input_name.c_str()};const char* outputs[]={impl_->output_name.c_str()};
        auto result=impl_->session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);checkpoint();
        const auto info=result[0].GetTensorTypeAndShapeInfo();const auto dims=info.GetShape();
        if(dims.size()!=3||dims[0]!=1||dims[1]<1||dims[1]>800||dims[2]!=static_cast<int64_t>(impl_->dict.size()+1))
            throw std::runtime_error("Unexpected multilingual OCR output shape");
        region.recognition=decode_unicode_ctc({result[0].GetTensorData<float>(),info.GetElementCount()},static_cast<std::size_t>(dims[1]),impl_->dict);
        checkpoint();
    }
    checkpoint();return regions;
}
}
