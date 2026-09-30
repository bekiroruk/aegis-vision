#include "aegisvision/clip.hpp"
#include "aegisvision/vision.hpp"
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace aegisvision {
std::vector<float> normalize_embedding(std::vector<float> vector) {
    double norm=0;
    for(float value : vector) {
        if(!std::isfinite(value)) throw std::invalid_argument("Embedding contains non-finite values");
        norm+=static_cast<double>(value)*value;
    }
    if(norm<=0 || !std::isfinite(norm)) throw std::invalid_argument("Embedding must be nonempty and nonzero");
    norm=std::sqrt(norm);
    for(float& value : vector) value=static_cast<float>(value/norm);
    return vector;
}
namespace {
void tensor_contract(const Ort::Session& session, std::size_t index, const char* name,
                     ONNXTensorElementDataType type, const std::vector<std::int64_t>& shape, bool output=false) {
    Ort::AllocatorWithDefaultOptions allocator;
    const auto actual_name=output ? session.GetOutputNameAllocated(index,allocator) : session.GetInputNameAllocated(index,allocator);
    const auto type_info=output ? session.GetOutputTypeInfo(index) : session.GetInputTypeInfo(index);
    const auto tensor=type_info.GetTensorTypeAndShapeInfo();
    if(std::string(actual_name.get())!=name || tensor.GetElementType()!=type || tensor.GetShape()!=shape)
        throw std::invalid_argument("Unsupported CLIP tensor contract: " + std::string(name));
}
std::vector<float> embedding(Ort::Value& value) {
    const auto info=value.GetTensorTypeAndShapeInfo();
    if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape()!=std::vector<std::int64_t>{1,512})
        throw std::runtime_error("CLIP output is not float32 [1,512]");
    const float* data=value.GetTensorData<float>();
    return normalize_embedding({data,data+512});
}
}
struct ClipEmbedder::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"aegis-clip"};
    Ort::SessionOptions options;
    Ort::Session vision{nullptr}, text{nullptr};
    ClipTokenizer tokenizer;
    std::string space;
    explicit Impl(const std::filesystem::path& bundle) : tokenizer(bundle/"tokenizer.json") {
        std::ifstream file(bundle/"manifest.json");
        if(!file) throw std::runtime_error("Missing CLIP manifest.json; run export_clip.py");
        const auto manifest=nlohmann::json::parse(file);
        if(manifest.at("model")!="openai/clip-vit-base-patch32" || manifest.at("dimension")!=512 || manifest.at("context_length")!=77 ||
           manifest.at("preprocessing")!="opencv-cubic-short224-floor-center-rgb-clip-v1")
            throw std::invalid_argument("Unsupported CLIP bundle configuration");
        for(const auto* name : {"vision.onnx","text.onnx","tokenizer.json"}) {
            std::ifstream data(bundle/name,std::ios::binary);
            if(!data) throw std::runtime_error("Missing CLIP bundle file: " + std::string(name));
            const auto actual=picosha2::hash256_hex_string(std::istreambuf_iterator<char>(data),std::istreambuf_iterator<char>());
            if(actual!=manifest.at("files_sha256").at(name).get<std::string>()) throw std::runtime_error("CLIP SHA256 mismatch: " + std::string(name));
        }
        space=picosha2::hash256_hex_string(manifest.at("files_sha256").dump()+manifest.at("preprocessing").get<std::string>());
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        vision=Ort::Session(env,(bundle/"vision.onnx").c_str(),options);
        text=Ort::Session(env,(bundle/"text.onnx").c_str(),options);
        if(vision.GetInputCount()!=1 || text.GetInputCount()!=2 || vision.GetOutputCount()!=1 || text.GetOutputCount()!=1)
            throw std::invalid_argument("Unsupported CLIP graph inputs/outputs");
        tensor_contract(vision,0,"pixel_values",ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,{1,3,224,224});
        tensor_contract(text,0,"input_ids",ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,{1,77});
        tensor_contract(text,1,"attention_mask",ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,{1,77});
        tensor_contract(vision,0,"embedding",ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,{1,512},true);
        tensor_contract(text,0,"embedding",ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,{1,512},true);
    }
};
ClipEmbedder::ClipEmbedder(const std::filesystem::path& bundle) : impl_(std::make_unique<Impl>(bundle)) {}
ClipEmbedder::~ClipEmbedder() = default;
const std::string& ClipEmbedder::space_id() const noexcept { return impl_->space; }
std::vector<float> ClipEmbedder::embed_text(std::string_view query) {
    auto tokens=impl_->tokenizer.encode(query);
    const std::array<std::int64_t,2> shape{1,77};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::array<Ort::Value,2> inputs{
        Ort::Value::CreateTensor<std::int64_t>(memory,tokens.ids.data(),77,shape.data(),2),
        Ort::Value::CreateTensor<std::int64_t>(memory,tokens.mask.data(),77,shape.data(),2)};
    const char* names[]{"input_ids","attention_mask"}; const char* output[]{"embedding"};
    auto result=impl_->text.Run(Ort::RunOptions{nullptr},names,inputs.data(),2,output,1);
    return embedding(result[0]);
}
std::vector<float> ClipEmbedder::embed_image(const Frame& frame,const Detection& detection) {
    if(!frame.image) throw std::invalid_argument("CLIP requires image pixels");
    const auto& image=*frame.image;
    if(image.width<=0 || image.height<=0 || image.stride<static_cast<std::size_t>(image.width)*3 ||
       image.stride>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(image.height) ||
       image.pixels.size()<image.stride*static_cast<std::size_t>(image.height))
        throw std::invalid_argument("Invalid BGR image buffer");
    const cv::Mat view(image.height,image.width,CV_8UC3,const_cast<std::uint8_t*>(image.pixels.data()),image.stride);
    const auto cropped=vision::crop(view,detection.bbox);
    const double ratio=224.0/std::min(cropped.cols,cropped.rows);
    if(ratio*std::max(cropped.cols,cropped.rows)>16384) throw std::invalid_argument("CLIP crop aspect ratio too large");
    cv::Mat resized;
    // Set the short side exactly: floating multiplication can yield 223.999999.
    const cv::Size size=cropped.cols<=cropped.rows
        ? cv::Size(224,static_cast<int>(224.0*cropped.rows/cropped.cols))
        : cv::Size(static_cast<int>(224.0*cropped.cols/cropped.rows),224);
    cv::resize(cropped,resized,size,0,0,cv::INTER_CUBIC);
    const auto center=resized(cv::Rect((resized.cols-224)/2,(resized.rows-224)/2,224,224));
    constexpr float mean[]{.48145466F,.4578275F,.40821073F}, stddev[]{.26862954F,.26130258F,.27577711F};
    std::vector<float> pixels(3*224*224);
    for(int y=0;y<224;++y) for(int x=0;x<224;++x) for(int c=0;c<3;++c)
        pixels[c*224*224+y*224+x]=(center.at<cv::Vec3b>(y,x)[2-c]/255.0F-mean[c])/stddev[c];
    const std::array<std::int64_t,4> shape{1,3,224,224};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(memory,pixels.data(),pixels.size(),shape.data(),shape.size());
    const char* name[]{"pixel_values"}; const char* output[]{"embedding"};
    auto result=impl_->vision.Run(Ort::RunOptions{nullptr},name,&input,1,output,1);
    return embedding(result[0]);
}
}
