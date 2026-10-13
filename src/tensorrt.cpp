#include "aegisvision/tensorrt.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <picosha2.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <exception>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

static_assert(NV_TENSORRT_MAJOR==10 && NV_TENSORRT_MINOR==13 && NV_TENSORRT_PATCH==3,
    "This backend is validated against TensorRT 10.13.3; revalidate before changing SDK");
namespace aegisvision::vision {
namespace {
using Json=nlohmann::json;
constexpr std::size_t input_bytes=1ULL*3*640*640*sizeof(float);
constexpr std::size_t output_bytes=1ULL*84*8400*sizeof(float);
void check(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(result));
}
class Logger final: public nvinfer1::ILogger {
    void log(Severity severity,const char* message) noexcept override {
        if(severity<=Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};
template<class T> using Trt=std::unique_ptr<T>;
std::vector<char> read(const std::filesystem::path& path,std::size_t limit) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()<=0 || static_cast<std::uint64_t>(file.tellg())>limit)
        throw std::invalid_argument("Missing, empty or oversized TensorRT input");
    std::vector<char> bytes(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);file.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    if(!file) throw std::runtime_error("TensorRT input read failed");
    return bytes;
}
std::string sha(const std::vector<char>& bytes) { return picosha2::hash256_hex_string(bytes); }
void write(const std::filesystem::path& path,const void* data,std::size_t size) {
    std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(data),static_cast<std::streamsize>(size));file.close();
    if(!file) throw std::runtime_error("TensorRT output write failed");
}
Json hardware() {
    check(cudaSetDevice(0));cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    int runtime{},driver{};check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
    return {{"gpu",prop.name},{"compute_major",prop.major},{"compute_minor",prop.minor},
        {"cuda_runtime",runtime},{"cuda_driver",driver},{"tensorrt",NV_TENSORRT_VERSION},
        {"device",0},{"gpu_total_bytes",prop.totalGlobalMem}};
}
bool shape(const nvinfer1::Dims& d,std::initializer_list<int> expected) {
    if(d.nbDims!=static_cast<int>(expected.size())) return false;
    int i=0;for(int x:expected) if(d.d[i++]!=x) return false;return true;
}
void validate_engine(const nvinfer1::ICudaEngine& engine) {
    if(engine.getNbIOTensors()!=2) throw std::invalid_argument("Expected two YOLO detection I/O tensors");
    bool input=false,output=false;
    for(int i=0;i<2;++i) {
        const auto name=engine.getIOTensorName(i);
        if(engine.getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
            engine.getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR ||
            engine.getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE)
            throw std::invalid_argument("YOLO engine requires linear device FP32 I/O");
        if(engine.getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT) {
            if(input || !shape(engine.getTensorShape(name),{1,3,640,640})) throw std::invalid_argument("Invalid static YOLO input");
            input=true;
        } else {
            if(output || !shape(engine.getTensorShape(name),{1,84,8400})) throw std::invalid_argument("Invalid static YOLO output");
            output=true;
        }
    }
    if(!input || !output) throw std::invalid_argument("YOLO input/output missing");
}
struct Buffer {
    void* ptr{};
    ~Buffer() { if(ptr) (void)cudaFree(ptr); }
    void allocate(std::size_t bytes) { check(cudaMalloc(&ptr,bytes)); }
};
struct Stream {
    cudaStream_t ptr{};
    ~Stream() { if(ptr) { (void)cudaStreamSynchronize(ptr);(void)cudaStreamDestroy(ptr); } }
};

// Legacy implicit calibration deliberately pinned to TensorRT 10.13.3.
// TensorRT 11 requires a separate explicit Q/DQ implementation.
class EntropyCalibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
    std::filesystem::path path_;
    std::vector<char> manifest_bytes_;
    Json manifest_;
    Buffer device_;
    std::size_t next_=0;
    std::exception_ptr error_;
    std::vector<char> cache_;
    std::filesystem::path image_path(const Json& entry) const {
        const std::filesystem::path relative(entry.at("path").get<std::string>());
        if(relative.is_absolute() || relative.has_root_path()) throw std::invalid_argument("Absolute calibration path");
        const auto base=std::filesystem::canonical(path_.parent_path());
        const auto target=std::filesystem::canonical(base/relative);
        const auto confined=target.lexically_relative(base);
        if(confined.empty()) throw std::invalid_argument("Invalid calibration path");
        for(const auto& part:confined) if(part=="..") throw std::invalid_argument("Calibration path escapes dataset");
        return target;
    }
    std::vector<char> image_bytes(const Json& entry) const {
        auto bytes=read(image_path(entry),5ULL*1024*1024);
        if(sha(bytes)!=entry.at("sha256").get<std::string>()) throw std::invalid_argument("Calibration image hash mismatch");
        return bytes;
    }
public:
    EntropyCalibrator(const std::filesystem::path& path,const std::string& model):
        path_(std::filesystem::absolute(path)),manifest_bytes_(read(path,256*1024)),
        manifest_(Json::parse(manifest_bytes_.begin(),manifest_bytes_.end())) {
        if(manifest_.at("version")!=1 || manifest_.at("kind")!="int8-calibration" ||
            manifest_.at("algorithm")!="entropy2" || manifest_.at("batch_size")!=1 ||
            manifest_.at("preprocessing")!="prepare_yolo-v1-640-rgb-nchw-0to1-pad114" ||
            manifest_.at("model_sha256")!=model || manifest_.at("images").size()!=128 ||
            manifest_.at("excluded_images").size()!=64)
            throw std::invalid_argument("Invalid calibration protocol/model");
        std::set<std::int64_t> ids;
        std::set<std::string> hashes,pixels;
        for(const auto& entry:manifest_.at("excluded_images")) {
            if(!ids.insert(entry.at("coco_image_id").get<std::int64_t>()).second ||
                !hashes.insert(entry.at("sha256").get<std::string>()).second)
                throw std::invalid_argument("Duplicate evaluation identity");
        }
        for(const auto& entry:manifest_.at("images")) {
            if(!ids.insert(entry.at("coco_image_id").get<std::int64_t>()).second ||
                !hashes.insert(entry.at("sha256").get<std::string>()).second ||
                !pixels.insert(entry.at("decoded_sha256").get<std::string>()).second)
                throw std::invalid_argument("Calibration duplicate/evaluation overlap");
            (void)image_bytes(entry);
        }
    }
    int32_t getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* bindings[],char const*[],int32_t count) noexcept override {
        try {
            if(error_) return false;
            if(count!=1 || !bindings) throw std::runtime_error("Invalid calibration bindings");
            if(next_==manifest_.at("images").size()) return false;
            const auto& entry=manifest_.at("images").at(next_);
            const auto bytes=image_bytes(entry);
            const cv::Mat image=cv::imdecode(bytes,cv::IMREAD_COLOR);
            if(image.empty() || image.cols>1920 || image.rows>1920 ||
                image.cols!=entry.at("width").get<int>() || image.rows!=entry.at("height").get<int>())
                throw std::invalid_argument("Invalid calibration image dimensions");
            const auto prepared=prepare_yolo(image,640);
            if(!device_.ptr) device_.allocate(input_bytes);
            check(cudaMemcpy(device_.ptr,prepared.blob.ptr<float>(),input_bytes,cudaMemcpyHostToDevice));
            bindings[0]=device_.ptr;++next_;
            if(next_%32==0) std::cerr << "INT8 calibration batches: " << next_ << "/128\n";
            return true;
        } catch(...) { error_=std::current_exception();return false; }
    }
    void const* readCalibrationCache(std::size_t& length) noexcept override {
        length=0;return nullptr; // Always calibrate; never silently reuse stale ranges.
    }
    void writeCalibrationCache(void const* data,std::size_t length) noexcept override {
        try {
            if(!data || !length || length>1024*1024) throw std::runtime_error("Invalid calibration cache");
            const auto* begin=static_cast<const char*>(data);cache_.assign(begin,begin+length);
        } catch(...) { error_=std::current_exception(); }
    }
    void verify() const {
        if(error_) std::rethrow_exception(error_);
        if(next_!=128 || cache_.empty()) throw std::runtime_error("Calibration incomplete");
        if(sha(read(path_,256*1024))!=sha(manifest_bytes_)) throw std::runtime_error("Calibration manifest changed");
        for(const auto& entry:manifest_.at("images")) (void)image_bytes(entry);
    }
    Json save(const std::filesystem::path& bundle) const {
        write(bundle/"calibration.cache",cache_.data(),cache_.size());
        write(bundle/"calibration-manifest.json",manifest_bytes_.data(),manifest_bytes_.size());
        return {{"algorithm","entropy2"},{"images",next_},{"batch_size",1},
            {"manifest_sha256",sha(manifest_bytes_)},{"cache_sha256",sha(cache_)},
            {"evaluation_manifest_sha256",manifest_.at("evaluation_manifest_sha256")}};
    }
};
}

void build_yolo_tensorrt(const std::filesystem::path& onnx,const std::filesystem::path& bundle,bool fp16,
    const std::filesystem::path& calibration_manifest) {
    if(!calibration_manifest.empty() && !fp16) throw std::invalid_argument("INT8 requires FP16 fallback");
    if(std::filesystem::exists(bundle)) throw std::invalid_argument("Engine output directory already exists");
    const auto bytes=read(onnx,256ULL*1024*1024);
    std::unique_ptr<EntropyCalibrator> calibrator;
    if(!calibration_manifest.empty())
        calibrator=std::make_unique<EntropyCalibrator>(calibration_manifest,sha(bytes));
    auto info=hardware();Logger logger;
    Trt<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if(!builder) throw std::runtime_error("TensorRT builder unavailable");
    Trt<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if(!network) throw std::runtime_error("TensorRT network allocation failed");
    Trt<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network,logger));
    if(!parser || !parser->parse(bytes.data(),bytes.size())) throw std::runtime_error("TensorRT ONNX parse failed");
    if(network->getNbInputs()!=1 || network->getNbOutputs()!=1 ||
        !shape(network->getInput(0)->getDimensions(),{1,3,640,640}) ||
        !shape(network->getOutput(0)->getDimensions(),{1,84,8400}) ||
        network->getInput(0)->getType()!=nvinfer1::DataType::kFLOAT || network->getOutput(0)->getType()!=nvinfer1::DataType::kFLOAT)
        throw std::invalid_argument("Only static FP32 I/O 80-class YOLOv8 detection ONNX is supported");
    Trt<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if(!config) throw std::runtime_error("TensorRT config allocation failed");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,512ULL*1024*1024);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    if(fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if(calibrator) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    if(calibrator) std::cerr << "Building INT8-mixed engine with fresh Entropy2 calibration; this can take several minutes.\n";
    const auto start=std::chrono::steady_clock::now();
    Trt<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));
    if(calibrator) calibrator->verify();
    if(!plan || plan->size()==0 || plan->size()>256ULL*1024*1024) throw std::runtime_error("TensorRT engine build failed");
    Trt<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if(!runtime) throw std::runtime_error("TensorRT runtime allocation failed");
    Trt<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(),plan->size()));
    if(!engine) throw std::runtime_error("Built engine cannot deserialize");
    validate_engine(*engine);
    if(sha(read(onnx,256ULL*1024*1024))!=sha(bytes)) throw std::runtime_error("ONNX changed during build");
    info["version"]=1;info["model_sha256"]=sha(bytes);
    info["engine_sha256"]=picosha2::hash256_hex_string(static_cast<const char*>(plan->data()),static_cast<const char*>(plan->data())+plan->size());
    info["precision"]=calibrator?"int8-mixed":fp16?"fp16-mixed":"fp32";info["tf32"]=false;
    info["input"]=Json::array({1,3,640,640});info["output"]=Json::array({1,84,8400});
    info["workspace_limit_bytes"]=512ULL*1024*1024;
    info["build_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    info["class_labels"]=coco_labels();info["deployment_approved"]=false;
    // Manifest is the completion marker. A partial bundle is not loadable.
    if(!std::filesystem::create_directories(bundle)) throw std::runtime_error("Engine output directory collision");
    write(bundle/"model.engine",plan->data(),plan->size());
    if(calibrator) info["calibration"]=calibrator->save(bundle);
    Trt<nvinfer1::IEngineInspector> inspector(engine->createEngineInspector());
    if(inspector) {
        const char* layers=inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
        if(layers) { const std::string text(layers);write(bundle/"layers.json",text.data(),text.size()); }
    }
    const auto text=info.dump(2);write(bundle/"manifest.json",text.data(),text.size());
}

struct TensorRtYoloDetector::Impl {
    Logger logger;
    Trt<nvinfer1::IRuntime> runtime;
    Trt<nvinfer1::ICudaEngine> engine;
    Trt<nvinfer1::IExecutionContext> context;
    Buffer input,output;
    Stream stream; // destroyed/synchronized before device buffers, context and engine
    Json info;
    YoloConfig config;
    bool failed=false;
    explicit Impl(const std::filesystem::path& bundle,YoloConfig cfg):config(cfg) {
        if(config.input_size!=640) throw std::invalid_argument("TensorRT engine input is fixed at 640");
        (void)prepare_yolo(cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0)),config.input_size);
        // Exercise shared config validation before any GPU allocation.
        const int dims[]={1,84,1};cv::Mat dummy(3,dims,CV_32F,cv::Scalar(0));
        (void)decode_yolo(dummy,prepare_yolo(cv::Mat(1,1,CV_8UC3,cv::Scalar::all(0))),coco_labels(),config);
        const auto metadata_bytes=read(bundle/"manifest.json",64*1024);
        info=Json::parse(metadata_bytes.begin(),metadata_bytes.end());
        if(info.at("version")!=1 || info.at("class_labels")!=coco_labels() ||
            info.at("input")!=Json::array({1,3,640,640}) || info.at("output")!=Json::array({1,84,8400}) ||
            (info.at("precision")!="fp16-mixed" && info.at("precision")!="fp32" && info.at("precision")!="int8-mixed"))
            throw std::invalid_argument("Invalid YOLO engine manifest");
        if(info.at("precision")=="int8-mixed") {
            const auto& calibration=info.at("calibration");
            if(sha(read(bundle/"calibration.cache",1024*1024))!=calibration.at("cache_sha256").get<std::string>() ||
                sha(read(bundle/"calibration-manifest.json",256*1024))!=calibration.at("manifest_sha256").get<std::string>())
                throw std::invalid_argument("INT8 calibration provenance hash mismatch");
        }
        const auto bytes=read(bundle/"model.engine",256ULL*1024*1024);
        if(sha(bytes)!=info.at("engine_sha256").get<std::string>()) throw std::invalid_argument("TensorRT engine hash mismatch");
        const auto current=hardware();
        for(const auto key:{"gpu","compute_major","compute_minor","cuda_runtime","tensorrt"})
            if(current.at(key)!=info.at(key)) throw std::invalid_argument("Engine hardware/SDK mismatch: rebuild locally");
        info["current_cuda_driver"]=current.at("cuda_driver");
        std::size_t before{},total{},after{};check(cudaMemGetInfo(&before,&total));
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if(!runtime) throw std::runtime_error("TensorRT runtime unavailable");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
        if(!engine) throw std::runtime_error("TensorRT engine deserialization failed");
        validate_engine(*engine);context.reset(engine->createExecutionContext());
        if(!context) throw std::runtime_error("TensorRT context allocation failed");
        input.allocate(input_bytes);output.allocate(output_bytes);
        check(cudaStreamCreateWithFlags(&stream.ptr,cudaStreamNonBlocking));
        for(int i=0;i<2;++i) {
            const auto name=engine->getIOTensorName(i);
            if(!context->setTensorAddress(name,engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT?input.ptr:output.ptr))
                throw std::runtime_error("TensorRT tensor address rejected");
        }
        check(cudaMemGetInfo(&after,&total));
        info["device_free_before_load_bytes"]=before;info["device_free_after_load_bytes"]=after;
        info["memory_note"]="Device-wide free-memory snapshots, not isolated process peak";
        info["io_allocation_bytes"]=input_bytes+output_bytes;
    }
};
TensorRtYoloDetector::TensorRtYoloDetector(const std::filesystem::path& bundle,YoloConfig config)
    :impl_(std::make_unique<Impl>(bundle,config)) {}
TensorRtYoloDetector::~TensorRtYoloDetector()=default;
Json TensorRtYoloDetector::metadata() const { return impl_->info; }
std::vector<Detection> TensorRtYoloDetector::detect(const Frame& frame) {
    if(impl_->failed) throw std::runtime_error("TensorRT worker faulted; create a new instance");
    if(!frame.image) throw std::invalid_argument("Frame has no image pixels");
    const auto& image=*frame.image;
    if(image.width<1 || image.height<1 || image.width>1920 || image.height>1920 ||
        image.stride<static_cast<std::size_t>(image.width)*3 ||
        image.stride>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(image.height) ||
        image.pixels.size()<image.stride*static_cast<std::size_t>(image.height))
        throw std::invalid_argument("Invalid or oversized TensorRT image buffer");
    const cv::Mat view(image.height,image.width,CV_8UC3,const_cast<std::uint8_t*>(image.pixels.data()),image.stride);
    const auto prepared=prepare_yolo(view,640);
    const int dims[]={1,84,8400};cv::Mat predictions(3,dims,CV_32F);
    try {
        check(cudaSetDevice(0));
        check(cudaMemcpyAsync(impl_->input.ptr,prepared.blob.ptr<float>(),input_bytes,cudaMemcpyHostToDevice,impl_->stream.ptr));
        if(!impl_->context->enqueueV3(impl_->stream.ptr)) throw std::runtime_error("TensorRT enqueue failed");
        check(cudaMemcpyAsync(predictions.ptr<float>(),impl_->output.ptr,output_bytes,cudaMemcpyDeviceToHost,impl_->stream.ptr));
        check(cudaStreamSynchronize(impl_->stream.ptr));
        if(!cv::checkRange(predictions)) throw std::runtime_error("Nonfinite TensorRT predictions");
    } catch(...) {
        // Host buffers must outlive any queued transfers, also on the error path.
        (void)cudaStreamSynchronize(impl_->stream.ptr);impl_->failed=true;throw;
    }
    return decode_yolo(predictions,prepared,coco_labels(),impl_->config);
}
} // namespace aegisvision::vision
