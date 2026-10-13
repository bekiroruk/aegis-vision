#include "aegisvision/tensorrt.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
using namespace aegisvision::vision;
void require(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
template<class F> void rejects(F f,const std::string& expected="") {
    bool rejected=false;try { f(); } catch(const std::exception& e) {
        rejected=std::string(e.what()).find(expected)!=std::string::npos;
    }
    require(rejected,"Invalid TensorRT input accepted");
}
int run(const std::vector<std::filesystem::path>& args) {
    try {
        const auto missing=std::filesystem::path("nonexistent-aegis-trt-test-bundle");
        rejects([&]{ TensorRtYoloDetector model(missing); });
        YoloConfig config;config.input_size=320;
        rejects([&]{ TensorRtYoloDetector model(missing,config); },"fixed at 640");
        config.input_size=640;config.confidence_threshold=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{ TensorRtYoloDetector model(missing,config); },"Invalid YOLO configuration");
        rejects([&]{ build_yolo_tensorrt("nonexistent-aegis.onnx",missing,true); });
        const auto scratch=std::filesystem::temp_directory_path()/
            ("aegis-trt-contract-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(scratch),"Temporary fixture collision");
        struct Cleanup { std::filesystem::path p;~Cleanup() { std::error_code ec;std::filesystem::remove_all(p,ec); } } cleanup{scratch};
        nlohmann::json manifest={{"version",1},{"class_labels",coco_labels()},
            {"input",{1,3,640,640}},{"output",{1,84,8400}},{"precision","fp16-mixed"},{"engine_sha256",std::string(64,'0')}};
        std::ofstream(scratch/"manifest.json") << manifest.dump();
        std::ofstream(scratch/"model.engine",std::ios::binary) << "not-an-engine";
        rejects([&]{ TensorRtYoloDetector model(scratch); },"hash mismatch");
        manifest["class_labels"][0]="wrong-order";
        std::ofstream(scratch/"manifest.json") << manifest.dump();
        rejects([&]{ TensorRtYoloDetector model(scratch); },"Invalid YOLO engine manifest");
        rejects([&]{ build_yolo_tensorrt("nonexistent-aegis.onnx",scratch,true); },"already exists");
        if(args.size()==3) {
            std::filesystem::copy_file(args[1]/"model.engine",scratch/"model.engine",std::filesystem::copy_options::overwrite_existing);
            nlohmann::json real_manifest;std::ifstream(args[1]/"manifest.json") >> real_manifest;
            real_manifest["gpu"]="not-the-build-device";
            std::ofstream(scratch/"manifest.json") << real_manifest.dump();
            rejects([&]{ TensorRtYoloDetector wrong(scratch); },"hardware/SDK mismatch");
            TensorRtYoloDetector model(args[1]);
            const auto image=cv::imread(args[2].string());require(!image.empty(),"Missing real image fixture");
            const auto frame=image_frame(image,"gpu-test","test");
            const auto first=model.detect(frame),second=model.detect(frame);
            require(!first.empty() && first.size()==second.size(),"Empty/unstable GPU predictions");
            for(std::size_t i=0;i<first.size();++i) {
                require(first[i].label==second[i].label && first[i].bbox.iou(second[i].bbox)>.999F &&
                    std::abs(first[i].score-second[i].score)<1e-5F,"Repeated GPU inference differs");
            }
            rejects([&]{ (void)model.detect(aegisvision::Frame{}); });
            auto broken=frame;auto pixels=std::make_shared<aegisvision::ImageBuffer>(*frame.image);
            pixels->stride=1;broken.image=pixels;
            rejects([&]{ (void)model.detect(broken); });
            pixels->stride=std::numeric_limits<std::size_t>::max();
            rejects([&]{ (void)model.detect(broken); });
            require(model.detect(frame).size()==first.size(),"Invalid frame poisoned healthy worker");
            std::cout << "Real GPU repeatability and buffer checks passed; detections=" << first.size() << '\n';
        }
        std::cout << "TensorRT contracts passed\n";return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
