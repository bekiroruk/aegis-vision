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
        rejects([&]{ build_yolo_tensorrt("nonexistent-aegis.onnx",missing,false,"calibration.json"); },"FP16 fallback");
        const auto scratch=std::filesystem::temp_directory_path()/
            ("aegis-trt-contract-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(scratch),"Temporary fixture collision");
        struct Cleanup { std::filesystem::path p;~Cleanup() { std::error_code ec;std::filesystem::remove_all(p,ec); } } cleanup{scratch};
        // Calibration rejection must happen before any GPU/model parsing work.
        std::ofstream(scratch/"fixture.onnx",std::ios::binary) << "test";
        nlohmann::json calibration={{"version",1},{"kind","int8-calibration"},{"algorithm","entropy2"},{"batch_size",1},
            {"preprocessing","prepare_yolo-v1-640-rgb-nchw-0to1-pad114"},
            {"model_sha256","9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"},
            {"images",nlohmann::json::array()},{"excluded_images",nlohmann::json::array()}};
        for(int i=0;i<64;++i) calibration["excluded_images"].push_back({{"coco_image_id",i},{"sha256","excluded-"+std::to_string(i)}});
        for(int i=0;i<128;++i) calibration["images"].push_back({{"coco_image_id",i+64},{"sha256","image-"+std::to_string(i)},
            {"decoded_sha256","pixels-"+std::to_string(i)},{"path","image.jpg"}});
        auto reject_calibration=[&](const nlohmann::json& fixture,const std::string& expected) {
            std::ofstream(scratch/"calibration.json") << fixture.dump();
            rejects([&]{ build_yolo_tensorrt(scratch/"fixture.onnx",scratch/"new-engine",true,scratch/"calibration.json"); },expected);
            require(!std::filesystem::exists(scratch/"new-engine"),"Invalid calibration published bundle");
        };
        auto bad=calibration;bad["batch_size"]=2;reject_calibration(bad,"protocol/model");
        bad=calibration;bad["model_sha256"]="other";reject_calibration(bad,"protocol/model");
        bad=calibration;bad["images"][0]["coco_image_id"]=0;reject_calibration(bad,"overlap");
        bad=calibration;bad["images"][0]["sha256"]="excluded-0";reject_calibration(bad,"overlap");
        bad=calibration;bad["images"][0]["path"]=scratch.string();reject_calibration(bad,"Absolute calibration path");
        bad=calibration;bad["images"][0]["path"]="..";reject_calibration(bad,"escapes dataset");
        std::ofstream(scratch/"image.jpg",std::ios::binary) << "test";
        reject_calibration(calibration,"image hash mismatch");
        bad=calibration;bad["images"][0]["sha256"]=calibration["model_sha256"];
        bad["images"][1]["decoded_sha256"]=bad["images"][0]["decoded_sha256"];
        reject_calibration(bad,"overlap");
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
            if(real_manifest.at("precision")=="int8-mixed") {
                for(const auto filename:{"calibration.cache","calibration-manifest.json"})
                    std::filesystem::copy_file(args[1]/filename,scratch/filename);
                std::ofstream(scratch/"manifest.json") << real_manifest.dump();
                std::ofstream(scratch/"calibration.cache",std::ios::app) << "mutation";
                rejects([&]{ TensorRtYoloDetector wrong(scratch); },"calibration provenance hash mismatch");
                std::filesystem::copy_file(args[1]/"calibration.cache",scratch/"calibration.cache",std::filesystem::copy_options::overwrite_existing);
            }
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
