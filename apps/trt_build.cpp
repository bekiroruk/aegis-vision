#include "aegisvision/tensorrt.hpp"
#include <iostream>
namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_trt_build MODEL.onnx NEW_BUNDLE fp16|fp32\n"
                "Locally builds static YOLOv8n detection engine; device 0, 512 MiB workspace, TF32 disabled.\n"
                "FP16 allows mixed-precision layers; input/output FP32. No INT8 or segmentation.\n";
            return 0;
        }
        if(args.size()!=4 || (args[3]!="fp16" && args[3]!="fp32")) throw std::invalid_argument("Use --help");
        aegisvision::vision::build_yolo_tensorrt(args[1],args[2],args[3]=="fp16");
        std::cout << "Engine and provenance saved: " << args[2].string() << '\n';return 0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
