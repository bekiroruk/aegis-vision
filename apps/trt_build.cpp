#include "aegisvision/tensorrt.hpp"
#include <iostream>
namespace {
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_trt_build MODEL.onnx NEW_BUNDLE fp16|fp32|int8 [CALIBRATION_MANIFEST]\n"
                "Locally builds static YOLOv8n detection engine; device 0, 512 MiB workspace, TF32 disabled.\n"
                "INT8 requires a disjoint 128-image manifest (Entropy2, TensorRT 10.13.3 only).\n"
                "FP16/INT8 allow mixed precision; input/output FP32. No segmentation.\n";
            return 0;
        }
        if(!((args.size()==4 && (args[3]=="fp16" || args[3]=="fp32")) ||
            (args.size()==5 && args[3]=="int8"))) throw std::invalid_argument("Use --help");
        aegisvision::vision::build_yolo_tensorrt(args[1],args[2],args[3]!="fp32",
            args.size()==5?args[4]:std::filesystem::path{});
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
