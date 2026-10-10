#include "aegisvision/ocr.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs=std::filesystem;
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout<<"Usage: aegisvision_ocr DETECTOR.onnx CRNN_EN.onnx IMAGE NEW_OUTPUT_DIR\n"
                "CPU PP-OCRv3 + CRNN EN; lowercase a-z/0-9, no Turkish characters.\n"
                "Writes result.json and preview.jpg. Max side 1920, 64 regions, cooperative 10s deadline.\n";return 0;
        }
        if(args.size()!=5) throw std::invalid_argument("Use --help for arguments");
        if(fs::exists(args[4])) throw std::invalid_argument("Output directory must not exist");
        if(!fs::is_regular_file(args[3]) || fs::file_size(args[3])>32*1024*1024)
            throw std::invalid_argument("Input must be a local image <=32 MiB");
        std::ifstream input(args[3],std::ios::binary);
        std::vector<unsigned char> bytes(static_cast<std::size_t>(fs::file_size(args[3])));
        if(bytes.empty() || !input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())))
            throw std::runtime_error("Cannot read input image");
        auto image=cv::imdecode(bytes,cv::IMREAD_COLOR);
        if(image.empty() || image.cols>1920 || image.rows>1920) throw std::invalid_argument("Image must decode with sides <=1920");
        cv::setNumThreads(1);
        aegisvision::vision::PpocrCrnn ocr(args[1],args[2]);
        const auto started=std::chrono::steady_clock::now();
        const auto regions=ocr.read(image);
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        if(!fs::create_directories(args[4])) throw std::runtime_error("Cannot create new output directory");
        std::ofstream report(args[4]/"result.json");
        report<<std::setprecision(9)<<"{\"schema_version\":1,\"backend\":\"ppocrv3-crnn-en-opencv-cpu\","
            "\"charset\":\"0123456789abcdefghijklmnopqrstuvwxyz\",\"recognition_confidence_kind\":\"mean_emitted_softmax_uncalibrated\","
            "\"polygon_order\":\"BL,TL,TR,BR\",\"width\":"<<image.cols<<",\"height\":"<<image.rows
            <<",\"elapsed_ms\":"<<elapsed<<",\"regions\":[";
        for(std::size_t i=0;i<regions.size();++i) {
            const auto& r=regions[i];if(i) report<<',';
            // Text is restricted to the EN decoder's ASCII alphabet; cannot contain JSON control characters.
            report<<"{\"text\":\""<<r.recognition.text<<"\",\"detection_confidence\":"<<r.detection_confidence
                <<",\"recognition_confidence\":"<<r.recognition.confidence<<",\"polygon\":[";
            std::vector<cv::Point> points;
            for(std::size_t j=0;j<4;++j) {
                if(j) report<<',';report<<'['<<r.polygon[j].x<<','<<r.polygon[j].y<<']';points.emplace_back(r.polygon[j]);
            }
            report<<"]}";cv::polylines(image,points,true,{0,255,0},2);
            cv::putText(image,std::to_string(i+1)+": "+r.recognition.text,
                {std::clamp(points[1].x,0,image.cols-1),std::max(16,points[1].y-6)},cv::FONT_HERSHEY_SIMPLEX,.55,{0,0,255},2);
            std::cout<<i+1<<"\t"<<r.recognition.text<<"\t"<<r.recognition.confidence<<'\n';
        }
        report<<"]}\n";report.close();if(!report) throw std::runtime_error("OCR report write failed");
        std::vector<unsigned char> jpeg;if(!cv::imencode(".jpg",image,jpeg)) throw std::runtime_error("Preview encode failed");
        std::ofstream preview(args[4]/"preview.jpg",std::ios::binary);
        preview.write(reinterpret_cast<const char*>(jpeg.data()),static_cast<std::streamsize>(jpeg.size()));preview.close();
        if(!preview) throw std::runtime_error("Preview write failed");
        std::cout<<"Completed regions="<<regions.size()<<" elapsed_ms="<<elapsed<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv,argv+argc));
}
