#include "aegisvision/ocr_multilingual.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
namespace fs=std::filesystem;
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2&&args[1]=="--help") {std::cout<<"Usage: aegisvision_ocr_multi BUNDLE IMAGE NEW_OUTPUT_DIR\nPP-OCRv6 tiny UTF-8 recognizer + PP-OCRv3 detector; CPU.\n";return 0;}
        if(args.size()!=4) throw std::invalid_argument("Use --help for arguments");
        if(fs::exists(args[3])) throw std::invalid_argument("Output directory must not exist");
        if(!fs::is_regular_file(args[2])||fs::file_size(args[2])>32*1024*1024) throw std::invalid_argument("Input must be <=32 MiB");
        std::ifstream input(args[2],std::ios::binary);std::vector<unsigned char> bytes(static_cast<std::size_t>(fs::file_size(args[2])));
        if(bytes.empty()||!input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()))) throw std::runtime_error("Cannot read image");
        auto image=cv::imdecode(bytes,cv::IMREAD_COLOR);
        cv::setNumThreads(1);aegisvision::vision::PpocrMultilingual ocr(args[1]);
        auto start=std::chrono::steady_clock::now();const auto regions=ocr.read(image);
        nlohmann::json report={{"schema_version",1},{"model_signature",ocr.signature()},{"charset","utf8-ppocrv6-tiny"},
            {"polygon_order","BL,TL,TR,BR"},{"confidence_kind","mean_emitted_probability_uncalibrated"},
            {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()},
            {"width",image.cols},{"height",image.rows},{"regions",nlohmann::json::array()}};
        for(std::size_t i=0;i<regions.size();++i) {
            const auto& r=regions[i];nlohmann::json polygon=nlohmann::json::array();std::vector<cv::Point> points;
            for(const auto& p:r.polygon) {polygon.push_back({p.x,p.y});points.emplace_back(p);}
            report["regions"].push_back({{"text",r.recognition.text},{"recognition_confidence",r.recognition.confidence},
                {"detection_confidence",r.detection_confidence},{"polygon",polygon}});
            cv::polylines(image,points,true,{0,255,0},2);
            cv::putText(image,std::to_string(i+1),points[1],cv::FONT_HERSHEY_SIMPLEX,.6,{0,0,255},2);
        }
        if(!fs::create_directories(args[3])) throw std::runtime_error("Cannot create output directory");
        std::ofstream output(args[3]/"result.json");output<<report.dump(2)<<'\n';output.close();if(!output) throw std::runtime_error("Report write failed");
        std::vector<unsigned char> jpeg;if(!cv::imencode(".jpg",image,jpeg)) throw std::runtime_error("JPEG encode failed");
        std::ofstream preview(args[3]/"preview.jpg",std::ios::binary);preview.write(reinterpret_cast<const char*>(jpeg.data()),static_cast<std::streamsize>(jpeg.size()));preview.close();
        if(!preview) throw std::runtime_error("Preview write failed");
        std::cout<<report.dump(2)<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv,argv+argc));
}
