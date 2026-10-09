#include "aegisvision/segmentation.hpp"
#include "aegisvision/quality_benchmark.hpp"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
#include <set>

namespace fs=std::filesystem;
using Json=nlohmann::json;
std::string utf8(const fs::path& p) {const auto s=p.u8string();return {s.begin(),s.end()};}
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_segment_quality MODEL.onnx IMAGE_QUALITY_MANIFEST.json NEW_OUTPUT_DIR\n"
                "C++ inference only; full-image COCO RLE predictions at confidence .35, max 100.\n";return 0;
        }
        if(args.size()!=4) throw std::invalid_argument("Use --help");
        if(fs::exists(args[3])) throw std::invalid_argument("Output must not exist");
        const auto hash=[](const fs::path& p){return aegisvision::evaluation::quality_file_sha256(p);};
        const auto model_hash=hash(args[1]), manifest_hash=hash(args[2]);
        if(fs::file_size(args[2])>64*1024*1024) throw std::invalid_argument("Manifest too large");
        std::ifstream file(args[2]);const auto manifest=Json::parse(file);
        if(manifest.at("kind")!="images" || !manifest.at("images").is_array() ||
            manifest.at("images").empty() || manifest.at("images").size()>100)
            throw std::invalid_argument("Expected 1..100 quality-manifest images");
        cv::setNumThreads(1);
        aegisvision::vision::YoloSegmenter segmenter(args[1]);
        Json predictions=Json::array(), images=Json::array();std::set<int> seen;
        const auto base=fs::canonical(args[2]).parent_path();
        fs::create_directories(args[3]);
        for(const auto& entry:manifest.at("images")) {
            const auto relative=fs::u8path(entry.at("path").get<std::string>());
            if(relative.is_absolute()) throw std::invalid_argument("Absolute image path");
            const auto path=fs::canonical(base/relative), inside=path.lexically_relative(base);
            if(inside.empty() || inside.is_absolute()) throw std::invalid_argument("Image outside manifest directory");
            for(const auto& part:inside) if(part=="..") throw std::invalid_argument("Image path escape");
            if(hash(path)!=entry.at("sha256").get<std::string>()) throw std::invalid_argument("Image hash mismatch");
            const int id=entry.at("coco_image_id").get<int>();
            if(!seen.insert(id).second) throw std::invalid_argument("Duplicate image ID");
            const auto image=cv::imread(utf8(path));
            if(image.empty() || image.cols!=entry.at("width").get<int>() || image.rows!=entry.at("height").get<int>())
                throw std::invalid_argument("Image dimensions mismatch");
            const auto masks=segmenter.segment(image);
            for(const auto& item:masks) predictions.push_back({{"image_id",id},{"label",item.detection.label},
                {"score",item.detection.score},{"segmentation",{{"size",{image.rows,image.cols}},
                {"counts",aegisvision::vision::mask_rle(item,image.size())}}}});
            if(images.empty() && !cv::imwrite(utf8(args[3]/"preview.jpg"),aegisvision::vision::paint_masks(image,masks)))
                throw std::runtime_error("Preview write failed");
            if(hash(path)!=entry.at("sha256").get<std::string>()) throw std::runtime_error("Image changed during inference");
            images.push_back({{"id",id},{"sha256",entry.at("sha256")},{"width",image.cols},{"height",image.rows}});
            std::cout << "images=" << images.size() << " masks=" << predictions.size() << std::endl;
        }
        if(hash(args[1])!=model_hash || hash(args[2])!=manifest_hash) throw std::runtime_error("Inputs changed");
        Json report{{"version",1},{"model_sha256",model_hash},{"manifest_sha256",manifest_hash},
            {"confidence",.35},{"nms_iou",.45},{"mask_threshold",.5},{"max_instances",100},
            {"images",images},{"predictions",predictions},{"tracking",false}};
        std::ofstream output(args[3]/"mask-predictions.json");output << report.dump();output.close();
        if(!output) throw std::runtime_error("Prediction write failed");
        return 0;
    } catch(const std::exception& e) {std::cerr << "Error: " << e.what() << '\n';return 1;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    std::vector<fs::path> args;for(int i=0;i<argc;++i)args.emplace_back(argv[i]);return run(args);
}
