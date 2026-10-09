#include "aegisvision/segmentation.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace fs=std::filesystem;
std::string utf8(const fs::path& path) { const auto s=path.u8string();return {s.begin(),s.end()}; }
int run(const std::vector<fs::path>& args) {
    try {
        if (args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_segment MODEL.onnx VIDEO NEW_OUTPUT_DIR [MAX_FRAMES]\n"
                "YOLOv8-seg CPU FP32; writes side-by-side segmented.avi, preview.jpg, instances.csv,\n"
                "first-frame ROI masks and summary.json. No tracking IDs. Max image side 1920.\n";
            return 0;
        }
        if (args.size()!=4 && args.size()!=5) throw std::invalid_argument("Use --help for arguments");
        int limit=0;
        if (args.size()==5) {
            const auto s=args[4].string();const auto [end,ec]=std::from_chars(s.data(),s.data()+s.size(),limit);
            if (ec!=std::errc{} || end!=s.data()+s.size() || limit<1) throw std::invalid_argument("Invalid frame limit");
        }
        if (!fs::is_regular_file(args[2])) throw std::invalid_argument("Input must be a local video file");
        if (fs::exists(args[3])) throw std::invalid_argument("Output directory must not exist");
        cv::setNumThreads(1);
        aegisvision::vision::YoloSegmenter segmenter(args[1]);
        cv::VideoCapture capture(utf8(args[2]));
        if (!capture.isOpened()) throw std::runtime_error("Cannot open video");
        const double fps=capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps<=0 || fps>240) throw std::runtime_error("Invalid video FPS");
        cv::Mat frame;
        if (!capture.read(frame)) throw std::runtime_error("Video has no decodable frame");
        if (frame.cols>1920 || frame.rows>1920) throw std::runtime_error("Video exceeds segmentation size limit");
        const auto size=frame.size();
        fs::create_directories(args[3]);
        cv::VideoWriter writer(utf8(args[3]/"segmented.avi"),cv::CAP_OPENCV_MJPEG,
            cv::VideoWriter::fourcc('M','J','P','G'),fps,{size.width*2,size.height});
        if (!writer.isOpened()) throw std::runtime_error("Cannot create mask video");
        std::ofstream rows(args[3]/"instances.csv");
        rows << "frame,instance,class_id,score,x,y,width,height,mask_pixels\n";
        rows << std::setprecision(9);
        int frames=0;std::size_t instances=0;
        do {
            if (frame.size()!=size) throw std::runtime_error("Video dimensions changed");
            const auto masks=segmenter.segment(frame);
            auto painted=aegisvision::vision::paint_masks(frame,masks);
            cv::Mat comparison;cv::hconcat(frame,painted,comparison);
            cv::putText(comparison,"Original",{8,20},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},2);
            cv::putText(comparison,"C++ YOLOv8 instance masks",{size.width+8,20},cv::FONT_HERSHEY_SIMPLEX,.6,{255,255,255},2);
            writer.write(comparison);
            if (frames==0 && !cv::imwrite(utf8(args[3]/"preview.jpg"),comparison)) throw std::runtime_error("Preview write failed");
            for (std::size_t i=0;i<masks.size();++i) {
                const auto& item=masks[i];const auto& r=item.region;
                rows << frames << ',' << i << ',' << item.detection.attributes.at("class_id") << ','
                    << item.detection.score << ',' << r.x << ',' << r.y << ',' << r.width << ',' << r.height << ','
                    << cv::countNonZero(item.mask) << '\n';
                if (frames==0 && !cv::imwrite(utf8(args[3]/("mask-"+std::to_string(i)+".png")),item.mask))
                    throw std::runtime_error("Mask write failed");
            }
            if (!rows) throw std::runtime_error("Instance report write failed");
            instances+=masks.size();++frames;
            if (frames%25==0) std::cout << "frames=" << frames << " instances=" << instances << std::endl;
        } while ((limit==0 || frames<limit) && capture.read(frame));
        writer.release();rows.close();
        if (!rows) throw std::runtime_error("Instance report flush failed");
        std::ofstream summary(args[3]/"summary.json");
        summary << "{\"frames\":" << frames << ",\"instances\":" << instances << ",\"fps\":" << fps
            << ",\"mask_threshold\":0.5,\"confidence\":0.35,\"nms_iou\":0.45,\"max_instances\":100,"
               "\"backend\":\"OpenCV CPU FP32\",\"tracking\":false,\"mask_format\":\"ROI uint8 0/255\"}\n";
        summary.close();if (!summary) throw std::runtime_error("Summary write failed");
        std::cout << "Completed frames=" << frames << " instances=" << instances << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    std::vector<fs::path> args;for(int i=0;i<argc;++i) args.emplace_back(argv[i]);return run(args);
}
