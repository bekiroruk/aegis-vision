#include "aegisvision/tensorrt.hpp"
#include "aegisvision/live_session.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <thread>
using namespace aegisvision;
using namespace std::chrono_literals;
namespace {
void require(bool value,const char* reason) { if(!value) throw std::runtime_error(reason); }
template<class F> void until(F check) {
    const auto deadline=std::chrono::steady_clock::now()+15s;
    while(!check()) {
        if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("GPU live worker timed out");
        std::this_thread::sleep_for(20ms);
    }
}
// Deterministic capture fixture only: real GPU inference, not a real RTSP transport test.
class Capture final:public vision::ILiveCapture {
    cv::Mat image_;
public:
    explicit Capture(cv::Mat image):image_(std::move(image)) {}
    bool open(const std::string&,const vision::LiveConfig&) override { return true; }
    bool read(cv::Mat& frame) override { std::this_thread::sleep_for(20ms);frame=image_.clone();return true; }
    void close() override {}
};
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if(args.size()!=3) throw std::invalid_argument("Expected FP16 bundle and real image");
        cv::setNumThreads(1);
        const auto image=cv::imread(args[2].string());require(!image.empty(),"Image missing");
        vision::TensorRtYoloDetector file_detector(args[1]);
        require(file_detector.metadata().at("precision")=="fp16-mixed","FP16 engine required");
        const auto frame=vision::image_frame(image,"file-fixture","file");
        LiveServiceConfig config;
        config.sources={{"fixture","Real GPU / fixture capture","rtsp://127.0.0.1/fixture"}};
        config.stream.duration_seconds=60;
        config.backend={{"engine","tensorrt"},{"device","cuda:0"},{"precision","fp16-mixed"}};
        std::atomic_int instances=0;
        LiveSessions live(config,[&] { ++instances;return std::make_unique<vision::TensorRtYoloDetector>(args[1]); },
            [image] {return std::make_unique<Capture>(image);});
        for(int iteration=0;iteration<2;++iteration) {
            const auto accepted=live.start("fixture");
            until([&] { const auto s=live.current();return s.at("has_preview")==true || s.at("state")=="failed"; });
            const auto before=live.current();require(before.at("has_preview")==true,"GPU live preview failed");
            for(int i=0;i<20;++i) require(!file_detector.detect(frame).empty(),"Concurrent GPU file inference empty");
            until([&] {return live.current().at("processed_frames").get<int>()>before.at("processed_frames").get<int>();});
            const auto snapshot=live.current();
            const auto preview=live.preview(accepted.at("id").get<std::string>());require(bool(preview.jpeg),"Missing concurrent GPU JPEG");
            require(!cv::imdecode(*preview.jpeg,cv::IMREAD_COLOR).empty(),"GPU JPEG not decodable");
            require(snapshot.at("queue_high_watermark").get<int>()<=1,"GPU live queue unbounded");
            require(snapshot.at("backend")==config.backend,"GPU backend label differs");
            require(live.request_stop(accepted.at("id").get<std::string>()),"GPU stop rejected");
            until([&] { return live.current().at("active")==false; });
            require(!live.preview(accepted.at("id").get<std::string>()).jpeg && live.current().at("state")=="stopped","Stale GPU preview after stop");
        }
        require(instances==2,"Live GPU context not recreated for each session");
        std::cout << "PASS: real FP16 inference in isolated file/live GPU contexts; fixture capture, JPEG, bounded queue, stop/restart. RTSP transport not tested.\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
