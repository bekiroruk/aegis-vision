#include "aegisvision/application_config.hpp"
#include "aegisvision/clip.hpp"
#include "aegisvision/service.hpp"
#include <charconv>
#include <algorithm>
#include <iostream>

namespace {
int run(std::vector<std::filesystem::path> args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "Usage: aegisvision_server SEARCH.toml DETECTOR.toml MEDIA_DIR WEB_DIR [PORT [JOB_DB]]\n"
                "   Live: append STREAM.toml RTSP_URL [FFMPEG] after PORT JOB_DB to enable preset local-pedestrians and opt-in archive.\n"
                "   Segmentation: append --segment-model MODEL.onnx at the end to enable queued segment_frame jobs.\n"
                "   Add --live-segmentation with live arguments and --segment-model to draw tracked live masks.\n"
                "   OCR: --ocr-models BUNDLE_DIR enables queued ocr_frame (PP-OCRv3 + CRNN EN).\n"
                "Local HTTP worker + dashboard. Defaults: port 8090, artifacts/service/jobs.sqlite. Creates Qdrant collection if missing.\n";
            return 0;
        }
        std::filesystem::path segmentation_model,ocr_bundle;
        bool live_segmentation=false;
        const auto live_flag=std::find(args.begin(),args.end(),std::filesystem::path("--live-segmentation"));
        if (live_flag!=args.end()) {live_segmentation=true;args.erase(live_flag);}
        const auto take_path=[&](const char* flag) {
            const auto it=std::find(args.begin()+1,args.end(),std::filesystem::path(flag));
            if(it==args.end()) return std::filesystem::path{};
            if(it+1==args.end() || (it+1)->string().starts_with("--")) throw std::invalid_argument("Missing model option value");
            auto value=*(it+1);args.erase(it,it+2);
            if(std::find(args.begin()+1,args.end(),std::filesystem::path(flag))!=args.end()) throw std::invalid_argument("Duplicate model option");
            return value;
        };
        segmentation_model=take_path("--segment-model");ocr_bundle=take_path("--ocr-models");
        if ((args.size() < 5 || args.size() > 7) && args.size() != 9 && args.size()!=10) throw std::invalid_argument("Missing arguments; run --help");
        if (live_segmentation && (segmentation_model.empty() || args.size()<9))
            throw std::invalid_argument("Live segmentation requires live source arguments and --segment-model");
        int port = 8090;
        if (args.size() >= 6) {
            const auto text = args[5].string();
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
            if (error != std::errc{} || end != text.data() + text.size() || port < 1 || port > 65535)
                throw std::invalid_argument("PORT must be 1..65535");
        }
        auto search = aegisvision::vision::load_application_settings(args[1]);
        auto detection = aegisvision::vision::load_application_settings(args[2]);
        if (search.mode != aegisvision::vision::ApplicationMode::Search || detection.mode != aegisvision::vision::ApplicationMode::Image)
            throw std::invalid_argument("Expected search-mode and image-mode TOML configurations");
        std::cout << "Loading CLIP and YOLO models..." << std::endl;
        aegisvision::ClipEmbedder clip(search.clip_bundle);
        search.qdrant.embedding_space = clip.space_id();
        aegisvision::QdrantVectorStore store(search.qdrant, true);
        auto detector = aegisvision::vision::make_configured_detector(detection);
        aegisvision::ServiceConfig config{args[3], args[4],
            aegisvision::yolo_index_signature(detection.detector_model, detection.detector)};
        std::unique_ptr<aegisvision::vision::YoloSegmenter> segmenter;
        std::unique_ptr<aegisvision::vision::PpocrCrnn> ocr;
        if(!ocr_bundle.empty()) {
            const auto detection_model=ocr_bundle/"text_detection_en_ppocrv3_2023may.onnx";
            const auto recognition_model=ocr_bundle/"text_recognition_CRNN_EN_2021sep.onnx";
            ocr=std::make_unique<aegisvision::vision::PpocrCrnn>(detection_model,recognition_model);
            config.ocr=ocr.get();config.ocr_signature=aegisvision::ocr_model_signature(detection_model,recognition_model);
        }
        if (!segmentation_model.empty()) {
            const aegisvision::vision::YoloConfig mask_config{640,.35F,.45F,100};
            segmenter=std::make_unique<aegisvision::vision::YoloSegmenter>(segmentation_model,mask_config);
            config.segmenter=segmenter.get();
            config.segmentation_signature="yolov8-seg-cpu-v1:"+
                aegisvision::yolo_index_signature(segmentation_model,mask_config);
        }
        config.persistence.database = args.size() >= 7 ? args[6] : std::filesystem::path("artifacts/service/jobs.sqlite");
        config.persistence.context = nlohmann::json::array({"clip-qdrant-v1", clip.space_id(),
            search.qdrant.host, search.qdrant.port, search.qdrant.collection, search.qdrant.dimension}).dump();
        aegisvision::LiveDetectorFactory live_detector;
        aegisvision::LiveSegmenterFactory live_segmenter;
        if (args.size() >= 9) {
            auto live = aegisvision::vision::load_application_settings(args[7]);
            if (live.mode != aegisvision::vision::ApplicationMode::Stream)
                throw std::invalid_argument("Live preview requires stream-mode TOML");
            const auto raw = args[8].u8string();
            const std::string url(raw.begin(), raw.end());
            aegisvision::vision::validate_rtsp_url(url);
            config.live = {{{"local-pedestrians", "Yerel RTSP yayını", url}}, live.live, live.video};
            config.live.archive.enabled=true;
            if (args.size()==10) config.archive_encoder.executable=args[9];
            live_detector = [live] { return aegisvision::vision::make_configured_detector(live); };
            if (live_segmentation) live_segmenter=[segmentation_model] {
                return std::make_unique<aegisvision::vision::YoloSegmenter>(segmentation_model);
            };
        }
        aegisvision::LocalService service(*detector, clip, store, config, std::move(live_detector),{},std::move(live_segmenter));
        const auto bound = service.bind(port);
        std::cout << "AegisVision ready: http://127.0.0.1:" << bound << " / collection=" << search.qdrant.collection << std::endl;
        return service.listen() ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
