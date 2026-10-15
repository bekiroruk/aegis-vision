#include "aegisvision/application_config.hpp"
#include "aegisvision/clip.hpp"
#include "aegisvision/service.hpp"
#include "aegisvision/ocr_multilingual.hpp"
#include <charconv>
#include <algorithm>
#include <iostream>
#ifdef AEGISVISION_SERVICE_TENSORRT
#include "aegisvision/tensorrt.hpp"
#include "aegisvision/quality_benchmark.hpp"
#endif

namespace {
std::unique_ptr<aegisvision::IDetector> server_detector(
    const aegisvision::vision::ApplicationSettings& settings,const std::filesystem::path& bundle) {
    if(bundle.empty()) return aegisvision::vision::make_configured_detector(settings);
#ifdef AEGISVISION_SERVICE_TENSORRT
    auto detector=std::make_unique<aegisvision::vision::TensorRtYoloDetector>(bundle,settings.detector);
    const auto info=detector->metadata();
    if(info.at("precision")!="fp16-mixed") throw std::invalid_argument("Service accepts only FP16-mixed TensorRT engines; INT8 is not approved");
    if(info.at("model_sha256")!=aegisvision::evaluation::quality_file_sha256(settings.detector_model))
        throw std::invalid_argument("TensorRT engine source differs from configured ONNX");
    return detector;
#else
    throw std::invalid_argument("This server was built without TensorRT; use a TensorRT-enabled build");
#endif
}
int run(std::vector<std::filesystem::path> args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "Usage: aegisvision_server SEARCH.toml DETECTOR.toml MEDIA_DIR WEB_DIR [PORT [JOB_DB]]\n"
                "   Live: append STREAM.toml RTSP_URL [FFMPEG] after PORT JOB_DB to enable preset local-pedestrians and opt-in archive.\n"
                "   Segmentation: append --segment-model MODEL.onnx at the end to enable queued segment_frame jobs.\n"
                "   Add --live-segmentation with live arguments and --segment-model to draw tracked live masks.\n"
                "   OCR: --ocr-models BUNDLE_DIR (CRNN EN) OR --ocr-multilingual BUNDLE_DIR (PP-OCRv6 UTF-8).\n"
                "   VLM: --vlm-port PORT enables queued scene_frame jobs through local llama.cpp.\n"
                "   Analysis: --analysis-config FILE.toml replaces segmentation/OCR/VLM flags.\n"
                "   GPU detection: --trt-bundle LOCAL_FP16_BUNDLE (requires TensorRT build; no CPU fallback).\n"
                "   Applies to file indexing and live detection, not segmentation/OCR/CLIP/VLM.\n"
                "   Demo: --replay-file treats the live source argument as a local video, not RTSP. Loops at configured output_fps.\n"
                "Local HTTP worker + dashboard. Defaults: port 8090, artifacts/service/jobs.sqlite. Creates Qdrant collection if missing.\n";
            return 0;
        }
        std::filesystem::path segmentation_model,ocr_bundle;
        bool live_segmentation=false;
        bool file_replay=false;
        const auto replay_flag=std::find(args.begin(),args.end(),std::filesystem::path("--replay-file"));
        if (replay_flag!=args.end()) {file_replay=true;args.erase(replay_flag);}
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
        auto multilingual_bundle=take_path("--ocr-multilingual");
        auto vlm_port_text=take_path("--vlm-port");
        const auto analysis_file=take_path("--analysis-config");
        const auto trt_bundle=take_path("--trt-bundle");
#ifndef AEGISVISION_SERVICE_TENSORRT
        if(!trt_bundle.empty()) throw std::invalid_argument("This server was built without TensorRT");
#endif
        int vlm_timeout_ms=60000;
        if(!analysis_file.empty()) {
            if(!segmentation_model.empty()||!ocr_bundle.empty()||!multilingual_bundle.empty()||!vlm_port_text.empty())
                throw std::invalid_argument("Do not mix analysis TOML with model flags");
            const auto analysis=aegisvision::vision::load_analysis_settings(analysis_file);
            segmentation_model=analysis.segmentation_model;
            if(analysis.ocr_multilingual) multilingual_bundle=analysis.ocr_bundle;
            else ocr_bundle=analysis.ocr_bundle;
            if(analysis.vlm_port) vlm_port_text=std::to_string(analysis.vlm_port);
            vlm_timeout_ms=analysis.vlm_timeout_ms;
        }
        if(!ocr_bundle.empty()&&!multilingual_bundle.empty()) throw std::invalid_argument("Choose one OCR backend");
        if ((args.size() < 5 || args.size() > 7) && args.size() != 9 && args.size()!=10) throw std::invalid_argument("Missing arguments; run --help");
        if (file_replay && args.size()<9) throw std::invalid_argument("Replay requires stream config and local video arguments");
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
        auto detector = server_detector(detection,trt_bundle);
        nlohmann::json backend={{"engine","opencv-dnn"},{"device","cpu"},{"precision","fp32"}};
        auto signature=aegisvision::yolo_index_signature(detection.detector_model,detection.detector);
#ifdef AEGISVISION_SERVICE_TENSORRT
        if(!trt_bundle.empty()) {
            const auto info=dynamic_cast<aegisvision::vision::TensorRtYoloDetector&>(*detector).metadata();
            backend={{"engine","tensorrt"},{"device","cuda:0"},{"precision",info.at("precision")},
                {"gpu",info.at("gpu")},{"engine_sha256",info.at("engine_sha256")}};
            // Separate cache/index identities from CPU and from other engine builds.
            signature="tensorrt-fp16-v1:"+info.at("engine_sha256").get<std::string>()+":"+signature;
        }
#endif
        std::cout << "Detection backend: " << backend.dump() << "\nLoading CLIP..." << std::endl;
        aegisvision::ClipEmbedder clip(search.clip_bundle);
        search.qdrant.embedding_space = clip.space_id();
        aegisvision::QdrantVectorStore store(search.qdrant, true);
        aegisvision::ServiceConfig config{args[3], args[4],signature};
        config.detector_backend=backend;
        std::unique_ptr<aegisvision::vision::YoloSegmenter> segmenter;
        std::unique_ptr<aegisvision::vision::IOcr> ocr;
        std::unique_ptr<aegisvision::vision::LocalVlm> vlm;
        if(!vlm_port_text.empty()) {
            const auto text=vlm_port_text.string();int vlm_port{};
            const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),vlm_port);
            if(error!=std::errc{}||end!=text.data()+text.size()||vlm_port<1||vlm_port>65535)
                throw std::invalid_argument("VLM port must be 1..65535");
            vlm=std::make_unique<aegisvision::vision::LocalVlm>(aegisvision::vision::VlmConfig{vlm_port,vlm_timeout_ms});
            config.vlm=vlm.get();config.vlm_signature="local-smolvlm-500m-v1:127.0.0.1:"+std::to_string(vlm_port);
        }
        if(!multilingual_bundle.empty()) {
            auto model=std::make_unique<aegisvision::vision::PpocrMultilingual>(multilingual_bundle);
            config.ocr_signature=model->signature();ocr=std::move(model);config.ocr=ocr.get();
        }
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
            live.live.file_replay=file_replay;
            const auto raw = (file_replay ? std::filesystem::absolute(args[8]) : args[8]).u8string();
            const std::string url(raw.begin(), raw.end());
            aegisvision::vision::validate_live_source(url,live.live);
            config.live = {{{file_replay ? "local-replay" : "local-pedestrians",
                file_replay ? "Yerel video demosu (kamera değil)" : "Yerel RTSP yayını", url}}, live.live, live.video};
            config.live.backend=live_segmentation ? nlohmann::json{{"engine","opencv-dnn"},{"device","cpu"},{"precision","fp32"}} : backend;
            config.live.archive.enabled=true;
            if (args.size()==10) config.archive_encoder.executable=args[9];
            // Each live worker loads and validates its own context/stream. Never shares queue-worker GPU state.
            live_detector = [live,trt_bundle,backend] {
                auto instance=server_detector(live,trt_bundle);
#ifdef AEGISVISION_SERVICE_TENSORRT
                if(!trt_bundle.empty() && dynamic_cast<aegisvision::vision::TensorRtYoloDetector&>(*instance).metadata().at("engine_sha256")!=backend.at("engine_sha256"))
                    throw std::invalid_argument("Live TensorRT engine changed since server startup");
#endif
                return instance;
            };
            if (live_segmentation) live_segmenter=[segmentation_model] {
                return std::make_unique<aegisvision::vision::YoloSegmenter>(segmentation_model);
            };
        }
        aegisvision::LocalService service(*detector, clip, store, config, std::move(live_detector),{},std::move(live_segmenter));
        const auto bound = service.bind(port);
        std::cout << "AegisVision ready: http://127.0.0.1:" << bound << " / collection=" << search.qdrant.collection << std::endl;
        return service.listen_until_signal() ? 0 : 1;
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
