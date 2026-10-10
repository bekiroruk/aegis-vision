#include "aegisvision/ocr_evaluation.hpp"
#include <toml++/toml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>

namespace fs=std::filesystem;
using namespace aegisvision::vision;
struct Sample {std::string id;fs::path image;int width{},height{};std::vector<OcrLabel> labels;};
std::string required_string(const toml::node_view<const toml::node>& node) {
    auto value=node.value<std::string>();
    if(!value||value->empty()||value->size()>1024) throw std::invalid_argument("Missing/invalid manifest string");
    return *value;
}
std::vector<Sample> read_manifest(const fs::path& path) {
    if(!fs::is_regular_file(path)||fs::file_size(path)>1024*1024) throw std::invalid_argument("Manifest must be <=1 MiB");
    std::ifstream stream(path);const auto root=toml::parse(stream);
    if(root["schema_version"].value<int>()!=1) throw std::invalid_argument("Expected manifest schema_version=1");
    const auto* samples=root["samples"].as_array();
    if(!samples||samples->empty()||samples->size()>1000) throw std::invalid_argument("Manifest requires 1..1000 samples");
    std::vector<Sample> result;std::set<std::string> ids;std::set<fs::path> files;
    for(const auto& node:*samples) {
        const auto* table=node.as_table();if(!table) throw std::invalid_argument("Sample must be a table");
        Sample s;s.id=required_string((*table)["id"]);
        if(!ids.insert(s.id).second) throw std::invalid_argument("Duplicate sample id");
        s.image=fs::weakly_canonical(path.parent_path()/fs::u8path(required_string((*table)["image"])));
        if(!files.insert(s.image).second) throw std::invalid_argument("Duplicate sample image");
        s.width=(*table)["width"].value<int>().value_or(0);s.height=(*table)["height"].value<int>().value_or(0);
        if(s.width<1||s.height<1||s.width>1920||s.height>1920) throw std::invalid_argument("Invalid manifest dimensions");
        const auto* regions=(*table)["regions"].as_array();
        if(!regions||regions->size()>64) throw std::invalid_argument("Sample requires regions array (empty for negative images)");
        for(const auto& region:*regions) {
            const auto* r=region.as_table();if(!r) throw std::invalid_argument("Region must be a table");
            const auto* box=(*r)["box"].as_array();if(!box||box->size()!=4) throw std::invalid_argument("Box requires x,y,width,height");
            float values[4];for(std::size_t i=0;i<4;++i) {
                auto value=(*box)[i].value<double>();if(!value) throw std::invalid_argument("Box must be numeric");
                values[i]=static_cast<float>(*value);
            }
            OcrLabel label{{values[0],values[1],values[2],values[3]},required_string((*r)["text"])};
            if(label.box.x+label.box.width>s.width||label.box.y+label.box.height>s.height)
                throw std::invalid_argument("Reference box exceeds image dimensions");
            s.labels.push_back(std::move(label));
        }
        (void)evaluate_ocr(s.labels,{}); // Validate alphabet and geometry before loading any model.
        result.push_back(std::move(s));
    }
    return result;
}
toml::table metrics(const OcrEvaluation& m) {
    toml::table t;
    t.insert("references",static_cast<int64_t>(m.references));t.insert("predictions",static_cast<int64_t>(m.predictions));
    t.insert("matches",static_cast<int64_t>(m.matches));t.insert("exact_matches",static_cast<int64_t>(m.exact_matches));
    t.insert("misses",static_cast<int64_t>(m.references-m.matches));t.insert("extras",static_cast<int64_t>(m.predictions-m.matches));
    t.insert("reference_characters",static_cast<int64_t>(m.reference_characters));
    t.insert("matched_characters",static_cast<int64_t>(m.matched_characters));
    t.insert("matched_edits",static_cast<int64_t>(m.matched_edits));t.insert("spatial_edits",static_cast<int64_t>(m.spatial_edits));
    // Undefined ratios are omitted, never reported as a perfect score.
    if(m.predictions) {t.insert("detection_precision",double(m.matches)/m.predictions);t.insert("exact_precision",double(m.exact_matches)/m.predictions);}
    if(m.references) {t.insert("detection_recall",double(m.matches)/m.references);t.insert("exact_recall",double(m.exact_matches)/m.references);}
    if(m.matched_characters) t.insert("matched_cer",double(m.matched_edits)/m.matched_characters);
    if(m.reference_characters) t.insert("spatial_character_error_ratio",double(m.spatial_edits)/m.reference_characters);
    return t;
}
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2&&args[1]=="--help") {
            std::cout<<"Usage: aegisvision_ocr_evaluate BUNDLE MANIFEST.toml NEW_OUTPUT_DIR\n"
                "       aegisvision_ocr_evaluate --validate-manifest MANIFEST.toml\n"
                "C++ CPU OCR dataset evaluation, fixed axis-aligned IoU >=0.5.\n"
                "Lowercase English alphanumeric labels only; unsupported text is rejected.\n"
                "Writes report.json only after all images succeed; no model downloads.\n";return 0;
        }
        if(args.size()==3&&args[1]=="--validate-manifest") {
            const auto samples=read_manifest(args[2]);
            std::cout<<"Valid labels: "<<samples.size()<<" samples (image contents and models not checked)\n";return 0;
        }
        if(args.size()!=4) throw std::invalid_argument("Use --help for arguments");
        if(fs::exists(args[3])) throw std::invalid_argument("Output directory must not exist");
        const auto samples=read_manifest(args[2]);
        cv::setNumThreads(1);
        PpocrCrnn model(args[1]/"text_detection_en_ppocrv3_2023may.onnx",args[1]/"text_recognition_CRNN_EN_2021sep.onnx");
        toml::array results;OcrEvaluation total;double elapsed_total=0;
        for(const auto& s:samples) {
            if(!fs::is_regular_file(s.image)||fs::file_size(s.image)>32*1024*1024) throw std::invalid_argument("Image missing or exceeds 32 MiB: "+s.id);
            std::ifstream input(s.image,std::ios::binary);std::vector<unsigned char> bytes(static_cast<std::size_t>(fs::file_size(s.image)));
            if(bytes.empty()||!input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()))) throw std::runtime_error("Image read failed: "+s.id);
            const auto image=cv::imdecode(bytes,cv::IMREAD_COLOR);
            if(image.empty()||image.cols!=s.width||image.rows!=s.height) throw std::invalid_argument("Image dimensions differ from manifest: "+s.id);
            const auto start=std::chrono::steady_clock::now();const auto regions=model.read(image);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();elapsed_total+=ms;
            std::vector<OcrLabel> predictions;toml::array texts;
            for(const auto& region:regions) {
                float left=1920,top=1920,right=0,bottom=0;
                for(const auto& p:region.polygon) {left=std::min(left,p.x);top=std::min(top,p.y);right=std::max(right,p.x);bottom=std::max(bottom,p.y);}
                predictions.push_back({{left,top,right-left,bottom-top},region.recognition.text});
                texts.push_back(toml::table{{"text",region.recognition.text},{"box",toml::array{left,top,right-left,bottom-top}}});
            }
            const auto score=evaluate_ocr(s.labels,predictions);total+=score;
            auto row=metrics(score);row.insert("id",s.id);row.insert("elapsed_ms",ms);row.insert("predicted_regions",std::move(texts));
            toml::array labels;for(const auto& label:s.labels) labels.push_back(toml::table{{"text",label.text},
                {"box",toml::array{label.box.x,label.box.y,label.box.width,label.box.height}}});
            row.insert("reference_regions",std::move(labels));results.push_back(std::move(row));
            std::cout<<s.id<<": exact="<<score.exact_matches<<'/'<<score.references<<" extras="<<score.predictions-score.matches<<" ms="<<ms<<'\n';
        }
        toml::table report{{"schema_version",1},{"backend","ppocrv3-crnn-en-opencv-cpu"},
            {"geometry","axis_aligned_box_iou"},{"minimum_iou",.5},{"matching","maximum_cardinality_then_iou_no_text"},
            {"normalization","none_lowercase_alphanumeric_labels_required"},{"sample_count",static_cast<int64_t>(samples.size())},
            {"inference_total_ms",elapsed_total},{"timing","one_pass_no_warmup_excludes_model_load_and_image_decode"},
            {"summary",metrics(total)},{"samples",std::move(results)}};
        if(!fs::create_directories(args[3])) throw std::runtime_error("Cannot create new output directory");
        std::ofstream output(args[3]/"report.json");output<<toml::json_formatter{report}<<'\n';output.close();
        if(!output) throw std::runtime_error("Report write failed");
        return 0; // Execution success, not an accuracy acceptance gate.
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv,argv+argc));
}
