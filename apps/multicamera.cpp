#include "aegisvision/multicamera.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
void fields(const Json& value,std::initializer_list<const char*> names) {
    if(!value.is_object() || value.size()!=names.size()) throw std::invalid_argument("Unexpected or missing fields");
    for(auto name:names) if(!value.contains(name)) throw std::invalid_argument("Missing field");
}
std::int64_t number(const Json& value) {
    if(!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>()>
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) throw std::invalid_argument("Expected bounded integer, not float");
    const auto result=value.get<std::int64_t>();if(result<0)throw std::invalid_argument("Expected nonnegative integer");return result;
}
Json key(const aegisvision::CameraTracklet& t) {return {{"camera",t.camera},{"epoch",t.epoch},{"local_id",t.local_id}};}
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout<<"Usage: aegisvision_multicamera INPUT.json NEW_OUTPUT_DIRECTORY\n"
                "Offline closed-tracklet association only. 512-D same-space person features, synchronized timestamps, explicit directed topology.\n"
                "No camera capture, clock sync or biometric identity guarantee. See docs/multicamera.md.\n";return 0;
        }
        if(args.size()!=3)throw std::invalid_argument("Run --help for usage");
        if(!fs::is_regular_file(args[1]) || fs::file_size(args[1])>16*1024*1024) throw std::invalid_argument("Input must be a regular JSON file <=16 MiB");
        if(fs::exists(args[2]))throw std::invalid_argument("Output directory already exists; nothing overwritten");
        std::ifstream input(args[1]);if(!input)throw std::runtime_error("Cannot open input");
        const auto data=Json::parse(input,[](int depth,Json::parse_event_t,Json&) {
            if(depth>16)throw std::invalid_argument("JSON nesting limit exceeded");return true;
        });
        fields(data,{"version","time_basis","config","tracklets"});
        if(number(data.at("version"))!=1 || data.at("time_basis")!="synchronized_ms")
            throw std::invalid_argument("Expected v1 synchronized_ms input; raw camera/arrival clocks are not interchangeable");
        const auto& cfg=data.at("config");
        fields(cfg,{"cameras","transitions","embedding_space","min_similarity","ambiguity_margin"});
        if(!cfg.at("cameras").is_array() || cfg.at("cameras").size()>16 ||
           !cfg.at("transitions").is_array() || cfg.at("transitions").size()>240 ||
           !data.at("tracklets").is_array() || data.at("tracklets").size()>512)
            throw std::invalid_argument("Camera/transition/tracklet bound exceeded");
        aegisvision::MultiCameraConfig c;
        c.cameras=cfg.at("cameras").get<std::vector<std::string>>();
        c.embedding_space=cfg.at("embedding_space").get<std::string>();
        c.min_similarity=cfg.at("min_similarity").get<double>();c.ambiguity_margin=cfg.at("ambiguity_margin").get<double>();
        for(const auto& e:cfg.at("transitions")) {
            fields(e,{"from","to","min_gap_ms","max_gap_ms"});
            c.transitions.push_back({e.at("from").get<std::string>(),e.at("to").get<std::string>(),number(e.at("min_gap_ms")),number(e.at("max_gap_ms"))});
        }
        std::vector<aegisvision::CameraTracklet> tracks;
        for(const auto& t:data.at("tracklets")) {
            fields(t,{"camera","epoch","local_id","start_ms","end_ms","embedding_space","embedding"});
            if(!t.at("embedding").is_array() || t.at("embedding").size()!=512)throw std::invalid_argument("Expected 512-D person prototype");
            tracks.push_back({t.at("camera").get<std::string>(),static_cast<std::uint64_t>(number(t.at("epoch"))),
                static_cast<std::uint64_t>(number(t.at("local_id"))),number(t.at("start_ms")),number(t.at("end_ms")),
                t.at("embedding_space").get<std::string>(),t.at("embedding").get<std::vector<float>>()});
        }
        const auto decisions=aegisvision::associate_cameras(tracks,c);
        auto items=Json::array();std::set<std::uint64_t> identities;std::size_t matches=0;
        for(std::size_t i=0;i<tracks.size();++i) {
            const auto& d=decisions[i];auto item=key(tracks[i]);identities.insert(d.global_id);
            item["global_id"]=d.global_id;item["reason"]=d.reason;
            item["start_ms"]=tracks[i].start_ms;item["end_ms"]=tracks[i].end_ms;
            item["predecessor"]=d.predecessor ? key(tracks[*d.predecessor]) : Json(nullptr);
            item["best_similarity"]=d.best_similarity ? Json(*d.best_similarity) : Json(nullptr);
            if(d.predecessor)++matches;items.push_back(std::move(item));
        }
        Json report{{"schema_version",1},{"algorithm","mutual-best-handoff-v1"},{"mode","offline_closed_tracklets"},
            {"time_basis","synchronized_ms"},{"config",cfg},{"tracklets",items},
            {"accepted_handoffs",matches},{"identity_hypotheses",identities.size()},
            {"verified_person_identities",false},{"warning","Batch-local hypotheses; not unique people, live multicamera validation or calibrated accuracy."}};
        if(!fs::create_directory(args[2]))throw std::runtime_error("Cannot create new output directory");
        std::ofstream output(args[2]/"report.json");output<<report.dump(2)<<'\n';output.close();
        if(!output)throw std::runtime_error("Cannot write report");
        std::cout<<"Offline association: "<<tracks.size()<<" tracklets, "<<matches<<" accepted handoffs, "<<identities.size()<<" identity hypotheses. Not verified people.\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
