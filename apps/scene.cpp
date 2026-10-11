#include "aegisvision/vlm.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <charconv>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iostream>
namespace fs=std::filesystem;
int run(const std::vector<fs::path>& args) {
    try {
        if(args.size()==2&&args[1]=="--help") {std::cout<<"Usage: aegisvision_scene IMAGE NEW_OUTPUT_DIR [PORT]\nLocal SmolVLM caption, unverified model output; no cloud requests.\n";return 0;}
        if(args.size()!=3&&args.size()!=4) throw std::invalid_argument("Use --help for arguments");
        if(fs::exists(args[2])) throw std::invalid_argument("Output directory must not exist");
        aegisvision::vision::VlmConfig config;
        if(args.size()==4) {auto s=args[3].string();auto [end,ec]=std::from_chars(s.data(),s.data()+s.size(),config.port);
            if(ec!=std::errc{}||end!=s.data()+s.size()) throw std::invalid_argument("Invalid port");}
        if(!fs::is_regular_file(args[1])||fs::file_size(args[1])>32*1024*1024) throw std::invalid_argument("Image must be <=32 MiB");
        std::ifstream input(args[1],std::ios::binary);std::vector<unsigned char> bytes(static_cast<std::size_t>(fs::file_size(args[1])));
        if(bytes.empty()||!input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()))) throw std::runtime_error("Cannot read image");
        auto image=cv::imdecode(bytes,cv::IMREAD_COLOR);aegisvision::vision::LocalVlm model(config);
        const auto start=std::chrono::steady_clock::now();const auto scene=model.describe(image);
        const nlohmann::json result={{"schema_version",1},{"backend","local-smolvlm-500m"},{"verified",false},
            {"warning","Generated interpretation; may hallucinate. Not detection/OCR ground truth."},
            {"summary",scene.summary},{"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()}};
        if(!fs::create_directories(args[2])) throw std::runtime_error("Cannot create output directory");
        std::ofstream output(args[2]/"scene.json");output<<result.dump(2)<<'\n';output.close();if(!output) throw std::runtime_error("Cannot write scene result");
        std::cout<<result.dump(2)<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv,argv+argc));
}
