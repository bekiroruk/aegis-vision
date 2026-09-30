#include "aegisvision/application_config.hpp"
#include "aegisvision/clip.hpp"
#include "aegisvision/qdrant.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <cmath>
#include <iostream>
#include <optional>

namespace {
std::string utf8(const std::filesystem::path& path) {
    const auto text=path.u8string(); return {text.begin(),text.end()};
}
int integer(const std::filesystem::path& argument) {
    const auto text=argument.string(); int value=0;
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("Expected an integer");
    return value;
}
float coordinate(const std::filesystem::path& argument) {
    const auto text=argument.string(); float value=0;
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size() || !std::isfinite(value)) throw std::invalid_argument("Invalid crop coordinate");
    return value;
}
int run(const std::vector<std::filesystem::path>& args) {
    try {
        if(args.size()==2 && args[1]=="--help") {
            std::cout << "Usage: aegisvision_search_cli BUNDLE PORT COLLECTION COMMAND [ARGS]\n"
                "   or: aegisvision_search_cli --config CONFIG.toml COMMAND [ARGS]\n"
                "  init                           Create collection if missing (never replace).\n"
                "  index ID IMAGE [x1 y1 x2 y2]    Upsert image or detection crop.\n"
                "  text QUERY [LIMIT]             Text-to-image search.\n"
                "  image IMAGE [LIMIT]            Image-to-image search.\n"
                "Local Qdrant only (127.0.0.1); commands emit JSON. LIMIT defaults to 5.\n";
            return 0;
        }
        std::vector<std::filesystem::path> parsed=args;
        std::optional<aegisvision::vision::ApplicationSettings> settings;
        if(args.size()>1 && args[1]=="--config") {
            if(args.size()<4) throw std::invalid_argument("Expected --config CONFIG.toml COMMAND [ARGS]");
            settings=aegisvision::vision::load_application_settings(args[2]);
            if(settings->mode!=aegisvision::vision::ApplicationMode::Search)
                throw std::invalid_argument("Search CLI requires pipeline.mode=search");
            parsed={args[0],settings->clip_bundle,std::to_string(settings->qdrant.port),settings->qdrant.collection};
            parsed.insert(parsed.end(),args.begin()+3,args.end());
        }
        if(parsed.size()<5) throw std::invalid_argument("Missing arguments; run --help");
        const auto command=parsed[4].string();
        if(!((command=="init" && parsed.size()==5) || (command=="index" && (parsed.size()==7 || parsed.size()==11)) ||
             ((command=="text" || command=="image") && (parsed.size()==6 || parsed.size()==7))))
            throw std::invalid_argument("Invalid command arguments; run --help");
        aegisvision::QdrantConfig config=settings ? settings->qdrant : aegisvision::QdrantConfig{};
        config.port=integer(parsed[2]); config.collection=utf8(parsed[3]);
        const int limit=(command=="text" || command=="image") && parsed.size()==7 ? integer(parsed[6]) : 5;
        if(limit<1 || limit>100) throw std::invalid_argument("Limit must be 1..100");
        aegisvision::ClipEmbedder clip(parsed[1]);
        config.embedding_space=clip.space_id();
        aegisvision::QdrantVectorStore store(config,command=="init");
        if(command=="init") {
            std::cout << nlohmann::json{{"collection",config.collection},{"space_id",clip.space_id()}}.dump(2) << '\n'; return 0;
        }
        std::vector<float> vector;
        std::map<std::string,std::string> metadata;
        if(command=="text") vector=clip.embed_text(utf8(parsed[5]));
        else {
            const auto path=command=="index" ? parsed[6] : parsed[5];
            const auto image=aegisvision::vision::load_image(path);
            aegisvision::BoundingBox box{0,0,static_cast<float>(image.cols),static_cast<float>(image.rows)};
            if(command=="index" && parsed.size()==11) box={coordinate(parsed[7]),coordinate(parsed[8]),coordinate(parsed[9]),coordinate(parsed[10])};
            if(box.x1<0 || box.y1<0 || box.x2>image.cols || box.y2>image.rows || box.area()<=0)
                throw std::invalid_argument("Crop must be nonempty and inside the image");
            vector=clip.embed_image(aegisvision::vision::image_frame(image,"search","local"),{box,"image",1,{}, {}});
            metadata={{"path",utf8(std::filesystem::absolute(path))},{"bbox",nlohmann::json::array({box.x1,box.y1,box.x2,box.y2}).dump()}};
        }
        if(command=="index") {
            store.upsert(utf8(parsed[5]),std::move(vector),metadata);
            std::cout << nlohmann::json{{"indexed",utf8(parsed[5])},{"metadata",metadata}}.dump(2) << '\n'; return 0;
        }
        auto results=nlohmann::json::array();
        for(const auto& result : store.search(vector,static_cast<std::size_t>(limit)))
            results.push_back({{"id",result.item_id},{"score",result.score},{"metadata",result.metadata}});
        std::cout << nlohmann::json{{"results",results},{"space_id",clip.space_id()}}.dump(2) << '\n';
        return 0;
    } catch(const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc,wchar_t* argv[]) {
#else
int main(int argc,char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv,argv+argc));
}
