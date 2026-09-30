#include "aegisvision/clip.hpp"
#include "aegisvision/qdrant.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <cmath>
#include <iostream>

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
                "  init                           Create collection if missing (never replace).\n"
                "  index ID IMAGE [x1 y1 x2 y2]    Upsert image or detection crop.\n"
                "  text QUERY [LIMIT]             Text-to-image search.\n"
                "  image IMAGE [LIMIT]            Image-to-image search.\n"
                "Local Qdrant only (127.0.0.1); commands emit JSON. LIMIT defaults to 5.\n";
            return 0;
        }
        if(args.size()<5) throw std::invalid_argument("Missing arguments; run --help");
        const auto command=args[4].string();
        if(!((command=="init" && args.size()==5) || (command=="index" && (args.size()==7 || args.size()==11)) ||
             ((command=="text" || command=="image") && (args.size()==6 || args.size()==7))))
            throw std::invalid_argument("Invalid command arguments; run --help");
        aegisvision::QdrantConfig config;
        config.port=integer(args[2]); config.collection=utf8(args[3]);
        const int limit=(command=="text" || command=="image") && args.size()==7 ? integer(args[6]) : 5;
        if(limit<1 || limit>100) throw std::invalid_argument("Limit must be 1..100");
        aegisvision::ClipEmbedder clip(args[1]);
        config.embedding_space=clip.space_id();
        aegisvision::QdrantVectorStore store(config,command=="init");
        if(command=="init") {
            std::cout << nlohmann::json{{"collection",config.collection},{"space_id",clip.space_id()}}.dump(2) << '\n'; return 0;
        }
        std::vector<float> vector;
        std::map<std::string,std::string> metadata;
        if(command=="text") vector=clip.embed_text(utf8(args[5]));
        else {
            const auto path=command=="index" ? args[6] : args[5];
            const auto image=aegisvision::vision::load_image(path);
            aegisvision::BoundingBox box{0,0,static_cast<float>(image.cols),static_cast<float>(image.rows)};
            if(command=="index" && args.size()==11) box={coordinate(args[7]),coordinate(args[8]),coordinate(args[9]),coordinate(args[10])};
            if(box.x1<0 || box.y1<0 || box.x2>image.cols || box.y2>image.rows || box.area()<=0)
                throw std::invalid_argument("Crop must be nonempty and inside the image");
            vector=clip.embed_image(aegisvision::vision::image_frame(image,"search","local"),{box,"image",1,{}, {}});
            metadata={{"path",utf8(std::filesystem::absolute(path))},{"bbox",nlohmann::json::array({box.x1,box.y1,box.x2,box.y2}).dump()}};
        }
        if(command=="index") {
            store.upsert(utf8(args[5]),std::move(vector),metadata);
            std::cout << nlohmann::json{{"indexed",utf8(args[5])},{"metadata",metadata}}.dump(2) << '\n'; return 0;
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
