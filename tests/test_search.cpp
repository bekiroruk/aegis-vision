#include "aegisvision/clip.hpp"
#include "aegisvision/qdrant.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <thread>
#include <cmath>

namespace {
using namespace aegisvision;
using Json=nlohmann::json;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f,const char* message) {
    try { f(); } catch(const std::exception&) { return; } throw std::runtime_error(message);
}
double cosine(const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"Vector dimension mismatch"); double sum=0;
    for(std::size_t i=0;i<a.size();++i) sum+=static_cast<double>(a[i])*b[i]; return sum;
}
void unit() {
    const auto normalized=normalize_embedding({3,4});
    require(std::abs(normalized[0]-.6)<1e-6,"Normalization failed");
    rejects([] { (void)normalize_embedding({0,0}); },"Zero vector accepted");
    rejects([] { (void)normalize_embedding({std::numeric_limits<float>::quiet_NaN()}); },"NaN accepted");
    const auto id=qdrant_point_id("space","item");
    require(id==qdrant_point_id("space","item") && id!=qdrant_point_id("other","item") && id.size()==36 && id[14]=='8',"Stable UUID failed");
    QdrantConfig invalid; invalid.host="example.com"; invalid.embedding_space="test";
    rejects([&] { QdrantVectorStore store(invalid); },"Remote HTTP unexpectedly allowed");
    invalid.host="127.0.0.1"; invalid.collection="../escape";
    rejects([&] { QdrantVectorStore store(invalid); },"Invalid collection name accepted");

    httplib::Server server;
    server.Get("/collections/test",[](const auto&,auto& response) {
        response.set_content(R"({"status":"ok","result":{"config":{"params":{"vectors":{"size":2,"distance":"Cosine"}}}}})","application/json");
    });
    server.Put("/collections/test/points",[](const auto& request,auto& response) {
        const auto point=Json::parse(request.body).at("points").at(0);
        const auto valid=point.at("payload").at("item_id")=="a\"b" && request.get_param_value("wait")=="true" && point.at("vector").size()==2;
        response.status=valid ? 200 : 400;
        response.set_content(R"({"status":"ok","result":{"status":"completed"}})","application/json");
    });
    server.Post("/collections/test/points/search",[](const auto& request,auto& response) {
        const auto body=Json::parse(request.body);
        const auto space=body.at("filter").at("must").at(0).at("match").at("value");
        if(body.at("limit")==99) { response.status=503; response.set_content("unavailable","text/plain"); return; }
        response.set_content(Json{{"status","ok"},{"result",Json::array({Json{{"score",1.0},{"payload",{{"item_id","a\"b"},{"space_id",space},{"metadata",{{"label","bus"}}}}}}})}}.dump(),"application/json");
    });
    const int port=server.bind_to_any_port("127.0.0.1"); require(port>0,"Cannot bind test HTTP server");
    std::thread worker([&] { server.listen_after_bind(); });
    struct Guard { httplib::Server& server; std::thread& worker; ~Guard() { server.stop(); worker.join(); } } guard{server,worker};
    server.wait_until_ready();
    require(server.is_running(),"HTTP fixture not ready");
    QdrantConfig config{"127.0.0.1",port,"test",2,"test-space",2};
    QdrantVectorStore store(config);
    store.upsert("a\"b",{3,4},{{"label","bus"}});
    const auto found=store.search({3,4},1);
    require(found.size()==1 && found[0].item_id=="a\"b" && found[0].metadata.at("label")=="bus","Search response invalid");
    rejects([&] { (void)store.search({1},1); },"Wrong dimension accepted");
    rejects([&] { (void)store.search({1,0},0); },"Zero limit accepted");
    rejects([&] { (void)store.search({1,0},99); },"HTTP failure swallowed");
    config.dimension=3;
    rejects([&] { QdrantVectorStore bad(config); },"Incompatible collection accepted");
}
void reference(const std::filesystem::path& bundle) {
    std::ifstream file(bundle/"reference.json"); const auto fixtures=Json::parse(file);
    ClipTokenizer tokenizer(bundle/"tokenizer.json"); ClipEmbedder clip(bundle);
    double minimum=1;
    for(const auto& fixture:fixtures.at("texts")) {
        const auto prompt=fixture.at("text").get<std::string>();
        const auto tokens=tokenizer.encode(prompt);
        require(Json(tokens.ids)==fixture.at("ids"), ("Tokenizer differs from HuggingFace: "+prompt).c_str());
        require(Json(tokens.mask)==fixture.at("mask"),"Attention mask differs from HuggingFace");
        const auto similarity=cosine(clip.embed_text(prompt),fixture.at("embedding").get<std::vector<float>>());
        minimum=std::min(minimum,similarity); require(similarity>.9999,"Text embedding differs from PyTorch");
    }
    for(const auto& fixture:fixtures.at("images")) {
        const auto image=vision::load_image(bundle/fixture.at("file").get<std::string>());
        const auto frame=vision::image_frame(image,"fixture","local");
        const Detection detection{{0,0,static_cast<float>(image.cols),static_cast<float>(image.rows)},"image",1,{}, {}};
        const auto similarity=cosine(clip.embed_image(frame,detection),fixture.at("embedding").get<std::vector<float>>());
        minimum=std::min(minimum,similarity); require(similarity>.9999,"Image embedding differs from PyTorch");
        const BoundingBox crop_box{0,0,static_cast<float>(image.cols/2),static_cast<float>(image.rows/2)};
        const auto part=vision::crop(image,crop_box);
        const auto cropped=clip.embed_image(frame,{crop_box,"crop",1,{}, {}});
        const auto full_part=clip.embed_image(vision::image_frame(part,"part","local"),
            {{0,0,static_cast<float>(part.cols),static_cast<float>(part.rows)},"crop",1,{}, {}});
        require(cosine(cropped,full_part)>.99999,"Detection crop differs from standalone image");
    }
    rejects([&] { (void)tokenizer.encode("  \t\n"); },"Empty query accepted");
    rejects([&] { (void)tokenizer.encode(std::string(500,'x')); },"Overlong query accepted");
    rejects([&] { (void)tokenizer.encode("<|endoftext|>"); },"Special token accepted");
    rejects([&] { (void)tokenizer.encode("<|ENDOFTEXT|>"); },"Normalized special token accepted");
    rejects([&] { (void)tokenizer.encode(std::string("\xff",1)); },"Invalid UTF-8 accepted");
    std::cout << std::setprecision(10) << "Reference texts=" << fixtures.at("texts").size() << " images=" << fixtures.at("images").size() << " minimum_cosine=" << minimum << '\n';
}
void tokenizer_reference(const std::filesystem::path& tokenizer_file,const std::filesystem::path& fixture_file) {
    std::ifstream file(fixture_file); const auto fixtures=Json::parse(file);
    ClipTokenizer tokenizer(tokenizer_file);
    for(const auto& fixture:fixtures) {
        const auto tokens=tokenizer.encode(fixture.at("text").get<std::string>());
        const auto ids=fixture.at("ids").get<std::vector<std::int64_t>>();
        for(std::size_t i=0;i<77;++i) {
            require(tokens.ids[i]==(i<ids.size() ? ids[i] : 49407),"Tokenizer golden ID mismatch");
            require(tokens.mask[i]==(i<ids.size() ? 1 : 0),"Tokenizer golden mask mismatch");
        }
    }
    std::cout << "Tokenizer golden cases=" << fixtures.size() << '\n';
}
void live(int port,const std::string& collection,bool write) {
    QdrantConfig config{"127.0.0.1",port,collection,3,"integration-v1",5};
    QdrantVectorStore store(config,write);
    if(write) {
        store.upsert("bus",{1,0,0},{{"label","old"}});
        store.upsert("bus",{1,0,0},{{"label","bus"}});
        store.upsert("fruit",{0,1,0},{{"label","fruit"}});
        config.embedding_space="integration-other";
        QdrantVectorStore other(config);
        other.upsert("bus",{1,0,0},{{"label","must-not-leak"}});
    }
    const auto results=store.search({1,0,0},10);
    require(results.size()==2 && results[0].item_id=="bus" && results[0].metadata.at("label")=="bus","Persistent search/idempotency/space isolation failed");
    require(results[0].score>.999,"Cosine score mismatch");
    std::cout << "Qdrant persistence/idempotency/space isolation passed\n";
}
}
int main(int argc,char* argv[]) {
    try {
        if(argc==4 && std::string(argv[1])=="--tokenizer") tokenizer_reference(argv[2],argv[3]);
        else if(argc==4 && (std::string(argv[1])=="--qdrant-write" || std::string(argv[1])=="--qdrant-read"))
            live(std::stoi(argv[2]),argv[3],std::string(argv[1])=="--qdrant-write");
        else { unit(); if(argc==2) reference(argv[1]); }
        std::cout << "Search tests passed\n"; return 0;
    } catch(const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
