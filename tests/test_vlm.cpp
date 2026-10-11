#include "aegisvision/vlm.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <stdexcept>
using namespace aegisvision::vision;using Json=nlohmann::json;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"Expected VLM failure");}
Json envelope(Json content){return {{"model","aegis-smolvlm-500m"},{"choices",Json::array({{{"finish_reason","stop"},{"message",{{"content",content}}}}})}};}
int main(){try{
    const auto good=envelope("A camera on a table.");
    require(parse_scene_response(good.dump()).summary=="A camera on a table.","Valid scene rejected");
    auto bad=good;bad["choices"][0]["finish_reason"]="length";rejects([&]{parse_scene_response(bad.dump());});
    bad=good;bad["model"]="other";rejects([&]{parse_scene_response(bad.dump());});
    for(const auto& scene:std::vector<Json>{"",42,Json::object(),"bad\ntext",std::string(513,'x')}) rejects([&]{parse_scene_response(envelope(scene).dump());});
    require(parse_scene_response(envelope("  Türkçe açıklama.\n").dump()).summary=="Türkçe açıklama.","UTF-8/trim mismatch");
    bad=good;bad["choices"][0]["message"]["tool_calls"]={{{"name","execute"}}};rejects([&]{parse_scene_response(bad.dump());});
    rejects([]{parse_scene_response(std::string(65537,'x'));});rejects([]{LocalVlm({0,100});});
    rejects([]{parse_scene_response(std::string(32,'[')+"0"+std::string(32,']'));});
    rejects([]{parse_scene_response(envelope("bad\u202etext").dump());});
    rejects([]{LocalVlm({8092,99});});rejects([]{LocalVlm({8092,120001});});
    httplib::Server server;std::atomic_int mode{0};std::atomic_bool started{false};
    server.Post("/v1/chat/completions",[&](const auto& req,auto& res){
        const auto body=Json::parse(req.body);
        if(body.at("model")!="aegis-smolvlm-500m"||body.at("stream")!=false||!body["messages"][0]["content"][0]["image_url"]["url"].get<std::string>().starts_with("data:image/jpeg;base64,/9j/")){res.status=400;return;}
        started=true;
        if(mode==1){res.status=503;return;}
        if(mode==2){res.set_content(std::string(65537,'x'),"application/json");return;}
        if(mode==3)std::this_thread::sleep_for(std::chrono::milliseconds(800));
        res.set_content(good.dump(),"application/json");
    });
    int port=server.bind_to_any_port("127.0.0.1");require(port>0,"Cannot bind fixture");
    std::thread thread([&]{server.listen_after_bind();});
    struct Guard{httplib::Server& s;std::thread& t;~Guard(){s.stop();t.join();}} guard{server,thread};
    server.wait_until_ready();LocalVlm model({port,1500});cv::Mat image(64,64,CV_8UC3,cv::Scalar(255,255,255));
    require(model.describe(image).summary=="A camera on a table.","HTTP scene mismatch");
    std::atomic_bool cancelled{true};rejects([&]{model.describe(image,&cancelled);});rejects([&]{model.describe(cv::Mat());});
    mode=1;rejects([&]{model.describe(image);});mode=2;rejects([&]{model.describe(image);});
    mode=3;LocalVlm quick({port,150});auto start=std::chrono::steady_clock::now();rejects([&]{quick.describe(image);});
    require(std::chrono::steady_clock::now()-start<std::chrono::seconds(2),"Deadline did not interrupt request");
    cancelled=false;started=false;std::jthread cancel_thread([&](std::stop_token stop){while(!started&&!stop.stop_requested())std::this_thread::sleep_for(std::chrono::milliseconds(5));cancelled=true;});
    rejects([&]{model.describe(image,&cancelled);});cancel_thread.request_stop();cancel_thread.join();
    mode=0;require(model.describe(image).summary=="A camera on a table.","Recovery after cancellation failed");
    std::cout<<"VLM contract, loopback transport, bounds, timeout and cancellation passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
