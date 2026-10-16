#include "aegisvision/image_queue_http.hpp"
#include <iostream>
#include <thread>
using namespace aegisvision;
using Json=nlohmann::json;
void check(bool value) {if(!value) throw std::runtime_error("Image HTTP assertion failed");}
struct Fake final : ImageQueueBackend {
    Json status() const override {return {{"enabled",true}};}
    Json submit(const std::string& id,const std::string&) const override {return {{"id",id},{"state","queued"}};}
    Json get(const std::string& id) const override {
        if(id=="missing") throw ImageQueueError(404,"not found");
        if(id=="offline") throw ImageQueueError(503,"unavailable");
        return {{"id",id},{"state","succeeded"}};
    }
    Json cancel(const std::string& id) const override {return {{"id",id},{"state","cancelled"}};}
    std::vector<unsigned char> preview(const std::string&) const override {return {0xff,0xd8,0xff,0xd9};}
};
int main() {
    try {
        for(bool enabled:{false,true}) {
            httplib::Server server;
            server.set_payload_max_length(8192);
            register_image_queue_routes(server,enabled?std::make_shared<Fake>():nullptr);
            const int port=server.bind_to_any_port("127.0.0.1");check(port>0);
            std::thread thread([&]{server.listen_after_bind();});
            try {
                httplib::Client client("127.0.0.1",port);client.set_read_timeout(3);client.set_connection_timeout(3);
                const auto health=client.Get("/api/redis-images");check(health && health->status==200);
                check(Json::parse(health->body).at("enabled")==enabled);
                auto post=[&](const std::string& body){return client.Post("/api/redis-images",body,"application/json");};
                check(post(R"({"id":"one","image":"bus.jpg"})")->status==(enabled?202:503));
                check(client.Post("/api/redis-images", "{}", "text/plain")->status==415);
                check(client.Get("/api/redis-images",{{"Origin","http://evil.test"}})->status==403);
                check(client.Post("/api/redis-images",{{"Origin","http://evil.test"}},"{}","application/json")->status==403);
                check(client.Get("/api/redis-images",{{"Host","evil.test"}})->status==403);
                if(enabled) {
                    for(const auto* invalid:{"[]","{",R"({"id":"x","image":"bus.jpg","extra":1})",R"({"id":"../x","image":"bus.jpg"})"})
                        check(post(invalid)->status==400);
                    check(post(std::string(8193,'x'))->status==413);
                    check(client.Get("/api/redis-images/missing")->status==404);
                    check(client.Get("/api/redis-images/offline")->status==503);
                    const auto preview=client.Get("/api/redis-images/one/preview.jpg");
                    check(preview->status==200 && preview->get_header_value("Content-Type")=="image/jpeg");
                    check(client.Post("/api/redis-images/one/cancel","{}","application/json")->status==200);
                    check(client.Post("/api/redis-images/one/cancel","[]","application/json")->status==400);
                }
            } catch(...) {server.stop();thread.join();throw;}
            server.stop();thread.join();
        }
        std::cout<<"Image HTTP: enabled/disabled, submit/status/cancel/preview, input bounds, Host/Origin and errors passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
