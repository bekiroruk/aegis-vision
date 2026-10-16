#include "aegisvision/redis_image_queue.hpp"
#include <iostream>
int main(int argc,char* argv[]) {
    try {
        if(argc!=6) throw std::invalid_argument("MODEL MEDIA REDIS_PORT QUEUE WEB");
        auto backend=aegisvision::make_redis_image_queue(argv[1],argv[2],std::stoi(argv[3]),argv[4]);
        httplib::Server server;
        server.set_payload_max_length(8192);
        server.set_read_timeout(5);server.set_write_timeout(10);
        server.set_default_headers({{"Content-Security-Policy","default-src 'self'; img-src 'self'; frame-ancestors 'none'"}});
        aegisvision::register_image_queue_routes(server,backend);
        if(!server.set_mount_point("/",argv[5])) throw std::runtime_error("Missing web directory");
        const int port=server.bind_to_any_port("127.0.0.1");
        if(port<=0) throw std::runtime_error("Bind failed");
        std::cout<<port<<std::endl;
        return server.listen_after_bind()?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
