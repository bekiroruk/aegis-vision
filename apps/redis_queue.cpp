#include "aegisvision/redis_queue.hpp"
#include <charconv>
#include <iostream>
#include <string>
int main(int argc,char* argv[]) {
    try {
        if(argc==2 && std::string(argv[1])=="--help") {
            std::cout << "Usage: aegisvision_redis_queue PORT QUEUE CONTEXT < command.json\n"
                "One bounded coordination command, loopback Redis only. Not a CV worker or web backend.\n";
            return 0;
        }
        if(argc!=4) throw std::invalid_argument("Use --help");
        const std::string text=argv[1];int port{};
        const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),port);
        if(error!=std::errc{}||end!=text.data()+text.size()) throw std::invalid_argument("Invalid port");
        std::string input;char c;
        while(std::cin.get(c)) { if(input.size()>=131072) throw std::invalid_argument("Command too large");input+=c; }
        const auto command=nlohmann::json::parse(input,[](int depth,nlohmann::json::parse_event_t,const nlohmann::json&) {
            if(depth>16) throw std::invalid_argument("Command nesting too deep");return true;
        });
        aegisvision::RedisLeaseQueue queue(port,argv[2],argv[3]);
        std::cout << queue.call(command).dump() << '\n'; return 0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
