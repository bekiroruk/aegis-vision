#include "aegisvision/redis_queue.hpp"
#include <iostream>
#include <stdexcept>
using Json=nlohmann::json;
using aegisvision::RedisLeaseQueue;
template<class F> void rejects(F f) {
    try {f();} catch(const std::invalid_argument&) {return;}
    throw std::runtime_error("Invalid Redis input accepted");
}
int main() {
    try {
        for(const int port:{0,-1,65536}) rejects([&]{RedisLeaseQueue q(port,"queue","ctx");});
        for(const auto* name:{"","a:b","../queue","{other}"}) rejects([&]{RedisLeaseQueue q(6379,name,"ctx");});
        rejects([]{RedisLeaseQueue q(6379,"queue","");});
        rejects([]{RedisLeaseQueue q(6379,"queue",std::string(257,'a'));});
        for(const Json& c:std::vector<Json>{Json(),Json::array(),{{"op","unknown"}},{{"op","get"}},
                {{"op","get"},{"id","a"},{"extra",true}},{{"op","get"},{"id","../a"}},
                {{"op","workers"},{"owner","w"}},{{"op","worker_ping"}},
                {{"op","worker_ping"},{"owner","../w"}},
                {{"op","worker_ping"},{"owner","w"},{"lease_ms",5000}},
                {{"op","submit"},{"id","a"},{"payload",Json::array()}},
                {{"op","submit"},{"id","a"},{"payload",{{"huge",std::string(8193,'x')}}}},
                {{"op","complete"},{"id","a"},{"owner","w"},{"token","bad"},{"result",Json::object()}}})
            rejects([&]{RedisLeaseQueue::validate(c);});
        for(const Json& lease:std::vector<Json>{true,1.2,99,300001,"1000"})
            rejects([&]{RedisLeaseQueue::validate({{"op","claim"},{"owner","w"},{"lease_ms",lease}});});
        for(const auto* op:{"get","cancel"}) RedisLeaseQueue::validate({{"op",op},{"id","job_1"}});
        RedisLeaseQueue::validate({{"op","workers"}});
        RedisLeaseQueue::validate({{"op","worker_ping"},{"owner","w"}});
        RedisLeaseQueue::validate({{"op","submit"},{"id","j"},{"payload",{{"text","Türkçe"}}}});
        RedisLeaseQueue::validate({{"op","claim"},{"owner","w"},{"lease_ms",30000}});
        RedisLeaseQueue::validate({{"op","renew"},{"id","j"},{"owner","w"},{"token","123"},{"lease_ms",100}});
        for(const auto* op:{"complete","fail"}) RedisLeaseQueue::validate({{"op",op},{"id","j"},
            {"owner","w"},{"token","123"},{"result",Json::object()}});
        std::cout<<"Redis command types, bounds, namespace, ownership and payload contracts passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
