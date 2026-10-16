#include "aegisvision/redis_queue.hpp"
#include "redis_script.hpp"
#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/time.h>
#endif
#include <hiredis.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace aegisvision {
namespace {
bool identifier(const std::string& s) {
    if(s.empty() || s.size()>64) return false;
    for(unsigned char c:s) if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_')) return false;
    return true;
}
void require(bool ok) { if(!ok) throw std::invalid_argument("Invalid Redis queue contract"); }
}
RedisLeaseQueue::RedisLeaseQueue(int port, std::string queue, std::string context)
    : port_(port), queue_(std::move(queue)), context_(std::move(context)) {
    require(port>0 && port<=65535 && identifier(queue_) && !context_.empty() && context_.size()<=256);
}
void RedisLeaseQueue::validate(const nlohmann::json& c) {
    require(c.is_object() && c.contains("op") && c.at("op").is_string());
    const auto op=c.at("op").get<std::string>();
    std::set<std::string> keys{"op"};
    if(op=="claim") keys.insert({"owner","lease_ms"});
    else if(op=="submit") keys.insert({"id","payload"});
    else if(op=="get" || op=="cancel") keys.insert("id");
    else if(op=="renew") keys.insert({"id","owner","token","lease_ms"});
    else if(op=="complete" || op=="fail") keys.insert({"id","owner","token","result"});
    else throw std::invalid_argument("Unknown Redis queue operation");
    require(c.size()==keys.size());
    for(const auto& key:keys) require(c.contains(key));
    for(const auto* name:{"id","owner"}) if(c.contains(name))
        require(c.at(name).is_string() && identifier(c.at(name).get<std::string>()));
    if(c.contains("token")) {
        require(c.at("token").is_string());
        const auto s=c.at("token").get<std::string>();
        require(!s.empty() && s.size()<=13 && s.find_first_not_of("0123456789")==std::string::npos);
    }
    if(c.contains("lease_ms")) {
        const auto& n=c.at("lease_ms");
        require(n.is_number_integer() && n>=100 && n<=300000);
    }
    for(const auto* name:{"payload","result"}) if(c.contains(name)) {
        require(c.at(name).is_object());
        require(c.at(name).dump().size() <= (std::string(name)=="payload" ? 8192U : 65536U));
    }
}
nlohmann::json RedisLeaseQueue::call(const nlohmann::json& command) const {
    validate(command);
    const timeval timeout{1,0};
    std::unique_ptr<redisContext,decltype(&redisFree)> connection(redisConnectWithTimeout("127.0.0.1",port_,timeout),redisFree);
    if(!connection || connection->err || redisSetTimeout(connection.get(),timeout)!=REDIS_OK)
        throw std::runtime_error("Redis connection unavailable; no write retried");
    const std::string prefix="aegis:{"+queue_+"}:";
    const std::vector<std::string> args{"EVAL",aegis_redis_script,"4",prefix+"jobs",prefix+"pending",
        prefix+"leases",prefix+"meta",context_,command.dump(),
        command.contains("payload") ? command.at("payload").dump() : ""};
    std::vector<const char*> argv; std::vector<std::size_t> lengths;
    for(const auto& a:args) {argv.push_back(a.data()); lengths.push_back(a.size());}
    std::unique_ptr<redisReply,decltype(&freeReplyObject)> reply(
        static_cast<redisReply*>(redisCommandArgv(connection.get(),static_cast<int>(argv.size()),argv.data(),lengths.data())),freeReplyObject);
    if(!reply) throw std::runtime_error("Redis reply lost: operation outcome unknown; no write retried");
    if(reply->type==REDIS_REPLY_ERROR)
        throw std::runtime_error("Redis queue rejected operation: "+std::string(reply->str,reply->len));
    if(reply->type!=REDIS_REPLY_STRING || reply->len>131072)
        throw std::runtime_error("Invalid Redis queue reply");
    return nlohmann::json::parse(reply->str,reply->str+reply->len);
}
}
