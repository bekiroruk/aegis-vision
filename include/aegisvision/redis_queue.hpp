#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace aegisvision {
// Experimental bounded Redis coordination backend, not the web JobQueue.
// One trusted standalone Redis on loopback, no TLS/auth/cluster/failover support.
// call() never retries an uncertain write; submit with the same ID to reconcile.
class RedisLeaseQueue {
public:
    RedisLeaseQueue(int port, std::string queue, std::string context);
    // submit(id,payload), claim(owner,lease_ms), renew(id,owner,token,lease_ms),
    // complete(id,owner,token,result), fail(id,owner,token,result), cancel(id), get(id).
    // null means unknown/no work/expired or stale ownership; errors throw.
    nlohmann::json call(const nlohmann::json& command) const;
    static void validate(const nlohmann::json& command);
private:
    int port_;
    std::string queue_, context_;
};
}
