#include "aegisvision/redis_image_queue.hpp"
#include "aegisvision/redis_image.hpp"
#include "aegisvision/redis_queue.hpp"
#include <algorithm>

namespace aegisvision {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
class Backend final : public ImageQueueBackend {
    fs::path root_;
    RedisImageWorker image_;
    RedisLeaseQueue queue_;
    std::string name_;
    Json call(const Json& c) const {
        try { return queue_.call(c); }
        catch (const std::invalid_argument&) { throw ImageQueueError(400,"Invalid queue request"); }
        catch (const std::exception& e) {
            const std::string reason=e.what();
            if(reason.find("QUEUE_ID_CONFLICT")!=std::string::npos) throw ImageQueueError(409,"Job ID belongs to different image bytes");
            if(reason.find("QUEUE_FULL")!=std::string::npos) throw ImageQueueError(429,"Queue history limit reached (128 jobs)");
            if(reason.find("QUEUE_CONTEXT_MISMATCH")!=std::string::npos) throw ImageQueueError(503,"Model or media context differs from this queue");
            throw ImageQueueError(503,"Redis unavailable or response uncertain; check the same job ID");
        }
    }
    Json raw(const std::string& id) const {
        auto job=call({{"op","get"},{"id",id}});
        if(job.is_null()) throw ImageQueueError(404,"Image job not found");
        return job;
    }
    static Json public_job(const Json& job) {
        return {{"id",job.at("id")},{"state",job.at("state")},{"attempts",job.at("attempts")},
            {"payload",job.at("payload")},{"result",job.value("result",Json())}};
    }
public:
    Backend(const fs::path& model,const fs::path& media,int port,const std::string& name)
        : root_(fs::canonical(media)),image_(model,root_),queue_(port,name,image_.context()),name_(name) {}
    Json status() const override {
        const auto presence=call({{"op","workers"}});
        std::vector<std::string> names;
        unsigned scanned=0;
        for(const auto& file:fs::directory_iterator(root_)) {
            if(++scanned>1024) break;
            if(file.is_symlink() || !file.is_regular_file()) continue;
            auto ext=file.path().extension().string();
            std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
            if(ext!=".jpg" && ext!=".jpeg" && ext!=".png") continue;
            const auto text=file.path().filename().u8string(); names.emplace_back(text.begin(),text.end());
        }
        std::sort(names.begin(),names.end()); if(names.size()>128) names.resize(128);
        return {{"enabled",true},{"queue",name_},{"context",image_.context()},{"images",names},
            {"backend","opencv-dnn-cpu"},{"worker_status",presence.at("worker_count").get<int>()>0 ? "recent_heartbeat" : "no_recent_heartbeat"},
            {"worker_count",presence.at("worker_count")},{"heartbeat_ttl_ms",presence.at("heartbeat_ttl_ms")}};
    }
    Json submit(const std::string& id,const std::string& file) const override {
        RedisLeaseQueue::validate({{"op","get"},{"id",id}});
        Json payload;
        try { payload=image_.request(file); }
        catch(const std::exception&) { throw ImageQueueError(400,"Image must exist inside the configured media directory"); }
        return public_job(call({{"op","submit"},{"id",id},{"payload",payload}}));
    }
    Json get(const std::string& id) const override { return public_job(raw(id)); }
    Json cancel(const std::string& id) const override {
        const auto job=call({{"op","cancel"},{"id",id}});
        if(job.is_null()) throw ImageQueueError(404,"Image job not found");
        return public_job(job);
    }
    std::vector<unsigned char> preview(const std::string& id) const override {
        const auto job=raw(id);
        if(job.at("state")!="succeeded") throw ImageQueueError(409,"Completed image job required");
        try { return image_.preview(job.at("payload"),job.at("result")); }
        catch(const std::exception&) { throw ImageQueueError(409,"Image changed or result cannot be verified; preview withheld"); }
    }
};
}
std::shared_ptr<ImageQueueBackend> make_redis_image_queue(const fs::path& model,const fs::path& media,
    int port,const std::string& queue) { return std::make_shared<Backend>(model,media,port,queue); }
}
