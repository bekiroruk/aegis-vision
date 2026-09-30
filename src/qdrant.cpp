#include "aegisvision/qdrant.hpp"
#include "aegisvision/clip.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <cmath>
#include <regex>
#include <stdexcept>

namespace aegisvision {
using Json=nlohmann::json;
std::string qdrant_point_id(std::string_view space,std::string_view item) {
    auto hex=picosha2::hash256_hex_string(Json::array({"aegisvision-v1",std::string(space),std::string(item)}).dump()).substr(0,32);
    hex[12]='8';
    const auto nibble=std::stoi(hex.substr(16,1),nullptr,16);
    hex[16]="89ab"[nibble&3];
    return hex.substr(0,8)+"-"+hex.substr(8,4)+"-"+hex.substr(12,4)+"-"+hex.substr(16,4)+"-"+hex.substr(20);
}
namespace {
Json response(const httplib::Result& result) {
    if(!result) throw std::runtime_error("Qdrant transport failed: " + httplib::to_string(result.error()));
    if(result->status<200 || result->status>=300) throw std::runtime_error("Qdrant HTTP " + std::to_string(result->status)+": "+result->body.substr(0,512));
    auto json=Json::parse(result->body);
    if(json.at("status")!="ok") throw std::runtime_error("Qdrant did not acknowledge request");
    return json;
}
}
struct QdrantVectorStore::Impl {
    QdrantConfig config;
    mutable httplib::Client client;
    std::string path;
    explicit Impl(QdrantConfig c) : config(std::move(c)),client(config.host,config.port),path("/collections/"+config.collection) {
        if((config.host!="127.0.0.1" && config.host!="localhost") || config.port<1 || config.port>65535 ||
           !std::regex_match(config.collection,std::regex("[A-Za-z0-9_-]{1,64}")) ||
           config.dimension==0 || config.dimension>65536 || config.embedding_space.empty() || config.embedding_space.size()>256 ||
           config.timeout_seconds<1 || config.timeout_seconds>120)
            throw std::invalid_argument("Invalid local Qdrant configuration");
        client.set_connection_timeout(config.timeout_seconds);
        client.set_read_timeout(config.timeout_seconds);
        client.set_write_timeout(config.timeout_seconds);
        client.set_follow_location(false);
    }
    std::vector<float> vector(const std::vector<float>& values) const {
        if(values.size()!=config.dimension) throw std::invalid_argument("Qdrant vector dimension mismatch");
        return normalize_embedding(values);
    }
};
QdrantVectorStore::QdrantVectorStore(QdrantConfig config,bool create) : impl_(std::make_unique<Impl>(std::move(config))) {
    auto info=impl_->client.Get(impl_->path);
    if(info && info->status==404 && create) {
        const auto body=Json{{"vectors",{{"size",impl_->config.dimension},{"distance","Cosine"}}}}.dump();
        const auto created=impl_->client.Put(impl_->path,body,"application/json");
        // A concurrent creator is safe only if the resulting collection matches below.
        if(!created || created->status!=409) (void)response(created);
        info=impl_->client.Get(impl_->path);
    }
    const auto vectors=response(info).at("result").at("config").at("params").at("vectors");
    if(!vectors.contains("size") || vectors.at("size")!=impl_->config.dimension || vectors.at("distance")!="Cosine")
        throw std::invalid_argument("Existing Qdrant collection must use matching unnamed Cosine vectors; left unchanged");
}
QdrantVectorStore::~QdrantVectorStore()=default;
std::size_t QdrantVectorStore::point_count() const {
    return response(impl_->client.Get(impl_->path)).at("result").at("points_count").get<std::size_t>();
}
void QdrantVectorStore::upsert(std::string id,std::vector<float> vector,std::map<std::string,std::string> metadata) {
    if(id.empty() || id.size()>4096) throw std::invalid_argument("Item ID must contain 1..4096 bytes");
    const Json payload{{"item_id",id},{"space_id",impl_->config.embedding_space},{"metadata",metadata}};
    const Json point{{"id",qdrant_point_id(impl_->config.embedding_space,id)},{"vector",impl_->vector(vector)},{"payload",payload}};
    const auto result=response(impl_->client.Put(impl_->path+"/points?wait=true",Json{{"points",Json::array({point})}}.dump(),"application/json"));
    if(result.at("result").at("status")!="completed") throw std::runtime_error("Qdrant write is not completed");
}
std::vector<SearchResult> QdrantVectorStore::search(const std::vector<float>& vector,std::size_t limit) const {
    if(limit<1 || limit>100) throw std::invalid_argument("Search limit must be 1..100");
    const Json filter{{"must",Json::array({Json{{"key","space_id"},{"match",{{"value",impl_->config.embedding_space}}}}})}};
    const Json body{{"vector",impl_->vector(vector)},{"limit",limit},{"with_payload",true},{"filter",filter}};
    const auto result=response(impl_->client.Post(impl_->path+"/points/search",body.dump(),"application/json"));
    std::vector<SearchResult> matches;
    for(const auto& point:result.at("result")) {
        const auto& payload=point.at("payload");
        const float score=point.at("score").get<float>();
        if(!std::isfinite(score) || payload.at("space_id")!=impl_->config.embedding_space) throw std::runtime_error("Invalid Qdrant search response");
        matches.push_back({payload.at("item_id").get<std::string>(),score,payload.at("metadata").get<std::map<std::string,std::string>>()});
    }
    return matches;
}
}
