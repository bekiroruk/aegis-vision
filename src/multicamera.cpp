#include "aegisvision/multicamera.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace aegisvision {
namespace {
bool camera_name(const std::string& value) {
    return !value.empty() && value.size()<=64 && value.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos;
}
struct Best {
    std::optional<std::size_t> index;
    double score{-2}, second{-2};
    void consider(std::size_t candidate,double similarity) {
        if(similarity>score) {second=score;score=similarity;index=candidate;}
        else second=std::max(second,similarity);
    }
    bool clear(double margin) const {return score-second>=margin;}
};
}
std::vector<CameraIdentity> associate_cameras(const std::vector<CameraTracklet>& tracks,const MultiCameraConfig& c) {
    if(tracks.size()>512 || c.cameras.size()<2 || c.cameras.size()>16 || c.transitions.size()>240 ||
        c.embedding_space.empty() || c.embedding_space.size()>512 ||
        !std::isfinite(c.min_similarity) || c.min_similarity<=0 || c.min_similarity>1 ||
        !std::isfinite(c.ambiguity_margin) || c.ambiguity_margin<=0 || c.ambiguity_margin>1)
        throw std::invalid_argument("Invalid multicamera bounds, embedding space or thresholds");
    std::set<std::string> cameras;
    for(const auto& camera:c.cameras)
        if(!camera_name(camera) || !cameras.insert(camera).second) throw std::invalid_argument("Invalid or duplicate camera");
    std::map<std::pair<std::string,std::string>,CameraTransition> topology;
    for(const auto& edge:c.transitions) {
        if(!cameras.contains(edge.from) || !cameras.contains(edge.to) || edge.from==edge.to ||
            edge.min_gap_ms<1 || edge.max_gap_ms<edge.min_gap_ms || edge.max_gap_ms>86400000 ||
            !topology.emplace(std::make_pair(edge.from,edge.to),edge).second)
            throw std::invalid_argument("Invalid or duplicate directed camera transition");
    }
    std::set<std::tuple<std::string,std::uint64_t,std::uint64_t>> keys;
    std::vector<std::vector<double>> features;
    features.reserve(tracks.size());
    for(const auto& track:tracks) {
        if(!cameras.contains(track.camera) || !track.epoch || !track.local_id || track.start_ms<0 ||
            track.end_ms<track.start_ms || track.embedding_space!=c.embedding_space || track.embedding.size()!=512 ||
            !keys.emplace(track.camera,track.epoch,track.local_id).second)
            throw std::invalid_argument("Invalid tracklet key, time, dimension or embedding space");
        double norm=0;
        for(float value:track.embedding) {
            if(!std::isfinite(value)) throw std::invalid_argument("Nonfinite person embedding");
            norm+=static_cast<double>(value)*value;
        }
        if(norm<1e-24) throw std::invalid_argument("Zero person embedding");
        norm=std::sqrt(norm);
        std::vector<double> feature;feature.reserve(512);
        for(float value:track.embedding) feature.push_back(value/norm);
        features.push_back(std::move(feature));
    }
    std::vector<Best> outgoing(tracks.size()),incoming(tracks.size());
    for(std::size_t i=0;i<tracks.size();++i) for(std::size_t j=0;j<tracks.size();++j) {
        const auto edge=topology.find({tracks[i].camera,tracks[j].camera});
        // Strict temporal separation prevents cycles and simultaneous identity reuse.
        if(edge==topology.end() || tracks[j].start_ms<=tracks[i].end_ms) continue;
        const auto gap=tracks[j].start_ms-tracks[i].end_ms;
        if(gap<edge->second.min_gap_ms || gap>edge->second.max_gap_ms) continue;
        const double similarity=std::clamp(std::inner_product(features[i].begin(),features[i].end(),features[j].begin(),0.),-1.,1.);
        outgoing[i].consider(j,similarity);incoming[j].consider(i,similarity);
    }
    std::vector<CameraIdentity> result(tracks.size());
    std::vector<std::size_t> order(tracks.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b) {
        return std::tie(tracks[a].start_ms,tracks[a].end_ms,tracks[a].camera,tracks[a].epoch,tracks[a].local_id)<
               std::tie(tracks[b].start_ms,tracks[b].end_ms,tracks[b].camera,tracks[b].epoch,tracks[b].local_id);
    });
    std::uint64_t next=0;
    for(auto j:order) {
        auto& decision=result[j];const auto& in=incoming[j];
        decision.reason="no_candidate";
        if(in.index) {
            decision.best_similarity=in.score;
            const auto i=*in.index;const auto& out=outgoing[i];
            if(in.score<c.min_similarity) decision.reason="below_threshold";
            else if(!in.clear(c.ambiguity_margin)) decision.reason="ambiguous_incoming";
            else if(out.index!=j) decision.reason="not_mutual_best";
            else if(!out.clear(c.ambiguity_margin)) decision.reason="ambiguous_outgoing";
            else {decision.predecessor=i;decision.global_id=result[i].global_id;decision.reason="matched";}
        }
        if(!decision.predecessor) decision.global_id=++next;
    }
    return result;
}
}
