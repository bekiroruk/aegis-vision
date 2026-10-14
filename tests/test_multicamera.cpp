#include "aegisvision/multicamera.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <map>
using namespace aegisvision;
namespace {
void require(bool value,const char* why) {if(!value) throw std::runtime_error(why);}
template<class F> void rejects(F action) {
    try {action();} catch(const std::invalid_argument&) {return;}
    throw std::runtime_error("Invalid input accepted");
}
CameraTracklet track(std::string camera,std::uint64_t id,std::int64_t begin,std::int64_t end,int person=0) {
    CameraTracklet t{std::move(camera),1,id,begin,end,"fixture-osnet-space",std::vector<float>(512,0)};
    t.embedding.at(person)=1;return t;
}
MultiCameraConfig config() {return {{"A","B","C"},{{"A","B",100,2000},{"B","C",100,2000}},"fixture-osnet-space",.85,.05};}
void checks() {
    auto c=config();
    std::vector<CameraTracklet> tracks{track("A",1,0,500),track("B",1,800,1200),track("C",1,1500,2000),track("B",2,800,1200,1)};
    const auto original=tracks;
    auto r=associate_cameras(tracks,c);
    require(r[0].global_id==r[1].global_id && r[1].global_id==r[2].global_id && r[3].global_id!=r[0].global_id,"Cross-camera chain missing");
    require(r[1].predecessor==0 && r[2].predecessor==1 && r[3].reason=="below_threshold","Wrong explanation or predecessor");
    require(tracks[0].embedding==original[0].embedding,"Input mutated");
    std::map<std::pair<std::string,std::uint64_t>,std::uint64_t> stable;
    for(std::size_t i=0;i<tracks.size();++i) stable[{tracks[i].camera,tracks[i].local_id}]=r[i].global_id;
    std::mt19937 random(71);
    for(int repeat=0;repeat<30;++repeat) {
        std::shuffle(tracks.begin(),tracks.end(),random);r=associate_cameras(tracks,c);
        for(std::size_t i=0;i<tracks.size();++i) require(r[i].global_id==stable.at({tracks[i].camera,tracks[i].local_id}),"Permutation changes identity");
    }
    tracks={track("A",1,0,500),track("A",2,0,500),track("B",1,800,1200)};
    r=associate_cameras(tracks,c);
    require(r[2].reason=="ambiguous_incoming" && !r[2].predecessor,"Look-alike sources were merged");
    tracks={track("A",1,0,500),track("B",1,800,1200),track("B",2,800,1200)};
    r=associate_cameras(tracks,c);
    require(!r[1].predecessor && !r[2].predecessor && r[1].global_id!=r[2].global_id,"One person forked into simultaneous camera tracks");
    for(auto begin:{499,500,599,2501}) {
        r=associate_cameras({track("A",1,0,500),track("B",1,begin,3000)},c);
        require(!r[1].predecessor,"Overlap or invalid travel time accepted");
    }
    for(auto begin:{600,2500}) {
        r=associate_cameras({track("A",1,0,500),track("B",1,begin,3000)},c);
        require(r[1].predecessor==0,"Inclusive travel bound rejected");
    }
    r=associate_cameras({track("B",1,0,500),track("A",1,800,1200)},c);
    require(!r[1].predecessor,"Unconfigured reverse direction matched");
    r=associate_cameras({track("A",1,0,500),track("A",2,800,1200)},c);
    require(!r[1].predecessor,"Same-camera fragments incorrectly merged");
    tracks={track("A",1,0,500),track("B",1,800,1200)};tracks[0].embedding[0]=100;
    require(associate_cameras(tracks,c)[1].predecessor==0,"Embedding normalization missing");
    tracks[1].embedding[0]=-1;
    require(associate_cameras(tracks,c)[1].reason=="below_threshold","Opposite features matched");
    tracks={track("A",1,0,500),track("A",1,800,1200)};tracks[1].epoch=2;
    require(associate_cameras(tracks,c)[0].global_id!=associate_cameras(tracks,c)[1].global_id,"Restarted local ID collided");
    tracks[1].epoch=1;rejects([&]{(void)associate_cameras(tracks,c);});
    tracks={track("A",1,0,500),track("B",1,800,1200),track("B",2,800,1200)};
    tracks[2].embedding[0]=.98F;tracks[2].embedding[1]=.199F;
    auto narrow=c;narrow.ambiguity_margin=.01;
    r=associate_cameras(tracks,narrow);
    require(r[1].reason=="matched" && r[2].reason=="not_mutual_best" && !r[2].predecessor,"Non-mutual candidate forced into same identity");
    for(int mode=0;mode<7;++mode) {
        auto t=track("A",1,0,500);
        if(mode==0)t.embedding[0]=std::numeric_limits<float>::quiet_NaN();
        if(mode==1)t.embedding.assign(512,0);
        if(mode==2)t.embedding.resize(511);
        if(mode==3)t.embedding_space="clip-not-reid";
        if(mode==4)t.start_ms=-1;
        if(mode==5)t.end_ms=-1;
        if(mode==6)t.camera="unknown";
        rejects([&]{(void)associate_cameras({t},c);});
    }
    for(int mode=0;mode<5;++mode) {
        auto bad=c;
        if(mode==0)bad.ambiguity_margin=0;
        if(mode==1)bad.min_similarity=std::numeric_limits<double>::infinity();
        if(mode==2)bad.transitions[0].min_gap_ms=0;
        if(mode==3)bad.transitions.push_back(bad.transitions[0]);
        if(mode==4)bad.cameras.push_back("A");
        rejects([&]{(void)associate_cameras({},bad);});
    }
    tracks.clear();for(int i=0;i<512;++i)tracks.push_back(track("A",i+1,0,500));
    require(associate_cameras(tracks,c).size()==512,"Valid bound rejected");
    tracks.push_back(track("A",513,0,500));rejects([&]{(void)associate_cameras(tracks,c);});
    require(associate_cameras({},c).empty(),"Empty batch not empty");
}
}
int main() {
    try {checks();std::cout<<"PASS: multicamera handoff, ambiguity, clock/topology gates, namespaces, bounds and deterministic IDs (synthetic features).\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
