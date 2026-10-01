#include "aegisvision/service.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <array>
#include <picosha2.h>

namespace aegisvision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr std::uint64_t mib = 1024 * 1024;
std::string utf8(const fs::path& p) { const auto u=p.generic_u8string(); return {u.begin(),u.end()}; }
bool same_path(const fs::path& a,const fs::path& b) {
#ifdef _WIN32
    auto left=a.generic_u8string(),right=b.generic_u8string();
    const auto lower=[](char8_t c) { return c>=u8'A'&&c<=u8'Z' ? static_cast<char8_t>(c+(u8'a'-u8'A')) : c; };
    std::transform(left.begin(),left.end(),left.begin(),lower);
    std::transform(right.begin(),right.end(),right.begin(),lower); return left==right;
#else
    return a==b;
#endif
}
void owned(const fs::path& root,const fs::path& path) {
    const auto absolute=fs::absolute(path).lexically_normal();
    const auto relative=absolute.lexically_relative(root);
    if (relative.empty() || relative.is_absolute() ||
        std::any_of(relative.begin(),relative.end(),[](const auto& part){ return part==".."; }))
        throw std::invalid_argument("Archive path escape");
    auto ancestor=absolute.root_path();
    for (const auto& part:absolute.relative_path()) {
        ancestor/=part;
        if (fs::is_symlink(fs::symlink_status(ancestor)) || !same_path(fs::weakly_canonical(ancestor),ancestor))
            throw std::invalid_argument("Archive path alias");
    }
}
Json read_json(const fs::path& root,const fs::path& path) {
    owned(root,path);
    if (!fs::is_regular_file(path) || fs::file_size(path)>mib) throw std::invalid_argument("Missing or oversized archive metadata");
    std::ifstream file(path,std::ios::binary); return Json::parse(file);
}
void write_once(const fs::path& root,const fs::path& path,const Json& value) {
    owned(root,path); auto temporary=path; temporary+=".partial"; owned(root,temporary);
    if (fs::exists(path)) return;
    if (fs::exists(temporary)) {
        // Only reclaim this bounded metadata staging file in an owned, sealed slot.
        if (!fs::is_regular_file(temporary) || fs::file_size(temporary)>mib) throw std::runtime_error("Invalid archive metadata staging file");
        fs::remove(temporary);
    }
    const auto text=value.dump(); if (text.size()>mib) throw std::runtime_error("Archive metadata limit");
    std::ofstream output(temporary,std::ios::binary); output.write(text.data(),static_cast<std::streamsize>(text.size())); output.flush();
    if (!output) throw std::runtime_error("Cannot commit archive metadata"); output.close();
    if (fs::exists(path)) throw std::runtime_error("Archive metadata already exists");
    fs::rename(temporary,path);
}
fs::path directory(const fs::path& root,const std::string& session,int index) {
    if (!std::regex_match(session,std::regex("live-[0-9]+-[0-9]+")) || session.size()>96 || index<1 || index>8)
        throw std::invalid_argument("Invalid archive key");
    std::ostringstream name; name<<"segment-"<<std::setw(4)<<std::setfill('0')<<index;
    const auto result=root/session/name.str(); owned(root,result); return result;
}
std::string file_digest(const fs::path& path) {
    if (!fs::is_regular_file(path) || fs::file_size(path)>12*mib) throw std::runtime_error("Archive integrity size limit");
    std::ifstream input(path,std::ios::binary); picosha2::hash256_one_by_one hash;
    std::array<char,65536> buffer{};
    if (!input) throw std::runtime_error("Cannot read archive integrity input");
    while (input) { input.read(buffer.data(),buffer.size()); hash.process(buffer.begin(),buffer.begin()+input.gcount()); }
    if (input.bad()) throw std::runtime_error("Archive integrity read failed"); hash.finish(); return picosha2::get_hash_hex_string(hash);
}
Json fingerprint(const fs::path& path) {
    return {{"bytes",fs::file_size(path)},{"modified",std::to_string(fs::last_write_time(path).time_since_epoch().count())},
        {"sha256",file_digest(path)}};
}
bool matches(const Json& value,const fs::path& path,bool content=false) {
    return fs::is_regular_file(path) && value.at("bytes")==fs::file_size(path) &&
        value.at("modified")==std::to_string(fs::last_write_time(path).time_since_epoch().count()) &&
        (!content || value.at("sha256")==file_digest(path));
}
bool active(const Json& job) { return job.at("state")=="queued" || job.at("state")=="running"; }
}

LocalService::Json LocalService::archive_segment(const std::string& session,int index) const {
    if (!config_.live.archive.enabled) throw std::invalid_argument("Archiving disabled");
    const auto& root=config_.live.archive.root; const auto dir=directory(root,session,index);
    const auto manifest=read_json(root,dir/"manifest.json");
    const auto seal=read_json(root,dir/"sealed.json");
    if (seal.at("manifest_sha256")!=file_digest(dir/"manifest.json")) throw std::invalid_argument("Sealed archive manifest changed");
    const auto frames=manifest.at("frames").get<int>(); const auto fps=manifest.at("source_fps").get<double>();
    const auto width=manifest.at("width").get<int>(),height=manifest.at("height").get<int>();
    const auto source=manifest.at("source_id").get<std::string>();
    if (manifest.at("version")!=1 || manifest.at("raw_sealed")!=true || manifest.at("session_id")!=session ||
        manifest.at("segment_index")!=index || frames<1 || frames>1000 || !std::isfinite(fps) || fps<1 || fps>60 ||
        width<2 || height<2 || width>1920 || height>1080 || width%2 || height%2 ||
        !std::regex_match(source,std::regex("[A-Za-z0-9_-]{1,64}")) ||
        manifest.at("raw_file")!="raw.avi" || manifest.at("media_file")!="clip.mp4")
        throw std::invalid_argument("Invalid sealed archive manifest");
    const auto raw=dir/"raw.avi"; owned(root,raw);
    if (!fs::is_regular_file(raw) || fs::file_size(raw)==0 || fs::file_size(raw)>12*mib || manifest.at("raw_bytes")!=fs::file_size(raw))
        throw std::invalid_argument("Invalid sealed archive video");
    const auto& map=manifest.at("frame_map");
    if (!map.is_array() || map.size()!=static_cast<std::size_t>(frames)) throw std::invalid_argument("Invalid archive frame map");
    std::int64_t previous=-1; std::uint64_t sequence=0;
    for (int i=0;i<frames;++i) {
        const auto& frame=map.at(i); const auto arrival=frame.at("arrival_ms").get<std::int64_t>();
        const auto next=frame.at("source_sequence").get<std::uint64_t>();
        if (frame.at("frame_index")!=i || arrival<previous || next<=sequence ||
            frame.at("source_session")!=manifest.at("source_session") || frame.at("tracking_epoch")!=manifest.at("tracking_epoch"))
            throw std::invalid_argument("Invalid archive frame provenance");
        previous=arrival; sequence=next;
    }
    if (manifest.at("arrival_start_ms")!=map.front().at("arrival_ms") || manifest.at("arrival_end_ms")!=map.back().at("arrival_ms"))
        throw std::invalid_argument("Invalid archive arrival span");
    return manifest;
}
void LocalService::verify_archive_media(const fs::path& path) const {
    const auto& root=config_.live.archive.root;
    const auto name=path.parent_path().filename().string();
    if (!std::regex_match(name,std::regex("segment-000[1-8]")))
        throw std::invalid_argument("Unpublished archive media");
    const auto session=path.parent_path().parent_path().filename().string();
    const auto index=std::stoi(name.substr(8));
    const auto dir=directory(root,session,index);
    if (!same_path(dir/"clip.mp4",path)) throw std::invalid_argument("Unpublished archive media");
    (void)archive_segment(session,index);
    const auto ready=read_json(root,dir/"clip-ready.json");
    owned(root,path);
    // Catalog polling uses cheap file attributes. Serving/preview must check
    // this exact clip's content, even if size and mtime were restored after an
    // edit. Do not rehash every archive slot on each byte-range request.
    if (!matches(ready.at("raw_fingerprint"),dir/"raw.avi") ||
        !matches(ready.at("manifest_fingerprint"),dir/"manifest.json",true) ||
        !matches(ready.at("media_fingerprint"),path,true))
        throw std::invalid_argument("Published archive media changed");
}
std::string LocalService::archive_submit(const std::string& session,int index) {
    std::lock_guard lock(archive_mutex_);
    const auto manifest=archive_segment(session,index);
    for (const auto& job:jobs_.list()) {
        const auto& r=job.at("request");
        if (r.value("type",std::string())=="index_live_archive" && r.value("session_id",std::string())==session &&
            r.value("segment_index",0)==index && active(job)) throw ArchiveJobBusy();
    }
    const auto dir=directory(config_.live.archive.root,session,index);
    const auto raw=fingerprint(dir/"raw.avi"),document=fingerprint(dir/"manifest.json");
    if (read_json(config_.live.archive.root,dir/"sealed.json").at("raw_sha256")!=raw.at("sha256"))
        throw std::invalid_argument("Sealed archive raw video changed");
    const auto ticket=dir/"encoding-ticket.json";
    if (fs::exists(ticket)) {
        const auto original=read_json(config_.live.archive.root,ticket);
        if (original.at("raw_fingerprint")!=raw || original.at("manifest_fingerprint")!=document)
            throw std::invalid_argument("Previously accepted archive input changed");
    }
    return jobs_.submit({{"type","index_live_archive"},{"session_id",session},{"segment_index",index},
        {"path",utf8((dir/"clip.mp4").lexically_relative(config_.media_root))},
        {"raw_fingerprint",raw},{"manifest_fingerprint",document}});
}
LocalService::Json LocalService::archive_list() const {
    const auto& config=config_.live.archive;
    auto segments=Json::array(); Json result{{"enabled",config.enabled},{"segments",segments},
        {"config",{{"segment_seconds",config.segment_seconds},{"max_segments_per_session",config.max_segments_per_session},
            {"max_total_segments",config.max_total_segments},{"max_bytes",config.max_bytes}}}};
    if (!config.enabled || !fs::exists(config.root)) return result;
    // Catalog is the bounded immutable manifest set, not a second unbounded job queue.
    std::vector<std::pair<std::string,int>> keys;
    std::size_t entries=0;
    for (fs::recursive_directory_iterator it(config.root),end;it!=end;++it) {
        if (++entries>4096) throw std::runtime_error("Archive catalog scan limit"); owned(config.root,it->path());
        if (it->path().filename()!="manifest.json" || !it->is_regular_file()) continue;
        const auto relative=it->path().lexically_relative(config.root);
        auto part=relative.begin(); if (part==relative.end()) continue; const auto session=part->string();
        if (++part==relative.end()) continue; const auto segment_name=part->string();
        if (++part==relative.end() || *part!="manifest.json" || ++part!=relative.end() ||
            !std::regex_match(session,std::regex("live-[0-9]+-[0-9]+")) ||
            !std::regex_match(segment_name,std::regex("segment-000[1-8]"))) continue;
        keys.emplace_back(session,std::stoi(segment_name.substr(8)));
        if (keys.size()>8) throw std::runtime_error("Archive catalog slot limit");
    }
    std::sort(keys.rbegin(),keys.rend()); const auto jobs=jobs_.list();
    for (const auto& [session,index]:keys) {
        try {
            const auto m=archive_segment(session,index); const auto dir=directory(config.root,session,index);
            Json value{{"session_id",session},{"source_id",m.at("source_id")},{"source_session",m.at("source_session")},
                {"segment_index",index},{"frames",m.at("frames")},{"source_fps",m.at("source_fps")},
                {"arrival_start_ms",m.at("arrival_start_ms")},{"arrival_end_ms",m.at("arrival_end_ms")},
                {"playback_duration_seconds",m.at("frames").get<double>()/m.at("source_fps").get<double>()},
                {"media_path",nullptr},{"index_state","pending"},{"job_id",nullptr},{"error",nullptr}};
            const auto ready=dir/"clip-ready.json";
            if (fs::exists(ready) && fs::exists(dir/"clip.mp4")) {
                const auto ticket=read_json(config.root,ready);
                owned(config.root,dir/"clip.mp4");
                if (matches(ticket.at("raw_fingerprint"),dir/"raw.avi") &&
                    matches(ticket.at("manifest_fingerprint"),dir/"manifest.json",true) &&
                    matches(ticket.at("media_fingerprint"),dir/"clip.mp4"))
                    value["media_path"]=utf8((dir/"clip.mp4").lexically_relative(config_.media_root));
            }
            const auto complete=dir/"indexed.json";
            if (fs::exists(complete)) {
                const auto receipt=read_json(config.root,complete);
                if (receipt.at("context")==config_.persistence.context && receipt.at("detector_signature")==config_.detector_signature &&
                    matches(receipt.at("raw_fingerprint"),dir/"raw.avi") &&
                    matches(receipt.at("manifest_fingerprint"),dir/"manifest.json",true)) value["index_state"]="succeeded";
            }
            for (const auto& job:jobs) {
                const auto& r=job.at("request");
                if (r.value("type",std::string())=="index_live_archive" && r.value("session_id",std::string())==session && r.value("segment_index",0)==index) {
                    value["job_id"]=job.at("id"); value["index_state"]=job.at("state"); value["error"]=job.at("error"); break;
                }
            }
            segments.push_back(value);
        } catch (const std::exception&) { // Keep damaged slots visible; never silently re-index changed inputs.
            segments.push_back({{"session_id",session},{"segment_index",index},{"frames",0},{"media_path",nullptr},
                {"index_state","failed"},{"job_id",nullptr},{"error","Archive files changed or metadata is invalid"}});
        }
    }
    result["segments"]=segments; return result;
}
LocalService::Json LocalService::index_archive(const Json& request,const JobQueue::Progress& progress,const std::atomic_bool& cancel) {
    const auto session=request.at("session_id").get<std::string>(); const auto index=request.at("segment_index").get<int>();
    const auto manifest=archive_segment(session,index); const auto& root=config_.live.archive.root;
    const auto dir=directory(root,session,index); const auto raw=dir/"raw.avi",clip=dir/"clip.mp4";
    if (request.at("raw_fingerprint")!=fingerprint(raw) || request.at("manifest_fingerprint")!=fingerprint(dir/"manifest.json"))
        throw std::runtime_error("Sealed archive changed after job acceptance");
    auto encoder=config_.archive_encoder; encoder.expected_frames=manifest.at("frames").get<std::uint64_t>(); encoder.fps=manifest.at("source_fps").get<double>();
    const auto ticket=dir/"encoding-ticket.json";
    {
        std::lock_guard lock(archive_mutex_);
        if (fs::exists(ticket) && (read_json(root,ticket).at("raw_fingerprint")!=request.at("raw_fingerprint") ||
            read_json(root,ticket).at("manifest_fingerprint")!=request.at("manifest_fingerprint")))
            throw std::runtime_error("Archive encoder ticket does not match sealed source");
        if (fs::exists(clip) && !fs::exists(ticket)) throw std::runtime_error("Unowned archive output already exists");
        write_once(root,ticket,{{"raw_fingerprint",request.at("raw_fingerprint")},{"manifest_fingerprint",request.at("manifest_fingerprint")}});
        auto partial=dir/"clip.partial.mp4"; owned(root,partial);
        // Crash recovery reclaims only the bounded temporary created by this
        // segment's accepted encoder ticket, never source/final/user media.
        if (fs::exists(partial)) {
            if (!fs::is_regular_file(partial) || fs::file_size(partial)>encoder.max_output_bytes)
                throw std::runtime_error("Invalid archive encoder staging file");
            fs::remove(partial);
        }
    }
    progress({{"phase","encoding"}});
    if (fs::exists(clip)) (void)vision::validate_archive_clip(raw,clip,encoder,cancel);
    else (void)vision::encode_archive_clip(raw,clip,encoder,cancel);
    if (cancel) throw IndexCancelled();
    {
        std::lock_guard lock(archive_mutex_);
        const auto ready=dir/"clip-ready.json";
        if (fs::exists(ready) && (read_json(root,ready).at("media_fingerprint")!=fingerprint(clip) ||
            read_json(root,ready).at("manifest_fingerprint")!=request.at("manifest_fingerprint")))
            throw std::runtime_error("Published archive clip changed");
        write_once(root,ready,{{"raw_fingerprint",request.at("raw_fingerprint")},{"manifest_fingerprint",request.at("manifest_fingerprint")},
            {"media_fingerprint",fingerprint(clip)}});
    }
    VideoIndexConfig options; options.detector_signature=config_.detector_signature; options.frame_stride=10;
    options.source_metadata={{"origin","live_archive"},{"live_session_id",session},{"live_source_id",manifest.at("source_id").get<std::string>()},
        {"segment_index",std::to_string(index)},{"source_session",manifest.at("source_session").dump()},
        {"tracking_epoch",manifest.at("tracking_epoch").dump()},{"live_timestamp_basis","decode arrival; not camera PTS"}};
    for (const auto& frame:manifest.at("frame_map")) options.frame_metadata.push_back({{"live_arrival_ms",frame.at("arrival_ms").dump()},
        {"live_source_sequence",frame.at("source_sequence").dump()}});
    const auto indexed=index_video(clip,detector_,embedder_,store_,options,
        [&](const IndexSummary& value) { auto update=summary(value); update["phase"]="indexing"; progress(update); },[&]{ return cancel.load(); });
    if (indexed.decoded_frames!=encoder.expected_frames) throw std::runtime_error("Archive frame map differs from indexed video");
    {
        std::lock_guard lock(archive_mutex_);
        write_once(root,dir/"indexed.json",{{"context",config_.persistence.context},{"detector_signature",config_.detector_signature},
            {"raw_fingerprint",request.at("raw_fingerprint")},{"manifest_fingerprint",request.at("manifest_fingerprint")},
            {"indexed_items",indexed.indexed_items}});
    }
    auto value=summary(indexed); value["media_path"]=request.at("path"); value["session_id"]=session; value["segment_index"]=index;
    progress(value); return value;
}
}
