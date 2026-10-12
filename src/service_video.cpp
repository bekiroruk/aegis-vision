#include "aegisvision/service.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <regex>
#include <set>

namespace aegisvision {
namespace {
namespace fs=std::filesystem;
constexpr std::uint64_t mib=1024*1024, raw_limit=256*mib, report_limit=16*mib;
std::string u8(const fs::path& path) { const auto s=path.u8string();return {s.begin(),s.end()}; }
void regular(const fs::path& path) {
    if (!fs::is_regular_file(fs::symlink_status(path)) || fs::canonical(path)!=path)
        throw std::runtime_error("Analysis output is not a regular owned file");
}
fs::path output_root(const fs::path& media) {
    const auto root=media/"analysis-output";
    if (!fs::is_directory(fs::symlink_status(root)) || fs::canonical(root)!=root)
        throw std::runtime_error("Analysis output directory is unavailable or aliased");
    return root;
}
fs::path allocate_output(const fs::path& media) {
    const auto root=output_root(media);
    std::uint64_t bytes=0;std::size_t count=0;
    for (const auto& entry:fs::directory_iterator(root)) {
        if (++count>31 || !fs::is_directory(entry.symlink_status()) || fs::canonical(entry.path())!=entry.path())
            throw std::runtime_error("Analysis output quota (32 jobs) reached or directory unsafe; preserve/download old results before manual cleanup");
        for (const auto& file:fs::directory_iterator(entry.path())) {
            regular(file.path());
            const auto size=file.file_size();
            if(size>1024*mib || bytes>1024*mib-size) throw std::runtime_error("Analysis output disk quota exceeded");
            bytes+=size;
        }
    }
    // Reserve space for this worker's raw video, MP4, preview and report.
    if (bytes>640*mib) throw std::runtime_error("Analysis output disk quota (1 GiB) reached; existing results are not deleted automatically");
    static std::atomic_uint64_t sequence{0};
    for (int attempt=0;attempt<10;++attempt) {
        const auto stamp=std::chrono::system_clock::now().time_since_epoch().count();
        const auto directory=root/("v-"+std::to_string(stamp)+"-"+std::to_string(++sequence));
        if (fs::create_directory(directory)) return directory;
    }
    throw std::runtime_error("Cannot allocate exclusive analysis output");
}
// Only this invocation's known files in its newly created directory are removed.
// A crash leaves bounded orphan files for manual inspection, never reuses partial output.
struct OutputGuard {
    fs::path directory;bool keep=false;
    ~OutputGuard() {
        if(keep) return;
        std::error_code ec;
        if(fs::weakly_canonical(directory,ec)!=directory || ec) return;
        for(const auto* name:{"raw.avi","video.mp4","video.partial.mp4","tracks.csv","preview.jpg"}) {
            fs::remove(directory/name,ec);ec.clear();
        }
        fs::remove(directory,ec);
    }
};
}

LocalService::Json LocalService::segment_video(const Json& request,const JobQueue::Progress& progress,
    const std::atomic_bool& cancel) {
    if(!config_.segmenter) throw std::runtime_error("Segmentation is not configured");
    const auto path=media_path(request.at("path").get<std::string>());
    const auto unchanged=[&] {
        if(request.at("source_bytes")!=fs::file_size(path) || request.at("source_modified")!=
            std::to_string(fs::last_write_time(path).time_since_epoch().count()))
            throw std::runtime_error("Video changed after job acceptance; submit a new job");
    };
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(30);
    const auto checkpoint=[&] {
        if(cancel) throw IndexCancelled();
        if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("Video analysis exceeded 30 minute budget");
    };
    unchanged();checkpoint();
    cv::VideoCapture capture(u8(path));
    if(!capture.isOpened()) throw std::runtime_error("Cannot open analysis video");
    const double fps=capture.get(cv::CAP_PROP_FPS), total=capture.get(cv::CAP_PROP_FRAME_COUNT);
    if(!std::isfinite(fps) || fps<1 || fps>60) throw std::runtime_error("Video export requires a known frame rate in 1..60 FPS");
    const int limit=request.at("max_frames").get<int>();
    if(limit<0 || limit>1000) throw std::runtime_error("Invalid video frame limit");
    if(!limit && std::isfinite(total) && total>1000) throw std::runtime_error("Full video exceeds 1000 frames; select an explicit shorter frame limit");
    const double header_width=capture.get(cv::CAP_PROP_FRAME_WIDTH), header_height=capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    if(header_width>1920 || header_height>1080) throw std::runtime_error("Video export supports at most 1920x1080");
    const auto directory=allocate_output(config_.media_root);OutputGuard owned{directory};
    cv::VideoWriter writer;
    std::ofstream rows(directory/"tracks.csv");
    rows.exceptions(std::ios::badbit|std::ios::failbit);
    rows << "frame,track_id,label,score,x,y,width,height,mask_pixels\n" << std::setprecision(9);
    vision::SegmentationPipeline pipeline(*config_.segmenter,true);
    cv::Mat frame;cv::Size size;
    int frames=0;std::uint64_t instances=0;std::set<std::uint64_t> track_ids;
    bool reached_end=false;
    while(!limit || frames<limit) {
        checkpoint();
        if(!capture.read(frame) || frame.empty()) {reached_end=true;break;}
        if(frames>=1000) throw std::runtime_error("Video exceeds 1000 frames; no partial success published");
        if(frame.type()!=CV_8UC3 || frame.cols<2 || frame.rows<2 || frame.cols>1920 || frame.rows>1080 || frame.cols%2 || frame.rows%2)
            throw std::runtime_error("Video export requires even dimensions up to 1920x1080");
        if(!frames) {
            size=frame.size();
            writer.open(u8(directory/"raw.avi"),cv::CAP_OPENCV_MJPEG,cv::VideoWriter::fourcc('M','J','P','G'),fps,size);
            if(!writer.isOpened()) throw std::runtime_error("Cannot create analysis video");
        }
        if(frame.size()!=size) throw std::runtime_error("Video dimensions changed");
        const auto masks=pipeline.analyze(frame);checkpoint();
        auto painted=vision::paint_masks(frame,masks);
        writer.write(painted);
        if(!frames && !cv::imwrite(u8(directory/"preview.jpg"),painted,{cv::IMWRITE_JPEG_QUALITY,80}))
            throw std::runtime_error("Cannot write video preview");
        for(const auto& item:masks) {
            // Numeric/ASCII labels only in CSV; prevent spreadsheet formula injection.
            auto label=item.detection.label;
            if(label.empty() || label.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-")!=std::string::npos ||
                label.front()=='-' || label.size()>128) label="object";
            const auto& r=item.region;
            rows << frames << ',' << item.track_id << ',' << label << ',' << item.detection.score << ','
                << r.x << ',' << r.y << ',' << r.width << ',' << r.height << ',' << cv::countNonZero(item.mask) << '\n';
            track_ids.insert(item.track_id);
        }
        rows.flush();
        if(static_cast<std::uint64_t>(rows.tellp())>report_limit || fs::file_size(directory/"raw.avi")>raw_limit)
            throw std::runtime_error("Video analysis output size limit exceeded");
        instances+=masks.size();++frames;
        progress({{"stage","segmenting_video"},{"processed_frames",frames},{"total_frames_hint",std::isfinite(total)&&total>0 ? (limit?std::min(total,double(limit)):total):0},
            {"instances",instances},{"track_ids",track_ids.size()}});
    }
    writer.release();rows.close();capture.release();checkpoint();unchanged();
    if(!frames) throw std::runtime_error("Video has no decodable frames");
    if(fs::file_size(directory/"raw.avi")>raw_limit) throw std::runtime_error("Raw video exceeds 256 MiB limit");
    progress({{"stage","encoding_video"},{"processed_frames",frames},{"instances",instances},{"track_ids",track_ids.size()}});
    auto encoding=config_.archive_encoder;
    encoding.expected_frames=frames;encoding.fps=fps;encoding.max_input_bytes=raw_limit;encoding.max_output_bytes=64*mib;
    vision::EncodedArchiveClip clip;
    try {clip=vision::encode_archive_clip(directory/"raw.avi",directory/"video.mp4",encoding,cancel);}
    catch(...) {if(cancel) throw IndexCancelled();throw;}
    checkpoint();unchanged();
    fs::remove(directory/"raw.avi");
    Json result{{"path",request.at("path")},{"export_id",directory.filename().string()},
        {"frames",frames},{"fps",clip.fps},{"duration_seconds",frames/clip.fps},{"width",clip.width},{"height",clip.height},
        {"bytes",clip.bytes},{"instances",instances},{"track_ids",track_ids.size()},{"tracking",true},
        {"tracker","IoU .30, max_missed 20; video-local IDs, not unique people"},{"audio",false},
        {"stop_reason",reached_end?"end_of_stream_or_decode_stop":"requested_frame_limit"},
        {"model_signature",config_.segmentation_signature}};
    owned.keep=true;return result;
}

void LocalService::video_result_routes() {
    http_.Get(R"(/api/jobs/([0-9]+-[0-9]+)/(video\.mp4|tracks\.csv|preview\.jpg))",[this](const httplib::Request& request,httplib::Response& response) {
        try {
            const auto job=jobs_.get(request.matches[1]);
            if(job.is_null() || job.at("state")!="succeeded" || job.at("request").at("type")!="segment_video")
                throw std::runtime_error("Completed video analysis required");
            const auto token=job.at("result").at("export_id").get<std::string>();
            if(!std::regex_match(token,std::regex("v-[0-9]+-[0-9]+"))) throw std::runtime_error("Invalid export ID");
            const auto directory=output_root(config_.media_root)/token;
            if(!fs::is_directory(fs::symlink_status(directory)) || fs::canonical(directory)!=directory)
                throw std::runtime_error("Analysis output directory unavailable");
            const std::string name=request.matches[2];const auto path=directory/name;regular(path);
            const auto bytes=fs::file_size(path);
            if(!bytes || bytes>64*mib) throw std::runtime_error("Analysis result size is invalid");
            if(name=="video.mp4" && job.at("result").at("bytes")!=bytes) throw std::runtime_error("Analysis video changed");
            const auto file=std::make_shared<std::ifstream>(path,std::ios::binary);
            if(!*file) throw std::runtime_error("Cannot open analysis output");
            const char* mime=name=="video.mp4"?"video/mp4":name=="preview.jpg"?"image/jpeg":"text/csv; charset=utf-8";
            response.set_header("Accept-Ranges","bytes");
            if(request.has_param("download") || name=="tracks.csv")
                response.set_header("Content-Disposition","attachment; filename=\"aegis-"+std::string(request.matches[1])+"-"+name+"\"");
            response.set_content_provider(static_cast<std::size_t>(bytes),mime,[file](std::size_t offset,std::size_t length,httplib::DataSink& sink) {
                file->clear();file->seekg(static_cast<std::streamoff>(offset));std::array<char,65536> buffer{};
                while(length) {
                    const auto count=std::min(length,buffer.size());
                    if(!file->read(buffer.data(),static_cast<std::streamsize>(count)) || !sink.write(buffer.data(),count)) return false;
                    length-=count;
                }
                return true;
            });
        } catch(const std::exception& e) {response.status=404;response.set_content(Json{{"error",e.what()}}.dump(),"application/json");}
    });
}
}
