#include "aegisvision/live_archive.hpp"
#include "archive_scan_retry.hpp"
#include <picosha2.h>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace aegisvision::vision {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr std::uintmax_t mib = 1024 * 1024;
constexpr std::uintmax_t reservation = 32 * mib;
constexpr std::size_t max_entries = 4096;
bool safe_id(const std::string& id) {
    return !id.empty() && id.size() <= 96 &&
        std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_';
        });
}
bool same_path(const fs::path& left, const fs::path& right) {
#ifdef _WIN32
    auto a = left.generic_u8string(), b = right.generic_u8string();
    const auto fold = [](char8_t c) {
        return c >= u8'A' && c <= u8'Z' ? static_cast<char8_t>(c + (u8'a' - u8'A')) : c;
    };
    std::transform(a.begin(), a.end(), a.begin(), fold);
    std::transform(b.begin(), b.end(), b.begin(), fold);
    return a == b;
#else
    return left == right;
#endif
}
// Check every existing ancestor, not merely the leaf. Canonical comparison also
// catches directory aliases/junctions that a platform may report as directories.
void no_aliases(const fs::path& path) {
    const auto absolute = fs::absolute(path).lexically_normal();
    auto current = absolute.root_path();
    for (const auto& component : absolute.relative_path()) {
        current /= component;
        const auto status = fs::symlink_status(current);
        if (status.type() == fs::file_type::not_found) continue;
        if (fs::is_symlink(status) || !same_path(fs::weakly_canonical(current), current))
            throw std::runtime_error("archive_path_alias");
    }
}
void inside(const fs::path& root, const fs::path& path) {
    const auto relative = fs::absolute(path).lexically_normal().lexically_relative(root);
    if (relative.empty() || relative.is_absolute() ||
        std::any_of(relative.begin(), relative.end(), [](const fs::path& part) { return part == ".."; }))
        throw std::runtime_error("archive_path_escape");
    no_aliases(path);
}
struct DiskUsage { std::uintmax_t bytes{}, slot_bytes{}, reserved_bytes{}; int slots{}; };
DiskUsage scan_once(const fs::path& root) {
    no_aliases(root);
    DiskUsage usage;
    if (!fs::exists(root)) return usage;
    if (!fs::is_directory(root)) throw std::runtime_error("archive_root_not_directory");
    std::size_t entries = 0;
    std::map<fs::path, std::uintmax_t> slot_sizes;
    for (fs::recursive_directory_iterator it(root), end; it != end; ++it) {
        if (++entries > max_entries) throw std::runtime_error("archive_scan_limit");
        const auto path = it->path();
        inside(root, path);
        const auto status = it->symlink_status();
        if (status.type() == fs::file_type::not_found)
            throw fs::filesystem_error("Archive entry disappeared during scan", path,
                std::make_error_code(std::errc::no_such_file_or_directory));
        if (fs::is_directory(status)) {
            if (it.depth() == 1) {
                ++usage.slots; // Includes incomplete/unrecognized slots.
                if (usage.slots > 8) throw std::runtime_error("archive_disk_quota");
                slot_sizes.try_emplace(path, 0);
            }
        } else if (fs::is_regular_file(status)) {
            const auto size = it->file_size();
            if (size > std::numeric_limits<std::uintmax_t>::max() - usage.bytes)
                throw std::runtime_error("archive_size_overflow");
            usage.bytes += size;
            if (usage.bytes > 256 * mib) throw std::runtime_error("archive_disk_quota");
            if (it.depth() >= 2) {
                usage.slot_bytes += size;
                // Keep the relative path alive while traversing its components.
                const auto relative = path.lexically_relative(root);
                auto parts = relative.begin();
                auto slot = root / *parts++; slot /= *parts;
                slot_sizes[slot] += size;
            }
        } else throw std::runtime_error("archive_unsupported_entry");
    }
    for (const auto& [path, bytes] : slot_sizes) {
        (void)path;
        usage.reserved_bytes += std::max(bytes, reservation);
    }
    return usage;
}
DiskUsage scan(const fs::path& root) {
    return detail::stable_archive_scan([&] { return scan_once(root); });
}
// Reserved slot bytes plus everything else. Over-sized/unmanaged slot contents
// still consume real bytes rather than being hidden behind a reservation.
std::uintmax_t accounted_bytes(const DiskUsage& usage) {
    return usage.reserved_bytes +
        (usage.bytes - usage.slot_bytes);
}
void new_json(const fs::path& root, const fs::path& path, const Json& value) {
    inside(root, path);
    auto temporary = path; temporary += ".partial";
    inside(root, temporary);
    if (fs::exists(path) || fs::exists(temporary)) throw std::runtime_error("archive_output_exists");
    const auto text = value.dump(2);
    if (text.size() > mib) throw std::runtime_error("archive_manifest_limit");
    std::ofstream output(temporary, std::ios::binary | std::ios::out);
    if (!output) throw std::runtime_error("archive_manifest_open_failed");
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.flush();
    if (!output) throw std::runtime_error("archive_manifest_write_failed");
    output.close();
    if (fs::exists(path)) throw std::runtime_error("archive_output_exists");
    fs::rename(temporary, path);
}
std::string file_sha256(const fs::path& root, const fs::path& path, std::uintmax_t limit) {
    inside(root, path);
    if (!fs::is_regular_file(path) || fs::file_size(path) > limit)
        throw std::runtime_error("archive_integrity_size_limit");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("archive_integrity_read_failed");
    std::array<char, 64 * 1024> buffer{};
    picosha2::hash256_one_by_one hasher;
    std::uintmax_t bytes = 0;
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        bytes += static_cast<std::uintmax_t>(count);
        if (bytes > limit) throw std::runtime_error("archive_integrity_size_limit");
        hasher.process(buffer.begin(), buffer.begin() + count);
    }
    if (!input.eof() || bytes != fs::file_size(path)) throw std::runtime_error("archive_integrity_read_failed");
    inside(root, path);
    hasher.finish();
    return picosha2::get_hash_hex_string(hasher);
}
// Finite classic AVI/MJPEG: RIFF/hdrl/strl/movi/idx1, explicit little-endian
// serialization (no compiler-packed structs). Format reference:
// https://learn.microsoft.com/en-us/windows/win32/directshow/avi-riff-file-reference
// Pre-encoded JPEG chunks make the hard byte limit exact, including final index.
class MjpegAvi {
public:
    bool is_open() const { return output_.is_open(); }
    bool fits(std::size_t jpeg_bytes, std::uintmax_t limit) const {
        const auto chunk_bytes = 8 + jpeg_bytes + (jpeg_bytes % 2);
        return bytes_ + chunk_bytes + 8 + (index_.size() + 1) * 16 <= limit;
    }
    void open(const fs::path& path, cv::Size size, double fps) {
        output_.open(path, std::ios::binary | std::ios::out);
        if (!output_) throw std::runtime_error("archive_encoder_open_failed");
        bytes_ = 224; index_.clear(); largest_ = 0; size_ = size; fps_ = fps;
        header(0, 0, 4, 0);
        check();
    }
    void write(const std::vector<unsigned char>& jpeg) {
        if (jpeg.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("archive_frame_too_large");
        index_.push_back({static_cast<std::uint32_t>(bytes_ - 220), static_cast<std::uint32_t>(jpeg.size())});
        fourcc("00dc"); u32(static_cast<std::uint32_t>(jpeg.size()));
        output_.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size()));
        if (jpeg.size() % 2) output_.put('\0');
        bytes_ += 8 + jpeg.size() + (jpeg.size() % 2);
        largest_ = std::max(largest_, static_cast<std::uint32_t>(jpeg.size()));
        output_.flush(); check();
    }
    void seal() {
        if (!is_open()) return;
        const auto movi_size = static_cast<std::uint32_t>(bytes_ - 220);
        fourcc("idx1"); u32(static_cast<std::uint32_t>(index_.size() * 16));
        for (const auto& entry : index_) { fourcc("00dc"); u32(0x10); u32(entry.offset); u32(entry.size); }
        bytes_ += 8 + index_.size() * 16;
        output_.seekp(0);
        header(static_cast<std::uint32_t>(bytes_ - 8), static_cast<std::uint32_t>(index_.size()), movi_size, largest_);
        output_.flush(); check(); output_.close();
        if (output_.fail()) throw std::runtime_error("archive_raw_write_failed");
    }
    void abandon() noexcept { try { output_.close(); } catch (...) {} }
private:
    struct Entry { std::uint32_t offset, size; };
    std::ofstream output_;
    std::vector<Entry> index_;
    std::uintmax_t bytes_{224};
    std::uint32_t largest_{};
    cv::Size size_;
    double fps_{};
    void fourcc(const char* value) { output_.write(value, 4); }
    void u32(std::uint32_t value) {
        const char bytes[]{static_cast<char>(value), static_cast<char>(value >> 8),
            static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
        output_.write(bytes, 4);
    }
    void u16(std::uint16_t value) {
        const char bytes[]{static_cast<char>(value), static_cast<char>(value >> 8)};
        output_.write(bytes, 2);
    }
    void check() const { if (!output_) throw std::runtime_error("archive_raw_write_failed"); }
    void header(std::uint32_t riff_size, std::uint32_t frames, std::uint32_t movi_size, std::uint32_t largest) {
        fourcc("RIFF"); u32(riff_size); fourcc("AVI ");
        fourcc("LIST"); u32(192); fourcc("hdrl"); fourcc("avih"); u32(56);
        u32(static_cast<std::uint32_t>(std::llround(1000000 / fps_)));
        u32(static_cast<std::uint32_t>(std::ceil(largest * fps_))); u32(0); u32(0x10);
        u32(frames); u32(0); u32(1); u32(largest); u32(static_cast<std::uint32_t>(size_.width));
        u32(static_cast<std::uint32_t>(size_.height)); for (int i = 0; i < 4; ++i) u32(0);
        fourcc("LIST"); u32(116); fourcc("strl"); fourcc("strh"); u32(56);
        fourcc("vids"); fourcc("MJPG"); u32(0); u16(0); u16(0); u32(0);
        u32(1000); u32(static_cast<std::uint32_t>(std::llround(fps_ * 1000))); u32(0); u32(frames);
        u32(largest); u32(0xffffffff); u32(0); u16(0); u16(0);
        u16(static_cast<std::uint16_t>(size_.width)); u16(static_cast<std::uint16_t>(size_.height));
        fourcc("strf"); u32(40); u32(40); u32(static_cast<std::uint32_t>(size_.width));
        u32(static_cast<std::uint32_t>(size_.height)); u16(1); u16(24); fourcc("MJPG");
        u32(static_cast<std::uint32_t>(size_.width * size_.height * 3)); for (int i = 0; i < 4; ++i) u32(0);
        fourcc("LIST"); u32(movi_size); fourcc("movi");
    }
};
}

struct LiveArchive::Impl {
    LiveArchiveConfig config;
    std::string session_id, source_id, state{"disabled"}, error;
    Callback finalized;
    MjpegAvi writer;
    ArchivedSegment segment;
    Json frame_map = Json::array();
    std::vector<Json> closed;
    cv::Size size;
    std::uint64_t source_session{}, last_sequence{};
    std::int64_t first_arrival{}, last_arrival{};
    int epoch{}, opened{}, queued{}, queue_failures{};
    bool done{false};

    Impl(LiveArchiveConfig value, std::string session, std::string source, Callback callback)
        : config(std::move(value)), session_id(std::move(session)), source_id(std::move(source)),
          finalized(std::move(callback)) {
        if (!config.enabled) return;
        try {
            if (!safe_id(session_id) || !safe_id(source_id) || config.root.empty() ||
                config.segment_seconds < 1 || config.segment_seconds > 600 ||
                config.max_frames < 1 || config.max_frames > 1000 ||
                config.max_segments_per_session < 1 || config.max_segments_per_session > 8 ||
                config.max_total_segments < 1 || config.max_total_segments > 8 ||
                config.max_segments_per_session > config.max_total_segments ||
                config.raw_limit < 64 * 1024 || config.raw_limit > 12 * mib ||
                config.max_bytes < reservation || config.max_bytes > 256 * mib ||
                !std::isfinite(config.output_fps) || config.output_fps < 1 || config.output_fps > 60)
                throw std::runtime_error("archive_invalid_configuration");
            config.root = fs::absolute(config.root).lexically_normal();
            const auto usage = scan(config.root);
            if (usage.slots >= config.max_total_segments || accounted_bytes(usage) > config.max_bytes - reservation)
                throw std::runtime_error("archive_disk_quota");
            const auto session_path = config.root / session_id;
            inside(config.root, session_path);
            if (fs::exists(session_path)) throw std::runtime_error("archive_session_exists");
            state = "idle";
        } catch (const std::exception& e) { fail(e.what()); }
        catch (...) { fail("archive_initialization_failed"); }
    }
    void fail(const std::string& reason) {
        writer.abandon();
        // Deliberately generic stable codes: no filesystem paths or camera URLs.
        error = reason.starts_with("archive_") && reason.size() < 96 ? reason : "archive_storage_failed";
        state = "error"; done = true;
    }
    void reserve_slot(const LiveFrame& frame, int tracking_epoch) {
        if (opened >= config.max_segments_per_session) {
            state = "limit_reached"; error = "archive_session_segment_limit"; done = true; return;
        }
        const auto usage = scan(config.root);
        if (usage.slots >= config.max_total_segments || accounted_bytes(usage) > config.max_bytes - reservation) {
            state = "limit_reached"; error = "archive_disk_quota"; done = true; return;
        }
        const auto session_path = config.root / session_id;
        inside(config.root, session_path);
        if (opened == 0 && fs::exists(session_path)) throw std::runtime_error("archive_session_exists");
        std::ostringstream name; name << "segment-" << std::setw(4) << std::setfill('0') << opened + 1;
        const auto directory = session_path / name.str();
        inside(config.root, directory);
        if (fs::exists(directory)) throw std::runtime_error("archive_output_exists");
        fs::create_directories(session_path);
        inside(config.root, directory);
        if (!fs::create_directory(directory)) throw std::runtime_error("archive_output_exists");
        ++opened;
        segment = {directory, directory / "raw.avi", directory / "clip.mp4", directory / "manifest.json",
            session_id, source_id, opened, 0, config.output_fps};
        inside(config.root, segment.raw_path);
        if (fs::exists(segment.raw_path)) throw std::runtime_error("archive_output_exists");
        writer.open(segment.raw_path, frame.image.size(), config.output_fps);
        size = frame.image.size(); source_session = frame.session; epoch = tracking_epoch;
        first_arrival = last_arrival = frame.arrival_ms; last_sequence = frame.sequence;
        frame_map = Json::array(); state = "recording";
    }
    void seal() {
        if (!writer.is_open()) return;
        writer.seal();
        inside(config.root, segment.raw_path);
        const auto raw_bytes = fs::file_size(segment.raw_path);
        if (segment.frames == 0 || raw_bytes == 0 || raw_bytes > config.raw_limit)
            throw std::runtime_error("archive_raw_limit");
        Json manifest{{"version", 1}, {"session_id", session_id}, {"source_id", source_id},
            {"segment_index", segment.index}, {"frames", segment.frames}, {"output_fps", segment.fps},
            {"source_fps", segment.fps}, {"source_session", source_session}, {"tracking_epoch", epoch},
            {"width", size.width}, {"height", size.height}, {"raw_file", "raw.avi"},
            {"media_file", "clip.mp4"}, {"raw_bytes", raw_bytes}, {"raw_sealed", true},
            {"index_status", "pending"}, {"time_basis", "decode_arrival_elapsed_ms_not_camera_pts"},
            {"playback_time_basis", "analyzed_frame_index_divided_by_output_fps_cfr"},
            {"arrival_start_ms", first_arrival}, {"arrival_end_ms", last_arrival}, {"frame_map", frame_map}};
        new_json(config.root, segment.manifest_path, manifest);
        new_json(config.root, segment.directory / "sealed.json",
            {{"raw_sha256", file_sha256(config.root, segment.raw_path, config.raw_limit)},
             {"manifest_sha256", file_sha256(config.root, segment.manifest_path, mib)}});
        Json record{{"segment_index", segment.index}, {"frames", segment.frames}, {"index_status", "pending"}};
        closed.push_back(record);
        // Callback admission is independent of a successful durable recording.
        if (finalized) {
            bool admitted = false;
            try {
                finalized(segment);
                admitted = true;
            } catch (...) {
                ++queue_failures;
                error = "archive_index_queue_unavailable";
                new_json(config.root, segment.directory / "index-pending.json",
                    {{"index_status", "pending"}, {"error", "index_queue_admission_failed"}});
            }
            if (admitted) {
                ++queued; closed.back()["index_status"] = "queued";
                new_json(config.root, segment.directory / "index-queued.json", {{"index_status", "queued"}});
            }
        }
        segment.frames = 0; frame_map = Json::array(); state = "idle";
    }
    void accept(const LiveFrame& frame, int tracking_epoch) {
        if (!config.enabled || done) return;
        if (frame.image.empty() || frame.image.type() != CV_8UC3 || frame.image.cols < 2 ||
            frame.image.rows < 2 || frame.image.cols % 2 != 0 || frame.image.rows % 2 != 0 ||
            frame.image.cols > 1920 || frame.image.rows > 1080 || frame.arrival_ms < 0 ||
            frame.sequence == 0 || frame.session == 0 || tracking_epoch < 1)
            throw std::runtime_error("archive_invalid_frame");
        if (writer.is_open() && (frame.session != source_session || tracking_epoch != epoch ||
            frame.image.size() != size || frame.arrival_ms < last_arrival || frame.sequence <= last_sequence ||
            frame.arrival_ms - first_arrival >= static_cast<std::int64_t>(config.segment_seconds) * 1000 ||
            segment.frames >= config.max_frames)) seal();
        // The original, unannotated pixels are JPEG encoded once. No resized
        // boxes or sampled detector result image is stored in this raw archive.
        std::vector<unsigned char> jpeg;
        if (!cv::imencode(".jpg", frame.image, jpeg, {cv::IMWRITE_JPEG_QUALITY, 80}))
            throw std::runtime_error("archive_encoder_preflight_failed");
        if (224 + jpeg.size() + (jpeg.size() % 2) + 8 + 8 + 16 > config.raw_limit)
            throw std::runtime_error("archive_frame_too_large");
        if (writer.is_open() && !writer.fits(jpeg.size(), config.raw_limit)) seal();
        if (!writer.is_open()) reserve_slot(frame, tracking_epoch);
        if (done) return;
        inside(config.root, segment.raw_path);
        const auto usage = scan(config.root);
        if (accounted_bytes(usage) > config.max_bytes || usage.slots > config.max_total_segments)
            throw std::runtime_error("archive_disk_quota");
        writer.write(jpeg);
        frame_map.push_back({{"frame_index", segment.frames}, {"arrival_ms", frame.arrival_ms},
            {"source_sequence", frame.sequence}, {"source_session", frame.session}, {"tracking_epoch", tracking_epoch}});
        ++segment.frames; last_arrival = frame.arrival_ms; last_sequence = frame.sequence;
        if (fs::file_size(segment.raw_path) > config.raw_limit) throw std::runtime_error("archive_raw_limit");
    }
};

LiveArchive::LiveArchive(LiveArchiveConfig config, std::string session_id, std::string source_id, Callback finalized)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(session_id), std::move(source_id), std::move(finalized))) {}
LiveArchive::~LiveArchive() { finish(); }
void LiveArchive::accept(const LiveFrame& frame, int tracking_epoch) noexcept {
    try { impl_->accept(frame, tracking_epoch); }
    catch (const std::exception& e) { impl_->fail(e.what()); }
    catch (...) { impl_->fail("archive_storage_failed"); }
}
void LiveArchive::finish() noexcept {
    if (!impl_ || impl_->done || !impl_->config.enabled) return;
    try { impl_->seal(); impl_->done = true; impl_->state = "finished"; }
    catch (const std::exception& e) { impl_->fail(e.what()); }
    catch (...) { impl_->fail("archive_storage_failed"); }
}
nlohmann::json LiveArchive::snapshot() const {
    return {{"enabled", impl_->config.enabled}, {"state", impl_->state}, {"error", impl_->error},
        {"closed_segments", impl_->closed.size()}, {"opened_segments", impl_->opened},
        {"current_frames", impl_->segment.frames}, {"queued_segments", impl_->queued},
        {"index_queue_failures", impl_->queue_failures}, {"segments", impl_->closed},
        {"output_fps", impl_->config.output_fps}, {"time_basis", "decode_arrival_not_camera_pts"}};
}
} // namespace aegisvision::vision
