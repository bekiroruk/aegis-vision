#include "aegisvision/crossview.hpp"
#include "aegisvision/reid.hpp"
#include "aegisvision/vision.hpp"
#include "aegisvision/yolo.hpp"
#include <nlohmann/json.hpp>
#include <picosha2.h>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <set>
#include <cmath>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;
using Json = nlohmann::json;
struct Counts {
    std::size_t queries{}, shared{}, rank1{}, tp{}, fp{}, missed{}, unknown{}, unknown_rejected{};
    void add(const CrossViewDecision& d) {
        ++queries; shared += d.has_counterpart; rank1 += d.has_counterpart && d.rank1_correct;
        tp += d.correct; fp += d.accepted && !d.correct; missed += d.has_counterpart && !d.correct;
        unknown += !d.has_counterpart; unknown_rejected += !d.has_counterpart && !d.accepted;
    }
    Json json() const {
        auto rate = [](std::size_t n, std::size_t d) -> Json { return d ? Json(static_cast<double>(n) / d) : Json(nullptr); };
        return {{"queries", queries}, {"queries_with_counterpart", shared}, {"rank1_correct", rank1},
            {"correct_accepts", tp}, {"wrong_accepts", fp}, {"missed_counterparts", missed},
            {"queries_without_counterpart", unknown}, {"correct_unknown_rejections", unknown_rejected},
            {"closed_set_rank1", rate(rank1, shared)}, {"accepted_precision", rate(tp, tp + fp)},
            {"counterpart_recall", rate(tp, shared)}, {"unknown_rejection_rate", rate(unknown_rejected, unknown)}};
    }
};
std::string hash(const fs::path& path) {
    if (!fs::is_regular_file(path) || fs::file_size(path) > 32 * 1024 * 1024)
        throw std::invalid_argument("Missing or oversized input asset");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read input asset");
    auto result = picosha2::hash256_hex_string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("Input hash failed");
    return result;
}
std::int64_t integer(const Json& v, std::int64_t low, std::int64_t high) {
    if (!v.is_number_integer() || (v.is_number_unsigned() && v.get<std::uint64_t>() > static_cast<std::uint64_t>(high)))
        throw std::invalid_argument("Expected bounded integer");
    auto n = v.get<std::int64_t>();
    if (n < low || n > high) throw std::invalid_argument("Integer out of bounds");
    return n;
}
fs::path asset(const fs::path& base, const Json& view) {
    const auto relative = fs::u8path(view.at("image").get<std::string>());
    if (relative.empty() || relative.is_absolute() || relative.has_root_name()) throw std::invalid_argument("Asset must be relative");
    for (const auto& component : relative) if (component == "..") throw std::invalid_argument("Asset escapes dataset");
    const auto resolved = fs::canonical(base / relative);
    auto relative_resolved = resolved.lexically_relative(fs::canonical(base));
    if (relative_resolved.empty() || *relative_resolved.begin() == "..") throw std::invalid_argument("Asset symlink escapes dataset");
    if (hash(resolved) != view.at("sha256").get<std::string>()) throw std::invalid_argument("Image SHA256 mismatch");
    return resolved;
}
BoundingBox box(const Json& person) {
    const auto& b = person.at("bbox");
    if (!b.is_array() || b.size() != 4) throw std::invalid_argument("Invalid person box");
    const auto x1 = integer(b[0], 0, 1919), y1 = integer(b[1], 0, 1079);
    const auto x2 = integer(b[2], 1, 1920), y2 = integer(b[3], 1, 1080);
    if (x2 - x1 < 20 || y2 - y1 < 64) throw std::invalid_argument("Person box violates fixed size filter");
    return {static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2), static_cast<float>(y2)};
}
void validate_people(const Json& view) {
    const auto& persons = view.at("persons");
    if (!persons.is_array() || persons.size() > 512) throw std::invalid_argument("Invalid person count");
    std::set<std::int64_t> ids;
    for (const auto& p : persons) {
        if (!ids.insert(integer(p.at("identity"), 0, INT64_MAX)).second) throw std::invalid_argument("Duplicate person label");
        static_cast<void>(box(p));
    }
}
std::vector<CrossViewPerson> embed(const Json& view, const cv::Mat& pixels, ReIdEmbedder& encoder) {
    const auto frame = vision::image_frame(pixels, "crossview", view.at("camera").get<std::string>());
    std::vector<CrossViewPerson> result;
    for (const auto& p : view.at("persons")) {
        // The identity never enters model inference or matching.
        Detection detection{box(p), "person", 1.F, {}, {}};
        auto feature = encoder.embed_image(frame, detection);
        result.push_back({static_cast<std::uint64_t>(integer(p.at("identity"), 0, INT64_MAX)), std::move(feature)});
    }
    return result;
}
void review(const fs::path& path, const cv::Mat& left, const cv::Mat& right,
    const Json& q, const Json& g, const std::vector<CrossViewDecision>& decisions) {
    const auto count = std::min<std::size_t>(8, decisions.size());
    if (!count) return;
    cv::Mat sheet(55 + static_cast<int>(count) * 170, 900, CV_8UC3, cv::Scalar(24, 24, 24));
    cv::putText(sheet, "First 8 queries | C1 -> C2 | GT crops, NOT tracking", {12, 30},
        cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    for (std::size_t i = 0; i < count; ++i) {
        const int y = 55 + static_cast<int>(i) * 170;
        auto thumbnail = [&](const cv::Mat& image, const Json& person, int x) {
            auto crop = vision::crop(image, box(person));
            cv::Mat resized; cv::resize(crop, resized, {80, 160});
            resized.copyTo(sheet(cv::Rect(x, y, 80, 160)));
        };
        thumbnail(left, q[i], 10);
        const auto& d = decisions[i];
        if (d.best_gallery) thumbnail(right, g[*d.best_gallery], 110);
        const auto color = d.accepted ? (d.correct ? cv::Scalar(80, 230, 100) : cv::Scalar(80, 80, 255)) : cv::Scalar(80, 200, 240);
        const auto id = q[i].at("identity").dump();
        const auto candidate = d.best_gallery ? g[*d.best_gallery].at("identity").dump() : "none";
        cv::putText(sheet, "GT person " + id + " | best candidate " + candidate, {210, y + 35},
            cv::FONT_HERSHEY_SIMPLEX, .6, color, 1, cv::LINE_AA);
        cv::putText(sheet, d.accepted ? (d.correct ? "ACCEPTED / CORRECT" : "ACCEPTED / WRONG") : "REJECTED: " + d.reason,
            {210, y + 70}, cv::FONT_HERSHEY_SIMPLEX, .55, color, 1, cv::LINE_AA);
        cv::putText(sheet, "cosine=" + (d.similarity ? std::to_string(*d.similarity) : "none") +
            " | counterpart=" + (d.has_counterpart ? "yes" : "no"), {210, y + 105},
            cv::FONT_HERSHEY_SIMPLEX, .55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    }
    vision::save_image(path, sheet);
}
int run(const std::vector<fs::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") {
            std::cout << "aegisvision_crossview MANIFEST.json OSNET_BUNDLE NEW_OUTPUT_DIR\n"
                         "Same-frame C1->C2 oracle-crop diagnostic. No detection/tracking/handoff score.\n";
            return 0;
        }
        if (args.size() != 4) throw std::invalid_argument("Use --help");
        if (fs::exists(args[3])) throw std::invalid_argument("Output directory must not exist");
        const auto manifest_hash = hash(args[1]);
        std::ifstream input(args[1]);
        const auto j = Json::parse(input, [](int depth, Json::parse_event_t, Json&) {
            if (depth > 16) throw std::invalid_argument("JSON nesting too deep"); return true;
        });
        integer(j.at("version"), 1, 1);
        if (j.at("protocol") != "wildtrack-c1-c2-same-frame-gt-crops-v1" ||
            j.at("cameras") != Json::array({"C1", "C2"}) || j.at("min_similarity") != .85 || j.at("ambiguity_margin") != .05)
            throw std::invalid_argument("Unsupported protocol or changed frozen thresholds");
        const auto& frames = j.at("frames");
        if (!frames.is_array() || frames.size() != 16) throw std::invalid_argument("Expected fixed 16-frame subset");
        const auto base = fs::absolute(args[1]).parent_path();
        std::vector<std::pair<fs::path, fs::path>> paths;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            if (integer(frames[i].at("frame"), 0, 1999) != static_cast<std::int64_t>(i * 125))
                throw std::invalid_argument("Unexpected subset frame");
            const auto& views = frames[i].at("views");
            if (!views.is_array() || views.size() != 2 || views[0].at("camera") != "C1" || views[1].at("camera") != "C2")
                throw std::invalid_argument("Expected C1 and C2 views in order");
            validate_people(views[0]); validate_people(views[1]);
            paths.emplace_back(asset(base, views[0]), asset(base, views[1]));
        }
        cv::setNumThreads(1);
        ReIdEmbedder encoder(args[2]);
        const auto output = fs::absolute(args[3]);
        fs::create_directories(output.parent_path());
        if (!fs::create_directory(output)) throw std::runtime_error("Cannot reserve output");
        Counts total;
        Json cases = Json::array();
        for (std::size_t i = 0; i < frames.size(); ++i) {
            const auto left = vision::load_image(paths[i].first), right = vision::load_image(paths[i].second);
            if (left.size() != cv::Size(1920, 1080) || right.size() != left.size()) throw std::invalid_argument("Unexpected image resolution");
            const auto& views = frames[i].at("views");
            const auto q = embed(views[0], left, encoder), g = embed(views[1], right, encoder);
            const auto decisions = evaluate_crossview(q, g, .85, .05);
            Counts counts;
            Json queries = Json::array();
            for (std::size_t k = 0; k < decisions.size(); ++k) {
                const auto& d = decisions[k]; total.add(d); counts.add(d);
                queries.push_back({{"query_identity", q[k].identity},
                    {"best_gallery_identity", d.best_gallery ? Json(g[*d.best_gallery].identity) : Json(nullptr)},
                    {"similarity", d.similarity ? Json(*d.similarity) : Json(nullptr)},
                    {"accepted", d.accepted}, {"correct", d.correct}, {"has_counterpart", d.has_counterpart},
                    {"rank1_correct", d.rank1_correct}, {"reason", d.reason}});
            }
            cases.push_back({{"frame", frames[i].at("frame")}, {"gallery_size", g.size()}, {"metrics", counts.json()}, {"queries", queries}});
            if (i == 0) review(output / "review.jpg", left, right, views[0].at("persons"), views[1].at("persons"), decisions);
            std::cout << "frame=" << frames[i].at("frame") << " queries=" << q.size() << " accepted_correct=" << counts.tp
                      << " accepted_wrong=" << counts.fp << std::endl;
        }
        Json report{{"version", 1}, {"complete", true}, {"protocol", j.at("protocol")}, {"manifest_sha256", manifest_hash},
            {"embedding_space", encoder.space_id()}, {"min_similarity", .85}, {"ambiguity_margin", .05},
            {"metrics", total.json()}, {"frames", cases}, {"handoff_evaluated", false}, {"detector_evaluated", false},
            {"warning", "Fixed diagnostic subset; oracle GT boxes; same-frame cross-camera appearance only. Correlated repeated persons. No threshold tuning or production acceptance."}};
        std::ofstream out(output / "report.json.part"); out << report.dump(2) << '\n'; out.close();
        if (!out) throw std::runtime_error("Cannot write report");
        fs::rename(output / "report.json.part", output / "report.json");
        std::cout << total.json().dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<fs::path>(argv, argv + argc));
}
