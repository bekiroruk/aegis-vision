#include "aegisvision/vision.hpp"
#include "sample_scene.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace aegisvision;
namespace fs = std::filesystem;

void help() {
    std::cout << "AegisVision image tools (OpenCV)\n"
        "  aegisvision_image inspect INPUT\n"
        "  aegisvision_image annotate INPUT BOXES.tsv OUTPUT.png\n"
        "  aegisvision_image align SOURCE TARGET OUTPUT_DIRECTORY\n"
        "  aegisvision_image demo OUTPUT_DIRECTORY\n"
        "BOXES.tsv: x1 y1 x2 y2 score label (whitespace separated; quote labels with spaces).\n"
        "annotate renders supplied boxes; it does not run an object detector.\n";
}

std::vector<Detection> load_boxes(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open boxes file");
    std::vector<Detection> boxes;
    std::string line;
    int number = 0;
    while (std::getline(input, line)) {
        ++number;
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream row(line);
        float x1, y1, x2, y2, score;
        std::string label, extra;
        if (!(row >> x1 >> y1 >> x2 >> y2 >> score >> std::quoted(label)) ||
            label.empty() || (row >> extra)) {
            throw std::runtime_error("Invalid boxes row " + std::to_string(number));
        }
        boxes.push_back({BoundingBox{x1, y1, x2, y2}, label, score, {}, {}});
    }
    return boxes;
}

void print_metrics(const vision::AlignmentResult& result) {
    std::cout << "matches=" << result.candidate_matches << " inliers=" << result.inliers
        << " inlier_ratio=" << result.inlier_ratio
        << " reprojection_rmse_px=" << result.reprojection_rmse_px << '\n';
}

int run(const std::vector<fs::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") { help(); return 0; }
        if (args.size() == 3 && args[1] == "inspect") {
            const auto image = vision::load_image(args[2]);
            std::cout << "width=" << image.cols << " height=" << image.rows << " channels=3\n";
        } else if (args.size() == 5 && args[1] == "annotate") {
            const auto image = vision::load_image(args[2]);
            vision::save_image(args[4], vision::annotate(image, load_boxes(args[3])));
        } else if (args.size() == 5 && args[1] == "align") {
            const auto result = vision::align(vision::load_image(args[2]), vision::load_image(args[3]));
            vision::save_alignment(args[4], result);
            print_metrics(result);
        } else if (args.size() == 3 && args[1] == "demo") {
            const auto source = sample_scene();
            cv::Mat target;
            cv::warpPerspective(source, target, sample_transform(), source.size());
            vision::save_image(args[2] / "source.png", source);
            vision::save_image(args[2] / "target.png", target);
            // Exercise the file decode route, not just the in-memory fixture.
            const auto result = vision::align(vision::load_image(args[2] / "source.png"),
                vision::load_image(args[2] / "target.png"));
            vision::save_alignment(args[2], result);
            const std::vector<Detection> boxes{
                {BoundingBox{70, 130, 260, 300}, "manual A-17", 1.0F, {}, {}},
                {BoundingBox{430, 300, 640, 440}, "manual B-42", 1.0F, {}, {}},
            };
            vision::save_image(args[2] / "annotated.png", vision::annotate(source, boxes));
            vision::save_image(args[2] / "crop.png", vision::crop(source, boxes.front().bbox));
            print_metrics(result);
        } else {
            help();
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
