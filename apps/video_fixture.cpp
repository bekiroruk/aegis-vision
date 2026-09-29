// Generate a reproducible moving-photo clip for local smoke testing, not a tracking benchmark.
#include "aegisvision/vision.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    try {
        if (argc != 3) throw std::runtime_error("Usage: aegisvision_video_fixture IMAGE OUTPUT.avi");
        const auto source = aegisvision::vision::load_image(argv[1]);
        const std::filesystem::path path(argv[2]);
        if (std::filesystem::exists(path)) throw std::runtime_error("Fixture output already exists");
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        cv::Mat base;
        cv::resize(source, base, {640, 480});
        const auto u8 = path.u8string();
        cv::VideoWriter writer(std::string(u8.begin(), u8.end()), cv::CAP_OPENCV_MJPEG,
            cv::VideoWriter::fourcc('M','J','P','G'), 10, base.size());
        if (!writer.isOpened()) throw std::runtime_error("Cannot open fixture writer");
        for (int index = 0; index < 30; ++index) {
            cv::Mat frame;
            const cv::Mat motion = (cv::Mat_<double>(2,3) << 1,0,index*0.4,0,1,index*0.2);
            cv::warpAffine(base, frame, motion, base.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
            writer.write(frame);
        }
        writer.release();
        std::cout << "Generated 30-frame moving-photo fixture at 10 fps; not a real video benchmark\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
