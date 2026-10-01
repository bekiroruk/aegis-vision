#include "aegisvision/clip_encoder.hpp"
#include <opencv2/videoio.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using namespace aegisvision::vision;
using namespace std::chrono_literals;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> std::string rejects(F operation, const char* message) {
    try { operation(); } catch (const std::exception& error) { return error.what(); }
    throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& value) {
    std::ofstream file(path, std::ios::binary); file << value;
    require(static_cast<bool>(file), "Fixture write failed");
}
std::string read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void fixture(const fs::path& path, int frames = 6, cv::Size size = {160,120}) {
    cv::VideoWriter writer(path.string(), cv::CAP_OPENCV_MJPEG,
        cv::VideoWriter::fourcc('M','J','P','G'), 10, size);
    require(writer.isOpened(), "MJPEG fixture writer unavailable");
    for (int i = 0; i < frames; ++i)
        writer.write(cv::Mat(size, CV_8UC3, cv::Scalar(20 + i * 10, 60, 90)));
    writer.release();
}
fs::path self_path(char* argument) {
#ifdef _WIN32
    std::vector<wchar_t> path(32768);
    const auto count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    require(count > 0 && count < path.size(), "Test executable path unavailable");
    (void)argument;
    return fs::path(path.data());
#else
    return fs::absolute(argument);
#endif
}
// The test executable doubles as a deterministic native child. Its argument
// vector is supplied exclusively by the encoder, never by a shell/script.
int fake_child(int count, char** arguments) {
    fs::path input, output;
    for (int i = 1; i < count; ++i) {
        if (std::string(arguments[i]) == "-i" && i + 1 < count) input = arguments[++i];
    }
    if (count > 1) output = arguments[count - 1];
    if (input.empty() || output.empty()) return 81;
    std::ofstream observed(input.parent_path() / "child-argv.txt", std::ios::binary);
    for (int i = 1; i < count; ++i) observed << arguments[i] << '\n';
    observed.close();
    const auto name = input.filename().string();
    if (name.find("hang") != std::string::npos) {
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = getpid();
#endif
        write(input.parent_path() / "child.pid", std::to_string(pid));
        std::this_thread::sleep_for(60s);
        return 0;
    }
    if (name.find("bad-output") != std::string::npos) { write(output, "not a video"); return 0; }
    if (name.find("oversized") != std::string::npos) { write(output, std::string(2048, 'x')); return 0; }
    if (name.find("race-output") != std::string::npos) {
        fs::copy_file(input.parent_path() / "template.mp4", output);
        write(input.parent_path() / "race.mp4", "existing destination wins");
        return 0;
    }
    return 17;
}
void reaped_child(const fs::path& pid_file) {
    const auto pid = std::stoul(read(pid_file));
#ifdef _WIN32
    HANDLE child = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (child) {
        const auto stopped = WaitForSingleObject(child, 0) == WAIT_OBJECT_0;
        CloseHandle(child);
        require(stopped, "Owned encoder process is still alive");
    }
#else
    int status{};
    errno = 0;
    require(waitpid(static_cast<pid_t>(pid), &status, WNOHANG) == -1 && errno == ECHILD,
            "Owned encoder process was not reaped");
#endif
}
bool executable_available(const fs::path& executable) {
#ifdef _WIN32
    std::vector<wchar_t> resolved(32768);
    const auto count = SearchPathW(nullptr, executable.c_str(), L".exe", static_cast<DWORD>(resolved.size()), resolved.data(), nullptr);
    return count > 0 && count < resolved.size();
#else
    if (executable.has_parent_path()) return access(executable.c_str(), X_OK) == 0;
    const char* path = std::getenv("PATH");
    if (!path) return false;
    const std::string value(path);
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const auto end = value.find(':', begin);
        const auto directory = value.substr(begin, end == std::string::npos ? value.size() - begin : end - begin);
        if (access((fs::path(directory.empty() ? "." : directory) / executable).c_str(), X_OK) == 0) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
#endif
}
void checks(const fs::path& root, const fs::path& self, const fs::path& ffmpeg, bool integration_required) {
    const auto input = root / "literal & (space) source.avi";
    fixture(input);
    const auto input_bytes = fs::file_size(input);
    ArchiveEncodeConfig config; config.executable = self; config.expected_frames = 6; config.fps = 10;
    std::atomic_bool cancel{false};
    const auto output = root / "literal & (space) output.mp4";
    const auto partial = root / "literal & (space) output.partial.mp4";
    const auto invoke = [&] { return encode_archive_clip(input, output, config, cancel); };

    const auto failure = rejects(invoke, "Nonzero child exit accepted");
    require(failure.find("literal") == std::string::npos && failure.find(root.string()) == std::string::npos,
            "Encoder error leaked a path");
    const auto argv = read(root / "child-argv.txt");
    require(argv.find(fs::absolute(input).string() + "\n") != std::string::npos &&
            argv.find(fs::absolute(partial).string() + "\n") != std::string::npos,
            "Native child argv lost literal spaces/metacharacters");
    require(argv.find("libx264\n") != std::string::npos && argv.find("yuv420p\n") != std::string::npos &&
            argv.find("+faststart\n") != std::string::npos && argv.find("-nostdin\n") != std::string::npos,
            "Required H.264 arguments missing");
    require(!fs::exists(output) && !fs::exists(partial) && fs::file_size(input) == input_bytes,
            "Failed child changed source or published output");

    write(output, "keep final");
    (void)rejects(invoke, "Existing final file overwritten");
    require(read(output) == "keep final", "Existing final was changed");
    fs::remove(output); // Exact test-owned fixture, never a user archive.
    write(partial, "keep partial");
    (void)rejects(invoke, "Existing partial file overwritten");
    require(read(partial) == "keep partial", "Existing partial was changed");
    fs::remove(partial);
    config.executable = root / "missing-native-encoder";
    (void)rejects(invoke, "Missing executable accepted");
    config.executable = self;
    cancel = true;
    require(rejects(invoke, "Initial cancellation ignored").find("cancelled") != std::string::npos,
            "Cancellation reason missing");
    cancel = false;
    config.expected_frames = 5;
    (void)rejects(invoke, "Extra input frames accepted");
    config.expected_frames = 7;
    (void)rejects(invoke, "Truncated input frame count accepted");
    config.expected_frames = 6; config.fps = 11;
    (void)rejects(invoke, "Manifest FPS mismatch accepted");
    config.fps = std::numeric_limits<double>::quiet_NaN();
    (void)rejects(invoke, "NaN FPS accepted");
    config.fps = 10;
    write(root / "invalid.avi", "not an AVI");
    (void)rejects([&] { return encode_archive_clip(root / "invalid.avi", output, config, cancel); }, "Invalid AVI accepted");
    fixture(root / "empty.avi", 0);
    (void)rejects([&] { return encode_archive_clip(root / "empty.avi", output, config, cancel); }, "Empty AVI accepted");
    fixture(root / "odd.avi", 6, {161,121});
    (void)rejects([&] { return encode_archive_clip(root / "odd.avi", output, config, cancel); }, "Odd dimensions accepted");
    std::error_code ec;
    fs::create_symlink(input, root / "symbolic.avi", ec);
    if (!ec) (void)rejects([&] { return encode_archive_clip(root / "symbolic.avi", output, config, cancel); }, "Symbolic input accepted");
    else std::cout << "SKIP symbolic-link fixture (OS permission unavailable)\n";

    fixture(root / "bad-output.avi");
    (void)rejects([&] { return encode_archive_clip(root / "bad-output.avi", output, config, cancel); }, "Undecodable child output accepted");
    require(!fs::exists(partial) && !fs::exists(output), "Invalid owned partial not removed");
    fixture(root / "oversized.avi");
    config.max_output_bytes = 1024;
    (void)rejects([&] { return encode_archive_clip(root / "oversized.avi", output, config, cancel); }, "Oversized child output accepted");
    require(!fs::exists(partial) && !fs::exists(output), "Oversized owned partial not removed");
    config.max_output_bytes = 12ULL * 1024 * 1024;

    fixture(root / "hang.avi");
    config.timeout = 1500ms;
    const auto began = std::chrono::steady_clock::now();
    require(rejects([&] { return encode_archive_clip(root / "hang.avi", output, config, cancel); }, "Hung child accepted").find("timed out") != std::string::npos,
            "Child timeout reason missing");
    require(std::chrono::steady_clock::now() - began < 4s, "Child deadline was not bounded");
    require(fs::exists(root / "child.pid"), "Hung native child did not start");
    reaped_child(root / "child.pid");
    fs::remove(root / "child.pid");

    config.timeout = 5s;
    std::jthread stopper([&] {
        const auto deadline = std::chrono::steady_clock::now() + 4s;
        while (!fs::exists(root / "child.pid") && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
        cancel = true;
    });
    const auto cancellation = rejects([&] { return encode_archive_clip(root / "hang.avi", output, config, cancel); }, "Running child cancellation ignored");
    stopper.join();
    require(cancellation.find("cancelled") != std::string::npos, "Running child cancellation reason missing");
    reaped_child(root / "child.pid");
    cancel = false;
    config.timeout = 30s;

    if (!executable_available(ffmpeg)) {
        require(!integration_required, "Required FFmpeg executable not found");
        std::cout << "SKIP real H.264 round-trip (FFmpeg not on PATH; --ffmpeg makes it required)\n";
        return;
    }
    config.executable = ffmpeg;
    const auto result = encode_archive_clip(input, output, config, cancel);
    require(result.frames == 6 && result.width == 160 && result.height == 120 &&
            std::abs(result.fps - 10) < 0.01 && result.bytes > 0 && result.bytes <= config.max_output_bytes,
            "H.264 result metadata is incorrect");
    require(!fs::exists(partial) && fs::file_size(input) == input_bytes, "Successful encoding changed source or left partial");
    const auto replay = validate_archive_clip(input, output, config, cancel);
    require(replay.frames == result.frames && replay.bytes == result.bytes, "Durable replay validation changed clip metadata");
    const auto mp4 = read(output);
    require(mp4.find("moov") != std::string::npos && mp4.find("mdat") != std::string::npos &&
            mp4.find("moov") < mp4.find("mdat"), "MP4 faststart metadata was not moved before video data");
    (void)rejects(invoke, "Successful output overwritten by retry");
    require(read(output) == mp4, "Retry changed successful final output");
    config.expected_frames = 5;
    (void)rejects([&] { return validate_archive_clip(input, output, config, cancel); }, "Replay accepted wrong frame count");
    config.expected_frames = 6;
    fs::copy_file(output, root / "template.mp4");
    fixture(root / "race-output.avi");
    config.executable = self;
    (void)rejects([&] { return encode_archive_clip(root / "race-output.avi", root / "race.mp4", config, cancel); }, "Racing final was overwritten");
    require(read(root / "race.mp4") == "existing destination wins" && !fs::exists(root / "race.partial.mp4"),
            "No-clobber publication lost existing final or leaked owned partial");
    std::cout << "PASS actual FFmpeg H.264 round-trip, faststart, durable replay and no-clobber publication\n";
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "-nostdin") {
        try { return fake_child(argc, argv); } catch (...) { return 82; }
    }
    const auto root = fs::temp_directory_path() / ("aegis-clip-encoder-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) / "space & (literal)";
    try {
        const bool integration_required = argc > 1;
        require(!integration_required || std::string(argv[1]) == "--ffmpeg", "Usage: test_clip_encoder [--ffmpeg [path]]");
        const fs::path ffmpeg = argc > 2 ? argv[2] : "ffmpeg";
        fs::create_directories(root);
        checks(root, self_path(argv[0]), ffmpeg, integration_required);
        fs::remove_all(root.parent_path()); // Fresh exact test-owned temporary directory.
        std::cout << "PASS native child argv, limits, cancellation, timeout/reaping and input validation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\nTest fixtures retained at " << root.string() << '\n';
        return 1;
    }
}
