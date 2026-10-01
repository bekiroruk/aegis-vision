#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "aegisvision/clip_encoder.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <cwctype>
#include <iomanip>
#include <locale>
#include <sstream>
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
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace aegisvision::vision {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t max_input_bytes = 12ULL * 1024 * 1024;
class EncodeError : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};

void checkpoint(const std::atomic_bool& cancel, Clock::time_point deadline) {
    if (cancel.load()) throw EncodeError("Archive encoding cancelled");
    if (Clock::now() >= deadline) throw EncodeError("Archive encoding timed out");
}
bool present(const fs::path& path) {
    std::error_code ec;
    const auto status = fs::symlink_status(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
        throw EncodeError("Archive path cannot be inspected");
    return status.type() != fs::file_type::not_found && status.type() != fs::file_type::none;
}
void safe_path(const fs::path& path) {
    if (path.empty() || path.native().find(fs::path::value_type{}) != fs::path::string_type::npos)
        throw EncodeError("Invalid archive path");
    auto current = path.root_path();
    for (const auto& component : path.relative_path()) {
        current /= component;
        std::error_code ec;
        const auto status = fs::symlink_status(current, ec);
        if (ec && ec != std::errc::no_such_file_or_directory)
            throw EncodeError("Archive path cannot be inspected");
        if (fs::is_symlink(status)) throw EncodeError("Archive symbolic links are not supported");
    }
}
void output_bound(const fs::path& path, std::uint64_t limit) {
    if (!present(path)) return;
    if (!fs::is_regular_file(fs::symlink_status(path)))
        throw EncodeError("Archive encoder output is not a regular file");
    if (fs::file_size(path) > limit) throw EncodeError("Archive clip exceeded its byte limit");
}
std::string decimal(double value) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << std::setprecision(12) << value;
    return text.str();
}

struct Decoded {
    std::uint64_t frames{};
    int width{}, height{};
    double fps{};
};
Decoded verify_video(const fs::path& path, const ArchiveEncodeConfig& config,
                     const std::atomic_bool& cancel, Clock::time_point deadline, bool input) {
    checkpoint(cancel, deadline);
    cv::VideoCapture video(path.string(), input ? cv::CAP_OPENCV_MJPEG : cv::CAP_FFMPEG);
    if (!video.isOpened()) throw EncodeError(input ? "Archive AVI cannot be decoded" : "Encoded archive MP4 cannot be decoded");
    Decoded result;
    result.fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(result.fps) || std::abs(result.fps - config.fps) > std::max(0.01, config.fps * 0.001))
        throw EncodeError("Archive frame rate does not match its manifest");
    const double header_width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    const double header_height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (!std::isfinite(header_width) || !std::isfinite(header_height) ||
        header_width < 2 || header_width > 1920 || header_height < 2 || header_height > 1080 ||
        std::floor(header_width) != header_width || std::floor(header_height) != header_height ||
        static_cast<int>(header_width) % 2 != 0 || static_cast<int>(header_height) % 2 != 0)
        throw EncodeError("Archive frame format is unsupported");
    if (!input) {
        const auto codec = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
        if (codec != cv::VideoWriter::fourcc('a','v','c','1') && codec != cv::VideoWriter::fourcc('H','2','6','4') &&
            codec != cv::VideoWriter::fourcc('h','2','6','4'))
            throw EncodeError("Encoded archive is not H.264");
    }
    cv::Mat frame;
    while (true) {
        checkpoint(cancel, deadline);
        if (!video.read(frame)) break;
        if (frame.empty() || frame.type() != CV_8UC3 || frame.cols < 2 || frame.rows < 2 ||
            frame.cols > 1920 || frame.rows > 1080 || frame.cols % 2 != 0 || frame.rows % 2 != 0)
            throw EncodeError("Archive frame format is unsupported");
        if (result.frames == 0) { result.width = frame.cols; result.height = frame.rows; }
        if (frame.cols != result.width || frame.rows != result.height)
            throw EncodeError("Archive frame dimensions changed");
        if (++result.frames > config.expected_frames)
            throw EncodeError("Archive frame count does not match its manifest");
    }
    checkpoint(cancel, deadline);
    if (result.frames != config.expected_frames)
        throw EncodeError("Archive frame count does not match its manifest");
    return result;
}

#ifdef _WIN32
class Handle {
public:
    explicit Handle(HANDLE value = nullptr) : value_(value) {}
    ~Handle() { if (value_ && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    HANDLE get() const { return value_; }
private: HANDLE value_;
};
// Microsoft CRT argv decoding: a quote is escaped by 2*n+1 backslashes;
// trailing backslashes are doubled before the closing quote. No cmd.exe rules.
std::wstring quoted(const std::wstring& argument) {
    std::wstring result = L"\"";
    std::size_t slashes = 0;
    for (const wchar_t c : argument) {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'\"') result.append(slashes * 2 + 1, L'\\');
        else result.append(slashes, L'\\');
        slashes = 0;
        result += c;
    }
    result.append(slashes * 2, L'\\');
    result += L'\"';
    return result;
}
fs::path resolve_executable(const fs::path& requested) {
    if (requested.empty() || requested.native().find(L'\0') != std::wstring::npos)
        throw EncodeError("Invalid archive encoder executable");
    std::vector<wchar_t> resolved(32768);
    const auto count = SearchPathW(nullptr, requested.c_str(), L".exe", static_cast<DWORD>(resolved.size()), resolved.data(), nullptr);
    if (!count || count >= resolved.size()) throw EncodeError("Archive encoder executable was not found");
    // PATH entries such as Windows WinGet links legitimately point at the real
    // trusted binary. Resolve that link; the application path itself remains
    // explicit and a batch/script interpreter is never selected.
    fs::path path = fs::canonical(fs::path(resolved.data()));
    auto extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t c) { return std::towlower(c); });
    if (extension != L".exe" || !fs::is_regular_file(fs::symlink_status(path)))
        throw EncodeError("Archive encoder must be a native executable");
    return path;
}
void run_encoder(const fs::path& executable, const std::vector<std::string>& arguments,
                 const fs::path& input, const fs::path& partial, const ArchiveEncodeConfig& config,
                 const std::atomic_bool& cancel, Clock::time_point deadline, bool& launched) {
    const auto application = resolve_executable(executable);
    std::wstring command = quoted(application.wstring());
    for (const auto& argument : arguments) {
        command += L' ';
        // Path arguments are converted natively rather than through the ANSI code page.
        if (argument == "@INPUT@") command += quoted(input.wstring());
        else if (argument == "@OUTPUT@") command += quoted(partial.wstring());
        else command += quoted(std::wstring(argument.begin(), argument.end()));
    }
    if (command.size() >= 32767) throw EncodeError("Archive encoder arguments exceed the platform limit");
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle null_handle(CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    Handle job(CreateJobObjectW(nullptr, nullptr));
    if (null_handle.get() == INVALID_HANDLE_VALUE || !job.get())
        throw EncodeError("Archive encoder process resources are unavailable");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
        throw EncodeError("Archive encoder process isolation failed");
    SIZE_T attributes_size = 0;
    (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attributes_size);
    std::vector<unsigned char> attributes(attributes_size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(list, 1, 0, &attributes_size))
        throw EncodeError("Archive encoder process isolation failed");
    struct AttributeGuard { LPPROC_THREAD_ATTRIBUTE_LIST list; ~AttributeGuard() { DeleteProcThreadAttributeList(list); } } attribute_guard{list};
    HANDLE inherited[] = {null_handle.get()};
    if (!UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), nullptr, nullptr))
        throw EncodeError("Archive encoder handle isolation failed");
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.StartupInfo.wShowWindow = SW_HIDE;
    startup.StartupInfo.hStdInput = startup.StartupInfo.hStdOutput = startup.StartupInfo.hStdError = null_handle.get();
    startup.lpAttributeList = list;
    PROCESS_INFORMATION process{};
    checkpoint(cancel, deadline);
    if (!CreateProcessW(application.c_str(), command.data(), nullptr, nullptr, TRUE,
        EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup.StartupInfo, &process))
        throw EncodeError("Archive encoder process could not start");
    Handle process_handle(process.hProcess), thread_handle(process.hThread);
    // Suspended creation makes assignment atomic with respect to child code execution.
    if (!AssignProcessToJobObject(job.get(), process.hProcess) || ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        (void)TerminateProcess(process.hProcess, 1);
        (void)WaitForSingleObject(process.hProcess, INFINITE);
        throw EncodeError("Archive encoder process isolation failed");
    }
    launched = true;
    try {
        while (true) {
            checkpoint(cancel, deadline);
            output_bound(partial, config.max_output_bytes);
            const auto wait = WaitForSingleObject(process.hProcess, 10);
            if (wait == WAIT_OBJECT_0) break;
            if (wait != WAIT_TIMEOUT) throw EncodeError("Archive encoder process wait failed");
        }
        DWORD code = 1;
        if (!GetExitCodeProcess(process.hProcess, &code) || code != 0)
            throw EncodeError("Archive encoder failed; verify FFmpeg and its libx264 encoder");
    } catch (...) {
        (void)TerminateJobObject(job.get(), 1);
        (void)WaitForSingleObject(process.hProcess, INFINITE);
        throw;
    }
}
#else
void run_encoder(const fs::path& executable, const std::vector<std::string>& arguments,
                 const fs::path& input, const fs::path& partial, const ArchiveEncodeConfig& config,
                 const std::atomic_bool& cancel, Clock::time_point deadline, bool& launched) {
    if (executable.empty() || executable.native().find('\0') != std::string::npos)
        throw EncodeError("Invalid archive encoder executable");
    std::vector<std::string> strings{executable.string()};
    for (const auto& argument : arguments)
        strings.push_back(argument == "@INPUT@" ? input.string() : argument == "@OUTPUT@" ? partial.string() : argument);
    std::vector<char*> argv;
    for (auto& value : strings) argv.push_back(value.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    if (posix_spawn_file_actions_init(&actions) != 0) throw EncodeError("Archive encoder process resources are unavailable");
    struct ActionGuard { posix_spawn_file_actions_t* actions; ~ActionGuard() { posix_spawn_file_actions_destroy(actions); } } action_guard{&actions};
    if (posix_spawnattr_init(&attributes) != 0) throw EncodeError("Archive encoder process resources are unavailable");
    struct AttributeGuard { posix_spawnattr_t* attributes; ~AttributeGuard() { posix_spawnattr_destroy(attributes); } } attribute_guard{&attributes};
    int setup = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    setup |= posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    setup |= posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 34)
    setup |= posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#else
    // Fail closed rather than inherit unrelated server sockets on older libc.
    throw EncodeError("Archive encoder requires close-from spawn support (glibc 2.34 or newer)");
#endif
    setup |= posix_spawnattr_setpgroup(&attributes, 0);
    setup |= posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    if (setup != 0) throw EncodeError("Archive encoder process isolation failed");
    checkpoint(cancel, deadline);
    pid_t child{};
    // posix_spawnp avoids unsafe C++/allocator work between fork and exec in this
    // multithreaded service. All strings/file actions were prepared in the parent.
    if (posix_spawnp(&child, strings.front().c_str(), &actions, &attributes, argv.data(), environ) != 0)
        throw EncodeError("Archive encoder executable could not start");
    launched = true;
    bool reaped = false;
    try {
        while (true) {
            checkpoint(cancel, deadline);
            output_bound(partial, config.max_output_bytes);
            siginfo_t info{};
            const auto result = waitid(P_PID, static_cast<id_t>(child), &info, WEXITED | WNOHANG | WNOWAIT);
            if (result == 0 && info.si_pid == child) {
                // Leave the leader as a zombie until the owned group is killed,
                // so its PID cannot be reused for an unrelated process group.
                (void)kill(-child, SIGKILL);
                int status{};
                pid_t waited;
                do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
                reaped = true;
                if (waited != child) throw EncodeError("Archive encoder process wait failed");
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                    throw EncodeError("Archive encoder failed; verify FFmpeg and its libx264 encoder");
                break;
            }
            if (result < 0 && errno != EINTR) {
                if (errno == ECHILD) reaped = true; // Another reaper must not cause an unrelated PID kill.
                throw EncodeError("Archive encoder process wait failed");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (...) {
        if (!reaped) {
            (void)kill(-child, SIGKILL);
            while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        }
        throw;
    }
}
#endif

void publish(const fs::path& partial, const fs::path& output) {
#ifdef _WIN32
    // No MOVEFILE_REPLACE_EXISTING: a racing final output is never overwritten.
    if (!MoveFileExW(partial.c_str(), output.c_str(), MOVEFILE_WRITE_THROUGH))
        throw EncodeError("Verified archive clip could not be published without overwriting");
#else
#ifdef SYS_renameat2
    constexpr unsigned rename_noreplace = 1;
    if (syscall(SYS_renameat2, AT_FDCWD, partial.c_str(), AT_FDCWD, output.c_str(), rename_noreplace) == 0) return;
    if (errno != ENOSYS && errno != EINVAL && errno != EOPNOTSUPP)
        throw EncodeError("Verified archive clip could not be published without overwriting");
#endif
    // Older filesystems: a same-directory hard link gives atomic no-clobber
    // publication too. Only our now-published temporary name is then unlinked.
    if (link(partial.c_str(), output.c_str()) != 0)
        throw EncodeError("Verified archive clip could not be published without overwriting");
    (void)unlink(partial.c_str());
#endif
}
void remove_owned_partial(const fs::path& partial) noexcept {
    std::error_code ec;
    const auto status = fs::symlink_status(partial, ec);
    if (!ec && fs::is_regular_file(status)) (void)fs::remove(partial, ec);
}
void validate_config(const ArchiveEncodeConfig& config) {
    if (config.timeout.count() < 1 || config.timeout > std::chrono::seconds(30) ||
        config.expected_frames == 0 || config.expected_frames > 1000 ||
        !std::isfinite(config.fps) || config.fps < 1 || config.fps > 60 ||
        config.max_output_bytes < 1024 || config.max_output_bytes > max_input_bytes)
        throw EncodeError("Invalid archive encoder limits");
}
std::pair<fs::path, fs::path> validate_paths(const fs::path& input_avi, const fs::path& output_mp4) {
    const auto input = fs::absolute(input_avi).lexically_normal();
    const auto output = fs::absolute(output_mp4).lexically_normal();
    if (input == output || input.extension() != ".avi" || output.extension() != ".mp4")
        throw EncodeError("Archive input/output extensions are invalid");
    safe_path(input); safe_path(output);
    if (!fs::is_regular_file(fs::symlink_status(input)) || fs::file_size(input) == 0 || fs::file_size(input) > max_input_bytes)
        throw EncodeError("Archive AVI must be a nonempty regular file within its byte limit");
    if (!fs::is_directory(fs::symlink_status(output.parent_path())))
        throw EncodeError("Archive output directory is unavailable");
    return {input, output};
}
}  // namespace

EncodedArchiveClip encode_archive_clip(const fs::path& input_avi, const fs::path& output_mp4,
                                      const ArchiveEncodeConfig& config, const std::atomic_bool& cancel) {
    fs::path partial;
    bool launched = false;
    try {
        validate_config(config);
        const auto deadline = Clock::now() + config.timeout;
        checkpoint(cancel, deadline);
        const auto [input, output] = validate_paths(input_avi, output_mp4);
        auto temporary_name = output.stem();
        temporary_name += ".partial.mp4";
        partial = output.parent_path() / temporary_name;
        safe_path(partial);
        if (present(output) || present(partial))
            throw EncodeError("Archive output or partial already exists; verify owned-job recovery first");
        const auto source = verify_video(input, config, cancel, deadline, true);
        const std::vector<std::string> arguments{
            "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-protocol_whitelist", "file",
            "-f", "avi", "-i", "@INPUT@", "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-threads", "1", "-r", decimal(config.fps), "-fps_mode", "cfr", "-movflags", "+faststart",
            "-fs", std::to_string(config.max_output_bytes), "-f", "mp4", "@OUTPUT@"};
        run_encoder(config.executable, arguments, input, partial, config, cancel, deadline, launched);
        checkpoint(cancel, deadline);
        output_bound(partial, config.max_output_bytes);
        if (!present(partial) || fs::file_size(partial) == 0)
            throw EncodeError("Archive encoder produced no video");
        const auto encoded = verify_video(partial, config, cancel, deadline, false);
        if (encoded.width != source.width || encoded.height != source.height)
            throw EncodeError("Encoded archive dimensions changed");
        const auto bytes = fs::file_size(partial);
        checkpoint(cancel, deadline);
        publish(partial, output);
        return {encoded.frames, bytes, encoded.width, encoded.height, encoded.fps};
    } catch (const EncodeError&) {
        if (launched) remove_owned_partial(partial);
        throw;
    } catch (...) {
        if (launched) remove_owned_partial(partial);
        throw EncodeError("Archive clip encoding or verification failed");
    }
}
EncodedArchiveClip validate_archive_clip(const fs::path& input_avi, const fs::path& output_mp4,
                                        const ArchiveEncodeConfig& config, const std::atomic_bool& cancel) {
    try {
        validate_config(config);
        const auto deadline = Clock::now() + config.timeout;
        checkpoint(cancel, deadline);
        const auto [input, output] = validate_paths(input_avi, output_mp4);
        if (!present(output) || !fs::is_regular_file(fs::symlink_status(output)) || fs::file_size(output) == 0)
            throw EncodeError("Published archive MP4 is unavailable");
        output_bound(output, config.max_output_bytes);
        const auto source = verify_video(input, config, cancel, deadline, true);
        const auto encoded = verify_video(output, config, cancel, deadline, false);
        if (encoded.width != source.width || encoded.height != source.height)
            throw EncodeError("Encoded archive dimensions changed");
        checkpoint(cancel, deadline);
        return {encoded.frames, fs::file_size(output), encoded.width, encoded.height, encoded.fps};
    } catch (const EncodeError&) { throw; }
    catch (...) { throw EncodeError("Published archive clip verification failed"); }
}
}  // namespace aegisvision::vision
