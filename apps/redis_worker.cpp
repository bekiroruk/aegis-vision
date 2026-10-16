#include "aegisvision/redis_image.hpp"
#include "aegisvision/redis_queue.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

namespace {
namespace fs = std::filesystem;
using namespace aegisvision;
using Json = nlohmann::json;
std::atomic<bool> stopping{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void stop(int) { stopping.store(true); }
std::string utf8(const fs::path& p) { const auto s = p.u8string(); return {s.begin(), s.end()}; }
int number(const fs::path& p) {
    const auto s = utf8(p);
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected a positive integer");
    return std::stoi(s);
}
void help() {
    std::cout << "C++ Redis YOLO image worker (OpenCV CPU, trusted local media)\n"
        "  aegisvision_redis_worker context MODEL.onnx MEDIA_ROOT\n"
        "  aegisvision_redis_worker request MODEL.onnx MEDIA_ROOT RELATIVE_IMAGE\n"
        "  aegisvision_redis_worker analyze MODEL.onnx MEDIA_ROOT RELATIVE_IMAGE\n"
        "  aegisvision_redis_worker run PORT QUEUE OWNER MODEL.onnx MEDIA_ROOT [--once [LEASE_MS]]\n"
        "request prints {context,payload} for the queue CLI. analyze is a Redis-free diagnostic.\n"
        "run defaults to polling with 30000ms lease; Ctrl+C/SIGTERM stops claims and suppresses active results.\n"
        "--once handles at most one job, exiting on empty. Exit 0=success/empty, 1=error/failed, 3=ownership lost.\n"
        "Results contain detections only; no output files or Qdrant side effects. HTTP/SQLite is unchanged.\n";
}
int run(const std::vector<fs::path>& args) {
    try {
        if (args.size() == 2 && args[1] == "--help") { help(); return 0; }
        if (args.size() < 2) { help(); return 2; }
        const auto mode = utf8(args[1]);
        if (mode == "context" || mode == "request" || mode == "analyze") {
            if (args.size() != (mode == "context" ? 4U : 5U)) { help(); return 2; }
            RedisImageWorker image(args[2], args[3]);
            if (mode == "context") std::cout << image.context() << '\n';
            else {
                const auto payload = image.request(utf8(args[4]));
                const Json output = mode == "request" ? Json{{"context", image.context()}, {"payload", payload}}
                    : image.analyze(payload, [] { return false; });
                std::cout << output.dump() << '\n';
            }
            return 0;
        }
        if (mode != "run" || args.size() < 7 || args.size() > 9
            || (args.size() >= 8 && args[7] != "--once")) { help(); return 2; }
        const int lease = args.size() == 9 ? number(args[8]) : 30000;
        const bool once = args.size() >= 8;
        RedisImageWorker image(args[5], args[6]);
        RedisLeaseQueue queue(number(args[2]), utf8(args[3]), image.context());
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
        std::cerr << "Redis image worker ready; context=" << image.context() << '\n';
        const JobHandler analyze = [&](const Json& payload, const StopCheck& cancelled) {
            try { return image.analyze(payload, cancelled); }
            catch (const std::exception& e) { std::cerr << "Analysis: " << e.what() << '\n'; throw; }
        };
        while (!stopping.load()) {
            const auto outcome = run_redis_job([&](const Json& c) { return queue.call(c); },
                analyze, utf8(args[4]), lease, [] { return stopping.load(); });
            const auto state = outcome == WorkerOutcome::Completed ? "completed" :
                outcome == WorkerOutcome::Failed ? "failed" : outcome == WorkerOutcome::Lost ? "ownership_lost" : "empty";
            if (outcome != WorkerOutcome::Empty || once) std::cout << Json{{"outcome", state}}.dump() << std::endl;
            if (outcome == WorkerOutcome::Lost) return stopping.load() ? 0 : 3;
            if (once) return outcome == WorkerOutcome::Failed ? 1 : 0;
            if (outcome == WorkerOutcome::Empty)
                for (int i = 0; i < 10 && !stopping.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    return run(std::vector<std::filesystem::path>(argv, argv + argc));
}
