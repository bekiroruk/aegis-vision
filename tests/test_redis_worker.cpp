#include "aegisvision/redis_worker.hpp"
#include "aegisvision/redis_queue.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace aegisvision;
using Json = nlohmann::json;
using namespace std::chrono_literals;
void check(bool ok) { if (!ok) throw std::runtime_error("Worker assertion failed"); }

void real_redis(int port) {
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    RedisLeaseQueue queue(port, "worker-" + suffix, "worker-integration-v1");
    const QueueCall call = [&](const Json& c) { return queue.call(c); };
    call({{"op", "submit"}, {"id", "long"}, {"payload", Json::object()}});
    check(run_redis_job(call, [&](const Json&, const StopCheck& cancelled) {
        // A competing worker cannot claim after the original deadline because
        // the real heartbeat has extended the real Redis lease.
        std::this_thread::sleep_for(3300ms);
        check(!cancelled());
        check(call({{"op", "claim"}, {"owner", "rival"}, {"lease_ms", 3000}}).is_null());
        return Json{{"real_heartbeat", true}};
    }, "w", 3000) == WorkerOutcome::Completed);
    check(call({{"op", "get"}, {"id", "long"}}).at("result").at("real_heartbeat") == true);
    call({{"op", "submit"}, {"id", "cancel"}, {"payload", Json::object()}});
    check(run_redis_job(call, [&](const Json&, const StopCheck& cancelled) {
        call({{"op", "cancel"}, {"id", "cancel"}});
        const auto limit = std::chrono::steady_clock::now() + 3s;
        while (!cancelled() && std::chrono::steady_clock::now() < limit) std::this_thread::sleep_for(10ms);
        check(cancelled()); return Json{{"must_not_publish", true}};
    }, "w", 3000) == WorkerOutcome::Lost);
    const auto cancelled = call({{"op", "get"}, {"id", "cancel"}});
    check(cancelled.at("state") == "cancelled" && cancelled.at("result").is_null());
    std::cout << "Real Redis: automatic heartbeat beyond original lease, rival exclusion, active cancellation passed\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc == 2) { real_redis(std::stoi(argv[1])); return 0; }
        const Json job{{"id", "j"}, {"token", "42"}, {"payload", {{"image", "x.jpg"}}}};
        std::atomic<int> renews{0}, publishes{0}, calls{0};
        std::atomic<bool> stop{false};
        std::string terminal;
        QueueCall queue = [&](const Json& c) -> Json {
            ++calls;
            const auto op = c.at("op").get<std::string>();
            if (op == "claim") return job;
            check(c.at("id") == "j" && c.at("owner") == "w" && c.at("token") == "42");
            if (op == "renew") { ++renews; return job; }
            ++publishes; terminal = op; return job;
        };
        const JobHandler fast = [](const Json& p, const StopCheck&) {
            check(p.at("image") == "x.jpg"); return Json{{"detections", Json::array()}};
        };
        check(run_redis_job(queue, fast, "w") == WorkerOutcome::Completed);
        check(terminal == "complete" && publishes == 1);
        check(run_redis_job([](const Json&) { return Json(); }, fast, "w") == WorkerOutcome::Empty);
        auto broken = [](const Json&, const StopCheck&) -> Json { throw std::runtime_error("bad image"); };
        check(run_redis_job(queue, broken, "w") == WorkerOutcome::Failed && terminal == "fail");
        check(run_redis_job(queue, [](const Json&, const StopCheck&) { return Json::array(); }, "w") == WorkerOutcome::Failed);
        check(run_redis_job(queue, [](const Json&, const StopCheck&) { return Json{{"huge", std::string(65537, 'x')}}; }, "w") == WorkerOutcome::Failed);
        const auto before = calls.load();
        check(run_redis_job(queue, fast, "w", 3000, [] { return true; }) == WorkerOutcome::Empty);
        check(calls == before);
        // Inference outlives the original lease, renewed by a separate thread.
        check(run_redis_job(queue, [](const Json&, const StopCheck&) {
            std::this_thread::sleep_for(3300ms); return Json::object();
        }, "w", 3000) == WorkerOutcome::Completed);
        check(renews >= 3);
        for (const bool transport_error : {false, true}) {
            const auto published_before = publishes.load();
            QueueCall lost = [&](const Json& c) -> Json {
                if (c.at("op") == "renew") {
                    if (transport_error) throw std::runtime_error("connection lost");
                    return nullptr;
                }
                return queue(c);
            };
            check(run_redis_job(lost, [](const Json&, const StopCheck& cancelled) {
                const auto limit = std::chrono::steady_clock::now() + 3s;
                while (!cancelled() && std::chrono::steady_clock::now() < limit)
                    std::this_thread::sleep_for(10ms);
                check(cancelled()); return Json::object();
            }, "w", 3000) == WorkerOutcome::Lost);
            check(publishes == published_before);
        }
        const auto published_before = publishes.load();
        check(run_redis_job(queue, [&](const Json&, const StopCheck&) {
            stop = true; return Json::object();
        }, "w", 3000, [&] { return stop.load(); }) == WorkerOutcome::Lost);
        check(publishes == published_before);
        // Redis remains the final arbiter even if no heartbeat has fired yet.
        check(run_redis_job([&](const Json& c) -> Json {
            return c.at("op") == "claim" ? job : Json();
        }, fast, "w") == WorkerOutcome::Lost);
        int uncertain_calls = 0;
        try {
            run_redis_job([&](const Json&) -> Json { ++uncertain_calls; throw std::runtime_error("uncertain"); }, fast, "w");
            check(false);
        } catch (const std::runtime_error&) { check(uncertain_calls == 1); }
        std::cout << "Worker: complete, empty, failure, bounds, heartbeat, cancellation, lost lease, shutdown, no retry passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
