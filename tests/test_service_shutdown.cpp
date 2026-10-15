#include "aegisvision/service.hpp"
#include <csignal>
#include <iostream>

namespace {
using namespace aegisvision;
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void until(F predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Shutdown fixture timeout");
        std::this_thread::sleep_for(5ms);
    }
}
class Detector : public IDetector {
public: std::vector<Detection> detect(const Frame&) override { return {}; }
};
class Embedder : public IEmbedder {
public:
    std::atomic_bool started{false}, release{false};
    std::vector<float> embed_image(const Frame&, const Detection&) override { return {1, 0}; }
    std::vector<float> embed_text(std::string_view text) override {
        if (text == "active") {
            started = true;
            // Simulates an in-flight inference call that cannot be preempted.
            until([&] { return release.load(); });
        }
        return {1, 0};
    }
};
class Store : public IVectorStore {
public:
    void upsert(std::string, std::vector<float>, std::map<std::string,std::string>) override {}
    std::vector<SearchResult> search(const std::vector<float>&, std::size_t) const override { return {}; }
};
Json response(const httplib::Result& result, int code = 200) {
    require(result && result->status == code, "Unexpected shutdown fixture HTTP response");
    return Json::parse(result->body);
}
std::string submit(httplib::Client& client, const char* query) {
    return response(client.Post("/api/jobs", Json{{"type","search"},{"query",query}}.dump(),
                                "application/json"), 202).at("id").get<std::string>();
}
std::atomic_int restored_calls{0};
extern "C" void previous_handler(int) { restored_calls.fetch_add(1); }
void check(const fs::path& root, bool external, int signal) {
    fs::create_directories(root / "media");
    Detector detector; Embedder embedder; Store store;
    ServiceConfig config{root / "media", fs::path(__FILE__).parent_path().parent_path() / "web", "shutdown-fixture"};
    config.max_pending = 3; config.max_retained = 8;
    config.persistence = {root / "jobs.sqlite", "shutdown-test-v1"};
    std::string active, waiting, cancelled;
    const auto original = std::signal(signal, previous_handler);
    {
        LocalService service(detector, embedder, store, config);
        const int port = service.bind(0);
        std::exception_ptr error;
        std::jthread control([&] {
            try {
                httplib::Client client("127.0.0.1", port);
                client.set_connection_timeout(1); client.set_read_timeout(2);
                until([&] { const auto r=client.Get("/api/health"); return r && r->status==200; });
                active = submit(client, "active");
                until([&] { return embedder.started.load(); });
                waiting = submit(client, "waiting");
                cancelled = submit(client, "cancelled");
                response(client.Post("/api/jobs/" + cancelled + "/cancel", "{}", "application/json"));
                std::cout << "READY " << port << std::endl;
                if (!external) std::raise(signal);
                // Wait for signal-driven HTTP shutdown, keeping inference active
                // until cancellation has been requested by LocalService::stop().
                until([&] { return !client.Get("/api/health"); });
                embedder.release = true;
            } catch (...) {
                error = std::current_exception(); embedder.release = true; service.stop();
            }
        });
        require(service.listen_until_signal(), "Signal-driven HTTP listen failed");
        control.join();
        if (error) std::rethrow_exception(error);
    } // Worker joins and SQLite closes before reopening.
    std::raise(signal);
    require(restored_calls == 1, "Previous signal handler not restored");
    std::signal(signal, original);
    // Same service context: active and pending jobs replay, user cancellation does not.
    {
        LocalService service(detector, embedder, store, config);
        const int port = service.bind(0);
        std::thread listener([&] { service.listen(); });
        struct Guard { LocalService& s; std::thread& t; ~Guard() { s.stop(); t.join(); } } guard{service,listener};
        httplib::Client client("127.0.0.1", port);
        until([&] { const auto r=client.Get("/api/health"); return r && r->status==200; });
        Json a, b, c;
        until([&] {
            a=response(client.Get("/api/jobs/"+active)); b=response(client.Get("/api/jobs/"+waiting));
            c=response(client.Get("/api/jobs/"+cancelled));
            return a.at("state")=="succeeded" && b.at("state")=="succeeded";
        });
        require(a.at("attempts")==2 && a.at("recoveries")==0, "Active job not gracefully requeued under same ID");
        require(b.at("attempts")==1, "Pending job ran before shutdown or replayed twice");
        require(c.at("state")=="cancelled" && c.at("attempts")==0, "User cancellation was lost");
    }
    std::cout << "Signal shutdown, handler restoration, same-ID replay and retained cancellation passed\n";
}
}
int main(int argc, char* argv[]) {
    const auto root = fs::temp_directory_path() / ("aegis-shutdown-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const bool external = argc == 2 && std::string(argv[1]) == "--external";
        const int signal = argc == 2 && std::string(argv[1]) == "--sigint" ? SIGINT : SIGTERM;
        check(root, external, signal);
        fs::remove_all(root); // Only this test's uniquely created fixture directory.
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << " fixtures=" << root << '\n'; return 1;
    }
}
