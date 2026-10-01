#include "aegisvision/jobs.hpp"
#include "aegisvision/indexing.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace {
using namespace aegisvision;
using Json = nlohmann::json;
namespace fs = std::filesystem;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void until(F predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Persistence test timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
Json fixture(const std::string& id, const char* state, bool cancelled = false, int recoveries = 0) {
    return {{"id", id}, {"state", state}, {"request", {{"type", "fixture"}}}, {"progress", {{"items", 2}}},
        {"result", nullptr}, {"error", nullptr}, {"created_ms", 1}, {"updated_ms", 1},
        {"cancel_requested", cancelled}, {"attempts", 1}, {"recoveries", recoveries}};
}
Json handler(const Json&, const JobQueue::Progress&, const std::atomic_bool&) { return {{"ok", true}}; }
void checks(const fs::path& root) {
    const JobPersistence config{root / "işler.sqlite", "fixture-context"};
    std::string running, waiting, cancelled, exhausted, completed;
    {
        JobStore store(config);
        bool owned = false;
        try { JobStore second(config); } catch (const JobStorageError&) { owned = true; }
        require(owned, "Two owners accepted for the same database");
        running = store.allocate_id(); waiting = store.allocate_id(); cancelled = store.allocate_id();
        exhausted = store.allocate_id(); completed = store.allocate_id();
        store.save(fixture(running, "running")); store.save(fixture(waiting, "queued"));
        store.save(fixture(cancelled, "running", true)); store.save(fixture(exhausted, "running", false, 3));
        auto done = fixture(completed, "succeeded"); done["result"] = {{"original", true}}; store.save(done);
    }
    {
        JobQueue queue(handler, 4, 8, config);
        until([&] { return queue.get(running).at("state") == "succeeded" && queue.get(waiting).at("state") == "succeeded"; });
        require(queue.get(running).at("recoveries") == 1 && queue.get(running).at("attempts") == 2, "Interrupted work did not replay");
        require(queue.get(running).at("progress").at("items") == 2, "Replay discarded last checkpoint");
        require(queue.get(cancelled).at("state") == "cancelled" && queue.get(cancelled).at("attempts") == 1, "Cancellation replayed");
        require(queue.get(exhausted).at("state") == "failed" && queue.get(exhausted).at("attempts") == 1, "Retry budget not enforced");
        require(queue.get(completed).at("result").at("original") == true, "Completed job changed on restart");
    }
    {
        JobQueue queue(handler, 4, 8, config);
        require(queue.get(running).at("attempts") == 2, "Completed work replayed twice");
        const auto fresh = queue.submit({{"type", "new"}});
        require(fresh != running && fresh != waiting && fresh != cancelled && fresh != exhausted && fresh != completed, "Job ID reused on restart");
        until([&] { return queue.get(fresh).at("state") == "succeeded"; });
        for (int i = 0; i < 6; ++i) {
            const auto next = queue.submit({{"type", "eviction"}});
            until([&] { return queue.get(next).at("state") == "succeeded"; });
        }
        require(queue.list().size() == 8, "Durable history exceeded limit");
    }
    { JobStore store(config); require(store.load().size() == 8, "History eviction not persisted"); }
    {
        JobStore store(config); const auto before = store.load();
        try { store.save(Json::object(), {before[0].at("id").get<std::string>()}); }
        catch (const Json::exception&) { }
        require(store.load() == before, "Failed transaction evicted accepted history");
    }
    bool mismatch = false;
    try { JobStore changed({config.database, "other-model"}); } catch (const JobStorageError&) { mismatch = true; }
    require(mismatch, "Changed model/store context accepted");
    { JobStore store(config); require(store.load().size() == 8, "Mismatch damaged history"); }

    const JobPersistence shutdown{root / "shutdown.sqlite", "shutdown"};
    std::string active, queued;
    {
        JobQueue queue([](const Json&, const JobQueue::Progress& progress, const std::atomic_bool& cancel) -> Json {
            progress({{"items", 3}}); until([&] { return cancel.load(); }); throw IndexCancelled();
        }, 1, 4, shutdown);
        active = queue.submit({{"type", "active"}});
        until([&] { return queue.get(active).at("progress").contains("items"); });
        queued = queue.submit({{"type", "waiting"}});
    }
    {
        JobStore store(shutdown); const auto jobs = store.load();
        require(jobs.size() == 2 && jobs[0].at("state") == "queued" && jobs[0].at("progress").at("items") == 3,
            "Graceful stop lost active work or checkpoint");
    }
    {
        JobQueue queue(handler, 1, 4, shutdown);
        until([&] { return queue.get(active).at("state") == "succeeded" && queue.get(queued).at("state") == "succeeded"; });
    }
    const JobPersistence invalid{root / "invalid.sqlite", "invalid"};
    { JobStore store(invalid); store.save({{"id", "1-1"}, {"state", "unknown"}}); }
    bool corrupt = false;
    try { JobQueue queue(handler, 1, 4, invalid); } catch (const JobStorageError&) { corrupt = true; }
    require(corrupt, "Invalid persisted document executed");
    const JobPersistence quota{root / "quota.sqlite", "quota", 8};
    std::string unfinished;
    {
        JobQueue queue([](const Json&, const JobQueue::Progress&, const std::atomic_bool&) -> Json {
            return {{"oversized", std::string(200000, 'x')}};
        }, 1, 4, quota);
        unfinished = queue.submit({{"type", "large-result"}});
        until([&] { return !queue.storage_error().empty(); });
        require(queue.get(unfinished).at("state") == "running", "Uncommitted result reported as successful");
        bool frozen = false;
        try { (void)queue.submit({{"type", "more"}}); } catch (const JobStorageError&) { frozen = true; }
        require(frozen, "Storage error did not freeze queue");
    }
    {
        JobQueue queue(handler, 1, 4, {quota.database, quota.context});
        until([&] { return queue.get(unfinished).at("state") == "succeeded"; });
        require(queue.get(unfinished).at("recoveries") == 1, "Storage failure lost recoverable accepted work");
    }
}
void crash(const fs::path& database) {
    JobQueue queue([](const Json&, const JobQueue::Progress& progress, const std::atomic_bool&) -> Json {
        progress({{"items", 7}});
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        progress({{"items", 8}});
        while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
    }, 1, 8, {database, "crash-fixture"});
    const auto active = queue.submit({{"type", "crash"}});
    until([&] { return queue.get(active).at("progress").value("items", 0) == 8; });
    (void)queue.submit({{"type", "waiting"}});
    std::cout << "Abrupt exit after durable running checkpoint and queued acceptance" << std::endl;
    std::_Exit(0); // Deliberately skips all destructors/SQLite close.
}
void recover(const fs::path& database) {
    JobQueue queue(handler, 1, 8, {database, "crash-fixture"});
    until([&] {
        const auto jobs = queue.list();
        return jobs.size() >= 2 && jobs[0].at("state") == "succeeded" && jobs[1].at("state") == "succeeded";
    });
    const auto jobs = queue.list();
    require(jobs[1].at("recoveries") == 1 && jobs[1].at("progress").at("items") == 8, "Abrupt-exit recovery failed");
    std::cout << "WAL recovery retained checkpoint, replayed active and queued work" << std::endl;
}
}
int main(int argc, char* argv[]) {
    const auto root = fs::temp_directory_path() / ("aegis-persistence-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        if (argc == 3 && std::string(argv[1]) == "--crash") { crash(fs::u8path(argv[2])); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--recover") { recover(fs::u8path(argv[2])); return 0; }
        checks(root); fs::remove_all(root); // This test's uniquely created fixtures only.
        std::cout << "Durability, ownership, replay, cancellation, retry limits, IDs and retention passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << " fixtures=" << root << '\n'; return 1; }
}
