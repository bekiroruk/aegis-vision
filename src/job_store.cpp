#include "aegisvision/job_store.hpp"
#include <sqlite3.h>
#include <chrono>

namespace aegisvision {
namespace {
using Json = nlohmann::json;
void check(sqlite3* db, int code) {
    if (code != SQLITE_OK && code != SQLITE_DONE && code != SQLITE_ROW)
        throw JobStorageError("Job database: " + std::string(sqlite3_errmsg(db)));
}
void execute(sqlite3* db, const char* sql) { check(db, sqlite3_exec(db, sql, nullptr, nullptr, nullptr)); }
class Statement {
public:
    Statement(sqlite3* db, const char* sql) : db_(db) { check(db, sqlite3_prepare_v2(db, sql, -1, &value_, nullptr)); }
    ~Statement() { sqlite3_finalize(value_); }
    void bind(int index, const std::string& value) {
        check(db_, sqlite3_bind_text(value_, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT));
    }
    int step() { const int result = sqlite3_step(value_); check(db_, result); return result; }
    std::string text(int column) const {
        const auto* value = sqlite3_column_text(value_, column);
        if (!value) throw JobStorageError("Null field in job database");
        return {reinterpret_cast<const char*>(value), static_cast<std::size_t>(sqlite3_column_bytes(value_, column))};
    }
private:
    sqlite3* db_{};
    sqlite3_stmt* value_{};
};
void metadata(sqlite3* db, const char* key, const std::string& value) {
    Statement insert(db, "INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)");
    insert.bind(1, key); insert.bind(2, value); insert.step();
    Statement select(db, "SELECT value FROM metadata WHERE key=?"); select.bind(1, key);
    if (select.step() != SQLITE_ROW || select.text(0) != value)
        throw JobStorageError(std::string("Job database configuration mismatch: ") + key + "; use a different database path");
}
}
struct JobStore::Impl {
    sqlite3* db{};
    ~Impl() { if (db) sqlite3_close_v2(db); }
};
JobStore::JobStore(const JobPersistence& config) : impl_(std::make_unique<Impl>()) {
    if (config.database.empty() || config.context.empty() || config.max_pages < 8 || config.max_pages > 65536)
        throw JobStorageError("Valid job database path/context/page limit required");
    if (!config.database.parent_path().empty()) std::filesystem::create_directories(config.database.parent_path());
    const auto raw = std::filesystem::absolute(config.database).u8string();
    const std::string path(raw.begin(), raw.end());
    const auto opened = sqlite3_open_v2(path.c_str(), &impl_->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    check(impl_->db, opened);
    auto* db = impl_->db;
    // Held for this connection's lifetime: two services cannot replay the same jobs.
    execute(db, "PRAGMA locking_mode=EXCLUSIVE");
    execute(db, "PRAGMA page_size=4096");
    {
        Statement mode(db, "PRAGMA journal_mode=WAL");
        if (mode.step() != SQLITE_ROW || mode.text(0) != "wal") throw JobStorageError("WAL unavailable for job database");
    }
    execute(db, "PRAGMA synchronous=FULL");
    {
        const auto sql = "PRAGMA max_page_count=" + std::to_string(config.max_pages);
        Statement quota(db, sql.c_str());
        if (quota.step() != SQLITE_ROW || quota.text(0) != std::to_string(config.max_pages))
            throw JobStorageError("Existing job database exceeds page limit");
    }
    execute(db, "PRAGMA wal_autocheckpoint=256");
    execute(db, "PRAGMA journal_size_limit=4194304");
    execute(db, "BEGIN EXCLUSIVE");
    try {
        execute(db, "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)");
        execute(db, "CREATE TABLE IF NOT EXISTS jobs(position INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,document TEXT NOT NULL)");
        metadata(db, "version", "1");
        metadata(db, "context", config.context);
        const auto epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        Statement prefix(db, "INSERT OR IGNORE INTO metadata(key,value) VALUES ('prefix',?)");
        prefix.bind(1, std::to_string(epoch)); prefix.step();
        execute(db, "INSERT OR IGNORE INTO metadata(key,value) VALUES ('sequence','0')");
        execute(db, "COMMIT");
    } catch (...) { sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr); throw; }
}
JobStore::~JobStore() = default;
std::vector<Json> JobStore::load() const {
    Statement query(impl_->db, "SELECT document FROM jobs ORDER BY position");
    std::vector<Json> result;
    while (query.step() == SQLITE_ROW) {
        if (result.size() >= 100000) throw JobStorageError("Job database contains too many records");
        try { result.push_back(Json::parse(query.text(0))); }
        catch (const Json::exception& error) { throw JobStorageError("Invalid job document: " + std::string(error.what())); }
    }
    return result;
}
std::string JobStore::allocate_id() {
    Statement increment(impl_->db, "UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='sequence'");
    increment.step();
    Statement query(impl_->db, "SELECT (SELECT value FROM metadata WHERE key='prefix') || '-' || value FROM metadata WHERE key='sequence'");
    if (query.step() != SQLITE_ROW) throw JobStorageError("Missing job ID sequence");
    return query.text(0);
}
void JobStore::save(const Json& job, const std::vector<std::string>& evicted) {
    auto* db = impl_->db;
    execute(db, "BEGIN IMMEDIATE");
    try {
        for (const auto& id : evicted) {
            Statement remove(db, "DELETE FROM jobs WHERE id=?"); remove.bind(1, id); remove.step();
        }
        Statement upsert(db, "INSERT INTO jobs(id,document) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET document=excluded.document");
        upsert.bind(1, job.at("id").get<std::string>()); upsert.bind(2, job.dump()); upsert.step();
        execute(db, "COMMIT");
    } catch (...) { sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr); throw; }
}
}
