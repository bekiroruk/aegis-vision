#pragma once
#include <filesystem>
#include <memory>

namespace aegisvision::vision {
// Exclusive native ownership of a dedicated local archive root. The zero-byte
// .owner.lock file is retained forever; the OS releases ownership on close/crash.
// Acquire before restoring durable archive jobs or starting any archive worker.
class ArchiveOwner {
public:
    explicit ArchiveOwner(const std::filesystem::path& root);
    ~ArchiveOwner();
    ArchiveOwner(const ArchiveOwner&) = delete;
    ArchiveOwner& operator=(const ArchiveOwner&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace aegisvision::vision
