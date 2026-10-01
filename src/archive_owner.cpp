#include "aegisvision/archive_owner.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace aegisvision::vision {
namespace {
namespace fs = std::filesystem;
constexpr const char* unavailable = "Archive root ownership unavailable";
bool same_path(const fs::path& left, const fs::path& right) {
#ifdef _WIN32
    auto a = left.generic_u8string(), b = right.generic_u8string();
    const auto fold = [](char8_t c) {
        return c >= u8'A' && c <= u8'Z' ? static_cast<char8_t>(c + (u8'a' - u8'A')) : c;
    };
    std::transform(a.begin(), a.end(), a.begin(), fold);
    std::transform(b.begin(), b.end(), b.begin(), fold);
    return a == b;
#else
    return left == right;
#endif
}
void no_aliases(const fs::path& path) {
    auto current = path.root_path();
    for (const auto& component : path.relative_path()) {
        current /= component;
        const auto status = fs::symlink_status(current);
        if (status.type() == fs::file_type::not_found) continue;
        if (fs::is_symlink(status) || !same_path(fs::weakly_canonical(current), current))
            throw std::runtime_error(unavailable);
    }
}
void validate_lock_path(const fs::path& path) {
    no_aliases(path);
    const auto status = fs::symlink_status(path);
    if (status.type() != fs::file_type::not_found &&
        (!fs::is_regular_file(status) || fs::file_size(path) != 0 || fs::hard_link_count(path) != 1))
        throw std::runtime_error(unavailable);
}
#ifdef _WIN32
fs::path handle_path(HANDLE handle) {
    std::wstring name(32768, L'\0');
    const auto length = GetFinalPathNameByHandleW(handle, name.data(), static_cast<DWORD>(name.size()),
        FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (!length || length >= name.size()) throw std::runtime_error(unavailable);
    name.resize(length);
    if (name.starts_with(L"\\\\?\\UNC\\")) name = L"\\\\" + name.substr(8);
    else if (name.starts_with(L"\\\\?\\")) name.erase(0, 4);
    return fs::path(name).lexically_normal();
}
#endif
}
struct ArchiveOwner::Impl {
#ifdef _WIN32
    HANDLE handle{INVALID_HANDLE_VALUE};
    ~Impl() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
    int handle{-1};
    ~Impl() { if (handle >= 0) ::close(handle); }
#endif
};
ArchiveOwner::ArchiveOwner(const fs::path& value) : impl_(std::make_unique<Impl>()) {
    try {
        if (value.empty()) throw std::runtime_error(unavailable);
        const auto root = fs::absolute(value).lexically_normal();
        if (root == root.root_path()) throw std::runtime_error(unavailable);
        no_aliases(root);
        if (fs::exists(root) && !fs::is_directory(root)) throw std::runtime_error(unavailable);
        fs::create_directories(root);
        no_aliases(root);
        if (!fs::is_directory(root) || !same_path(fs::canonical(root), root)) throw std::runtime_error(unavailable);
        const auto lock = root / ".owner.lock";
        validate_lock_path(lock);
#ifdef _WIN32
        // shareMode=0 rejects a second owner even in the same process. No
        // TRUNCATE/DELETE disposition; OPEN_REPARSE_POINT never follows the leaf.
        // https://learn.microsoft.com/en-us/windows/desktop/FileIO/creating-and-opening-files
        impl_->handle = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        if (impl_->handle == INVALID_HANDLE_VALUE) throw std::runtime_error(unavailable);
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(impl_->handle, &info) ||
            GetFileType(impl_->handle) != FILE_TYPE_DISK ||
            (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) ||
            info.nFileSizeHigh || info.nFileSizeLow || info.nNumberOfLinks != 1 ||
            !same_path(handle_path(impl_->handle), lock)) throw std::runtime_error(unavailable);
#else
        impl_->handle = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600);
        if (impl_->handle < 0) throw std::runtime_error(unavailable);
        struct stat info{};
        if (::fstat(impl_->handle, &info) || !S_ISREG(info.st_mode) || info.st_size != 0 || info.st_nlink != 1)
            throw std::runtime_error(unavailable);
        // flock locks an open-file description, unlike process-associated record
        // locks: independently opening the same inode twice cannot own it twice.
        // https://man7.org/linux/man-pages/man2/flock.2.html
        if (::flock(impl_->handle, LOCK_EX | LOCK_NB)) throw std::runtime_error(unavailable);
        struct stat named{};
        if (::lstat(lock.c_str(), &named) || !S_ISREG(named.st_mode) ||
            named.st_dev != info.st_dev || named.st_ino != info.st_ino) throw std::runtime_error(unavailable);
#endif
        // Recheck ancestors after opening: no aliases/path escapes accepted even
        // if the leaf looked safe before the native open operation.
        no_aliases(root);
        validate_lock_path(lock);
    } catch (...) {
        // Preserve ownership errors as stable, secret/path-free startup messages.
        throw std::runtime_error(unavailable);
    }
}
ArchiveOwner::~ArchiveOwner() = default;
} // namespace aegisvision::vision
