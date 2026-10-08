#pragma once

#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace aegisvision::vision::detail {
// A metadata/MP4 publisher may rename a staging file after directory enumeration.
// Restart the ENTIRE scan, never accept a partially counted quota snapshot.
// Only disappearance is transient; aliases, permissions and quota errors fail closed.
template<class Scan>
auto stable_archive_scan(Scan scan) -> decltype(scan()) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        try { return scan(); }
        catch (const std::filesystem::filesystem_error& error) {
            if (error.code() != std::errc::no_such_file_or_directory) throw;
            if (attempt == 2) throw std::runtime_error("archive_scan_unstable");
        }
    }
    throw std::runtime_error("archive_scan_unstable");
}
} // namespace aegisvision::vision::detail
