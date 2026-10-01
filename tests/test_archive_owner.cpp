#include "aegisvision/archive_owner.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;
using aegisvision::vision::ArchiveOwner;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
template<class F> void rejects(F action) {
    try { action(); }
    catch (const std::exception& error) {
        require(std::string(error.what()) == "Archive root ownership unavailable", "Ownership error exposes private paths");
        return;
    }
    throw std::runtime_error("Invalid or duplicate archive owner accepted");
}
void checks(const fs::path& scratch) {
    const auto root = scratch / "archive";
    {
        ArchiveOwner first(root);
        require(fs::exists(root / ".owner.lock") && fs::file_size(root / ".owner.lock") == 0,
            "Native ownership marker is absent/nonempty");
        rejects([&] { ArchiveOwner second(root); });
        ArchiveOwner separate(scratch / "separate");
        rejects([&] { ArchiveOwner same_normalized(root / "sub/.."); });
    }
    require(fs::exists(root / ".owner.lock") && fs::file_size(root / ".owner.lock") == 0,
        "Owner destructor deleted or wrote the stable marker");
    { ArchiveOwner recovered(root); rejects([&] { ArchiveOwner duplicate(root); }); }
    { ArchiveOwner reopened(root); }
    rejects([&] { ArchiveOwner empty(fs::path{}); });

    const auto occupied = scratch / "occupied"; fs::create_directory(occupied);
    { std::ofstream marker(occupied / ".owner.lock"); marker << "do not replace"; }
    rejects([&] { ArchiveOwner invalid(occupied); });
    require(fs::file_size(occupied / ".owner.lock") == 14, "Owner overwrote material nonempty marker");
    const auto directory = scratch / "directory-marker"; fs::create_directories(directory / ".owner.lock");
    rejects([&] { ArchiveOwner invalid(directory); });
    const auto file_root = scratch / "file-root"; { std::ofstream file(file_root); file << 'x'; }
    rejects([&] { ArchiveOwner invalid(file_root); });

    const auto outside = scratch / "outside"; fs::create_directory(outside);
    const auto aliases = scratch / "aliases"; fs::create_directory(aliases);
    std::error_code error;
    fs::create_directory_symlink(outside, aliases / "alias", error);
    if (!error) {
        rejects([&] { ArchiveOwner alias(aliases / "alias/new-archive"); });
        require(!fs::exists(outside / "new-archive"), "Alias root wrote outside the assigned subtree");
        const auto leaf_root = scratch / "leaf-alias"; fs::create_directory(leaf_root);
        const auto target = outside / "empty"; { std::ofstream file(target); }
        fs::create_symlink(target, leaf_root / ".owner.lock");
        rejects([&] { ArchiveOwner alias(leaf_root); });
    } else std::cout << "Directory symlink creation unavailable; alias test skipped on this host\n";

    const auto linked_root = scratch / "hardlink"; fs::create_directory(linked_root);
    const auto linked_target = outside / "hardlink-empty"; { std::ofstream file(linked_target); }
    error.clear(); fs::create_hard_link(linked_target, linked_root / ".owner.lock", error);
    if (!error) rejects([&] { ArchiveOwner alias(linked_root); });
}
}
int main() {
    const auto scratch = fs::temp_directory_path() / ("aegis-archive-owner-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directory(scratch); checks(scratch);
        fs::remove_all(scratch); // This test's unique owned scratch directory only.
        std::cout << "Native archive ownership/exclusivity/release/path tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\nOwned diagnostic artifacts retained at " << scratch << '\n';
        return 1;
    }
}
