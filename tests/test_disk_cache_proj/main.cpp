#include <deep_jit/cache/disk.hpp>

#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (not condition)
        throw std::runtime_error(message);
}

void expect_failure(deep_jit::DiskCacheEntry& entry, std::errc expected) {
    const auto temporary = entry.path;
    try {
        entry.commit();
    } catch (const fs::filesystem_error& error) {
        const bool nonempty_destination = expected == std::errc::directory_not_empty and
                                          error.code() == std::errc::file_exists;
        require(error.code() == expected or nonempty_destination, "original rename error was lost");
        require(error.path1() == temporary and error.path2() == entry.commit_path,
                "rename paths were lost");
        require(not entry.hit and not entry.committed and entry.path == temporary,
                "failed publication changed entry state");
        require(deep_jit::read(temporary / "payload") == "new", "temporary payload was lost");
        return;
    }
    throw std::runtime_error("commit returned success after failed publication");
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "expected scenario and cache root");
        const std::string scenario = argv[1];
        const fs::path root = argv[2];
        deep_jit::DiskCache cache({root});
        fs::path temporary;
        {
            auto entry = cache.entry("test", "digest");
            require(not entry.hit, "expected initial miss");
            temporary = entry.path;
            deep_jit::write_file_sync(temporary / "payload", "new");
            if (scenario == "incomplete") {
                deep_jit::make_dirs(entry.commit_path);
                deep_jit::write_file_sync(entry.commit_path / "partial", "old");
                expect_failure(entry, std::errc::directory_not_empty);
                expect_failure(entry, std::errc::directory_not_empty);
                require(deep_jit::read(entry.commit_path / "partial") == "old", "destination changed");
                require(not fs::exists(entry.commit_path / deep_jit::kCommitFileName), "destination marked committed");
            } else if (scenario == "permission" or scenario == "marker_permission") {
                const auto parent = entry.commit_path.parent_path();
                deep_jit::make_dirs(parent);
                const auto permissions = fs::status(parent).permissions();
                fs::permissions(parent, scenario == "permission" ? fs::perms::owner_read | fs::perms::owner_exec
                                                                  : fs::perms::owner_read);
                try {
                    expect_failure(entry, std::errc::permission_denied);
                } catch (...) {
                    fs::permissions(parent, permissions);
                    throw;
                }
                fs::permissions(parent, permissions);
                require(not fs::exists(entry.commit_path), "permission failure published an entry");
                require(entry.commit() == entry.commit_path, "retry failed after restoring permissions");
            } else if (scenario == "winner") {
                // Both entries miss before either publishes, then the loser observes the winner.
                auto winner = cache.entry("test", "digest");
                deep_jit::write_file_sync(winner.path / "payload", "winner");
                winner.commit();
                require(entry.commit() == winner.path, "loser did not reuse winner");
                require(deep_jit::read(entry.path / "payload") == "winner", "winner payload replaced");
                require(entry.hit and entry.committed, "loser state not committed");
                require(not fs::exists(temporary), "loser temporary leaked");
            } else {
                require(scenario == "normal", "unknown scenario");
                require(entry.commit() == entry.commit_path, "wrong published path");
                require(entry.commit() == entry.commit_path, "repeated commit failed");
                require(entry.hit and entry.committed, "successful state not committed");
                auto hit = cache.entry("test", "digest");
                require(hit.hit and deep_jit::read(hit.path / "payload") == "new", "published entry not reusable");
                require(hit.commit() == entry.path, "hit commit changed path");
            }
        }
        require(not fs::exists(temporary), "temporary leaked after destruction");
        std::cout << scenario << ": passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
