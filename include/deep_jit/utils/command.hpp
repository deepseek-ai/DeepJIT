#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

inline std::string quote_command_path(const std::filesystem::path& path) {
#if defined(_WIN32)
    auto value = path.string();
    DJ_HOST_ASSERT(value.find_first_of("\"%!\r\n") == std::string::npos,
                   "unsupported shell metacharacter in command path: {}", value);
    // Command-line argument parsing treats backslashes before a quote specially.
    const auto last = value.find_last_not_of('\\');
    const auto trailing = last == std::string::npos ? value.size() : value.size() - last - 1;
    value.append(trailing, '\\');
    return "\"" + value + "\"";
#else
    return path.string();
#endif
}

inline std::string call_external_command(std::string command, const bool print_command = false) {
    DJ_HOST_ASSERT(not command.empty(), "command must not be empty");
    if (print_command)
        std::printf("Running command: %s\n", command.c_str());

    command += " 2>&1";
#if defined(_WIN32)
    // cmd.exe /c strips the outer quotes, preserving a quoted executable path.
    command = "\"" + command + "\"";
#endif
    const auto close_pipe = [](FILE* file) {
#if defined(_WIN32)
        return _pclose(file);
#else
        return pclose(file);
#endif
    };
#if defined(_WIN32)
    std::unique_ptr<FILE, decltype(close_pipe)> pipe(_popen(command.c_str(), "r"), close_pipe);
#else
    std::unique_ptr<FILE, decltype(close_pipe)> pipe(popen(command.c_str(), "r"), close_pipe);
#endif
    DJ_HOST_ASSERT(pipe != nullptr, "failed to run command: {}", command);

    std::array<char, 512> buffer;
    std::string output;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        output += buffer.data();

    const bool read_failed = std::ferror(pipe.get()) != 0;
    const auto status = close_pipe(pipe.release());
    DJ_HOST_ASSERT(not read_failed, "failed to read command output: {}", command);
    DJ_HOST_ASSERT(status != -1, "failed to wait for command: {}", command);
#if defined(_WIN32)
    const auto exit_code = status;
#else
    // NOTES: if the child was killed by a signal (e.g., SIGINT from Ctrl+C),
    // WEXITSTATUS would incorrectly return 0. Treat signal death as failure.
    const auto exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    DJ_HOST_ASSERT(exit_code == 0, "command failed with exit code {}:\n{}\n{}", exit_code, command, output);
    return output;
}

}  // namespace deep_jit
