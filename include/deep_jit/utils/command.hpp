#pragma once

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <deep_jit/utils/exception.hpp>

extern char** environ;

namespace deep_jit {

inline std::string call_external_command(std::string command, bool print_command = false,
                                         std::chrono::milliseconds timeout = std::chrono::minutes(5)) {
    DJ_HOST_ASSERT(not command.empty(), "command must not be empty");
    DJ_HOST_ASSERT(timeout.count() >= 0, "command timeout must not be negative");
    if (print_command) std::printf("Running command: %s\n", command.c_str());
    int fd[2];
    DJ_HOST_ASSERT(pipe2(fd, O_CLOEXEC) == 0, "command pipe failed");
    pid_t pid = -1;
    try {
        DJ_HOST_ASSERT(fcntl(fd[0], F_SETFL, O_NONBLOCK) == 0, "nonblocking pipe failed");
        posix_spawn_file_actions_t actions;
        DJ_HOST_ASSERT(posix_spawn_file_actions_init(&actions) == 0);
        int error = posix_spawn_file_actions_adddup2(&actions, fd[1], STDOUT_FILENO);
        if (error == 0) error = posix_spawn_file_actions_adddup2(&actions, fd[1], STDERR_FILENO);
        if (error == 0) error = posix_spawn_file_actions_addclose(&actions, fd[0]);
        if (error == 0) error = posix_spawn_file_actions_addclose(&actions, fd[1]);
        char shell[] = "sh", option[] = "-c";
        char* args[] = {shell, option, command.data(), nullptr};
        pid_t child;
        if (error == 0) error = posix_spawn(&child, "/bin/sh", &actions, nullptr, args, environ);
        posix_spawn_file_actions_destroy(&actions);
        DJ_HOST_ASSERT(error == 0, "command spawn failed ({}): {}", error, command);
        pid = child;
        close(fd[1]);
        fd[1] = -1;

        const auto deadline = timeout.count() == 0 ? std::chrono::steady_clock::time_point::max()
                                                   : std::chrono::steady_clock::now() + timeout;
        std::array<char, 4096> buffer;
        std::string output;
        int status = 0;
        bool eof = false;
        while (not eof or pid > 0) {
            if (std::chrono::steady_clock::now() >= deadline)
                DJ_PANIC("command timed out after {} ms:\n{}\n{}", timeout.count(), command, output);
            if (not eof) {
                const auto count = ::read(fd[0], buffer.data(), buffer.size());
                if (count > 0) { output.append(buffer.data(), static_cast<size_t>(count)); continue; }
                if (count == 0) eof = true;
                else DJ_HOST_ASSERT(errno == EAGAIN or errno == EINTR, "command read failed");
            }
            if (pid > 0) {
                const auto reaped = waitpid(pid, &status, WNOHANG);
                DJ_HOST_ASSERT(reaped >= 0 or errno == EINTR, "command wait failed");
                if (reaped > 0) pid = -1;
            }
            if (eof and pid < 0) break;
            pollfd polling{eof ? -1 : fd[0], POLLIN, 0};
            DJ_HOST_ASSERT(poll(&polling, 1, 10) >= 0 or errno == EINTR, "command poll failed");
        }
        const auto exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        DJ_HOST_ASSERT(exit_code == 0, "command failed with exit code {}:\n{}\n{}", exit_code, command, output);
        close(fd[0]);
        return output;
    } catch (...) {
        for (int f : fd) if (f >= 0) close(f);
        if (pid > 0) kill(pid, SIGKILL);
        throw;
    }
}

}  // namespace deep_jit
