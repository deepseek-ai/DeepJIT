#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#if defined(_WIN32)
#include <deep_jit/utils/windows.hpp>
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
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

inline std::string call_external_command(std::string command, bool print_command = false,
                                         std::chrono::milliseconds timeout = std::chrono::minutes(5)) {
    DJ_HOST_ASSERT(not command.empty(), "command must not be empty");
    DJ_HOST_ASSERT(timeout.count() >= 0, "command timeout must not be negative");
    if (print_command) std::printf("Running command: %s\n", command.c_str());
#if defined(_WIN32)
    const auto close_handle = [](HANDLE handle) { ::CloseHandle(handle); };
    using Handle = std::unique_ptr<void, decltype(close_handle)>;
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE pipe_read, pipe_write;
    DJ_HOST_ASSERT(::CreatePipe(&pipe_read, &pipe_write, &security, 0),
                   "command pipe failed ({})", ::GetLastError());
    Handle input_pipe(pipe_read), output_pipe(pipe_write);
    DJ_HOST_ASSERT(::SetHandleInformation(input_pipe.get(), HANDLE_FLAG_INHERIT, 0),
                   "command pipe inheritance failed ({})", ::GetLastError());
    const auto input = ::CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     &security, OPEN_EXISTING, 0, nullptr);
    DJ_HOST_ASSERT(input != INVALID_HANDLE_VALUE, "command stdin failed ({})", ::GetLastError());
    Handle stdin_handle(input);

    // Restrict inheritance so concurrent commands cannot keep each other's pipes open.
    SIZE_T attribute_size = 0;
    ::InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_size);
    DJ_HOST_ASSERT(attribute_size != 0, "command attribute sizing failed ({})", ::GetLastError());
    const auto attribute_storage = std::make_unique<char[]>(attribute_size);
    auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.get());
    DJ_HOST_ASSERT(::InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_size),
                   "command attribute initialization failed ({})", ::GetLastError());
    const auto delete_attributes = [](LPPROC_THREAD_ATTRIBUTE_LIST list) {
        ::DeleteProcThreadAttributeList(list);
    };
    const std::unique_ptr<std::remove_pointer_t<LPPROC_THREAD_ATTRIBUTE_LIST>, decltype(delete_attributes)>
        attribute_guard(attributes);
    HANDLE inherited[] = {output_pipe.get(), stdin_handle.get()};
    DJ_HOST_ASSERT(::UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                               inherited, sizeof(inherited), nullptr, nullptr),
                   "command handle inheritance failed ({})", ::GetLastError());

    Handle job(::CreateJobObjectW(nullptr, nullptr));
    DJ_HOST_ASSERT(job != nullptr, "command job creation failed ({})", ::GetLastError());
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    DJ_HOST_ASSERT(::SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits)),
                   "command job configuration failed ({})", ::GetLastError());

    std::array<char, MAX_PATH> system_directory{};
    const auto length = ::GetSystemDirectoryA(system_directory.data(), static_cast<UINT>(system_directory.size()));
    DJ_HOST_ASSERT(length > 0 and length < system_directory.size(),
                   "command shell lookup failed ({})", ::GetLastError());
    const auto shell = std::string(system_directory.data(), length) + "\\cmd.exe";
    auto shell_command = "\"" + shell + "\" /D /S /C \"" + command + "\"";
    STARTUPINFOEXA startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdin_handle.get();
    startup.StartupInfo.hStdOutput = output_pipe.get();
    startup.StartupInfo.hStdError = output_pipe.get();
    startup.lpAttributeList = attributes;
    PROCESS_INFORMATION process_info{};
    DJ_HOST_ASSERT(::CreateProcessA(shell.c_str(), shell_command.data(), nullptr, nullptr, TRUE,
                                    CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                    &startup.StartupInfo, &process_info),
                   "command spawn failed ({}): {}", ::GetLastError(), command);
    Handle process(process_info.hProcess), thread(process_info.hThread);
    if (not ::AssignProcessToJobObject(job.get(), process.get())) {
        const auto error = ::GetLastError();
        ::TerminateProcess(process.get(), 1);
        DJ_PANIC("command job assignment failed ({}): {}", error, command);
    }
    // Assign the suspended shell before it can create any compiler descendants.
    DJ_HOST_ASSERT(::ResumeThread(thread.get()) != static_cast<DWORD>(-1),
                   "command resume failed ({})", ::GetLastError());
    thread.reset();
    output_pipe.reset();
    stdin_handle.reset();

    const auto deadline = timeout.count() == 0 ? std::chrono::steady_clock::time_point::max()
                                               : std::chrono::steady_clock::now() + timeout;
    std::array<char, 4096> buffer;
    std::string output;
    bool eof = false;
    for (;;) {
        if (std::chrono::steady_clock::now() >= deadline)
            DJ_PANIC("command timed out after {} ms:\n{}\n{}", timeout.count(), command, output);
        if (not eof) {
            DWORD available = 0;
            if (not ::PeekNamedPipe(input_pipe.get(), nullptr, 0, nullptr, &available, nullptr)) {
                const auto error = ::GetLastError();
                DJ_HOST_ASSERT(error == ERROR_BROKEN_PIPE, "command pipe read failed ({})", error);
                eof = true;
            } else if (available > 0) {
                DWORD count = 0;
                DJ_HOST_ASSERT(::ReadFile(input_pipe.get(), buffer.data(),
                                          std::min(available, static_cast<DWORD>(buffer.size())), &count, nullptr),
                               "command read failed ({})", ::GetLastError());
                output.append(buffer.data(), count);
                continue;
            }
        }
        const auto status = ::WaitForSingleObject(process.get(), 0);
        DJ_HOST_ASSERT(status == WAIT_OBJECT_0 or status == WAIT_TIMEOUT,
                       "command wait failed ({})", ::GetLastError());
        if (eof and status == WAIT_OBJECT_0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    DWORD exit_code = 0;
    DJ_HOST_ASSERT(::GetExitCodeProcess(process.get(), &exit_code),
                   "command exit status failed ({})", ::GetLastError());
    // Preserve the CRLF translation of the previous Windows text-mode pipe.
    size_t size = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] == '\r' and i + 1 < output.size() and output[i + 1] == '\n') continue;
        output[size++] = output[i];
    }
    output.resize(size);
    DJ_HOST_ASSERT(exit_code == 0, "command failed with exit code {}:\n{}\n{}", exit_code, command, output);
    return output;
#else
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
#endif
}

}  // namespace deep_jit
