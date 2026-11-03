#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <cassert>
#include <csignal>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace deep_jit;
using namespace std::chrono_literals;
static volatile std::sig_atomic_t interrupted = 0;

static void expect_failure(const std::string& command, const std::string& message,
                           std::chrono::milliseconds timeout = 2s) {
    const auto start = std::chrono::steady_clock::now();
    try { call_external_command(command, false, timeout); assert(false); }
    catch (const exception::Exception& error) {
        assert(std::string(error.what()).find(message) != std::string::npos);
    }
    assert(std::chrono::steady_clock::now() - start < 2s);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::filesystem::path root(argv[1]);
    make_dirs(root);
    assert(call_external_command("sh -c 'printf stdout; printf stderr >&2'") == "stdoutstderr");
    assert(call_external_command("sleep 0.05; printf done") == "done");
    expect_failure("{ echo compiler-diagnostic >&2; exit 7; }", "compiler-diagnostic");
    expect_failure("exit 7", "exit code 7");
    expect_failure("kill -TERM $$", "exit code 143");
    std::cout << "PASS: output capture and exit/signal diagnostics\n";
    assert(call_external_command("printf first >&2; printf second") == "firstsecond");
    assert(call_external_command("printf 'a\\000b'") == std::string("a\0b", 3));
    assert(call_external_command("sleep 0.05; printf done", false, 0ms) == "done");
    for (const auto* command : {"exec sleep infinity", "exec 1>&- 2>&-; exec sleep infinity",
                               "while :; do echo progress; sleep 0.01; done", "sleep infinity & exit 0"})
        expect_failure(command, "command timed out after 100 ms", 100ms);
    expect_failure("mkfifo " + (root / "fifo").string() + "; exec cat " + (root / "fifo").string(),
                   "command timed out after 100 ms", 100ms);
    const auto pid_file = root / "child.pid";
    expect_failure("echo $$ > " + pid_file.string() + "; exec sleep infinity",
                   "command timed out after 100 ms", 100ms);
    const auto child = std::stoi(read(pid_file));
    const auto deadline = std::chrono::steady_clock::now() + 1s;
    int status;
    pid_t reaped;
    while ((reaped = waitpid(child, &status, WNOHANG)) == 0) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(1ms);
    }
    assert(reaped == child); // the command helper left this child for the caller to reap
    assert(WIFSIGNALED(status) and WTERMSIG(status) == SIGKILL);
    std::cout << "PASS: timeouts send SIGKILL without reaping the child or killing the host\n";

    // The suite starts this test in a private session. Keep the host alive to
    // verify terminal-style SIGINT reaches the shell in the same process group.
    assert(::getpgrp() == ::getpid());
    assert(std::stoi(call_external_command("ps -o pgid= -p $$")) == ::getpgrp());
    struct sigaction action{}, previous{};
    action.sa_handler = [](int) { interrupted = 1; };
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    assert(::sigaction(SIGINT, &action, &previous) == 0);
    expect_failure("kill -INT 0", "exit code 130");
    assert(interrupted);
    assert(::sigaction(SIGINT, &previous, nullptr) == 0);
    assert(call_external_command("printf recovered") == "recovered");
    std::cout << "PASS: group SIGINT reaches both host and command without forwarding\n";
}
