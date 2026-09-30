// CPU-only: c++ -std=c++20 -I include tests/test_shell_args.cpp -o /tmp/test-shell-args && /tmp/test-shell-args
#include <deep_jit/utils/str.hpp>
#include <array>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <sys/wait.h>

int main() {
    using namespace deep_jit::str;
    const std::vector<std::string> values = {"", "a b", "a'b", "$HOME", "line\nnext", "中文", "\\", "\"", "*", "--flag"};
    std::vector<std::string> args = {"/usr/bin/printf", "%s\\0"};
    args.insert(args.end(), values.begin(), values.end());
    FILE* pipe = popen(shell_join(args).c_str(), "r");
    assert(pipe != nullptr);
    std::string result;
    std::array<char, 256> buffer;
    while (const auto n = fread(buffer.data(), 1, buffer.size(), pipe)) result.append(buffer.data(), n);
    const auto status = pclose(pipe);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    std::string expected;
    for (const auto& value : values) { expected += value; expected += '\0'; }
    assert(result == expected);
    assert(join({"a b", "c"}) == "a b c");
    assert(shell_join({}).empty());
    assert(shell_quote("") == "''");
    bool rejected = false;
    try { (void)shell_quote(std::string("a\0b", 3)); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
