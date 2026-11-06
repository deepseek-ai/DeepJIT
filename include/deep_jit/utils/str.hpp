#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace deep_jit::str {

inline std::string join(const std::vector<std::string>& values, const std::string_view separator = " ") {
    if (values.empty())
        return {};

    std::size_t size = separator.size() * (values.size() - 1);
    for (const auto& value : values)
        size += value.size();

    std::string result;
    result.reserve(size);
    result += values.front();
    for (std::size_t i = 1; i < values.size(); ++i) {
        result.append(separator);
        result.append(values[i]);
    }
    return result;
}


// Render one argv element for the POSIX shell used by popen(). Keep join()
// unchanged: it also defines existing cache-key and diagnostic formats.
inline std::string shell_quote(const std::string_view value) {
    if (value.find('\0') != std::string_view::npos)
        throw std::invalid_argument("shell argument must not contain NUL");
    std::string result = "'";
    for (const char c : value) {
        if (c == '\'')
            result += "'\\''";
        else
            result += c;
    }
    result += '\'';
    return result;
}

inline std::string shell_join(const std::vector<std::string>& values) {
    std::vector<std::string> quoted;
    quoted.reserve(values.size());
    for (const auto& value : values)
        quoted.emplace_back(shell_quote(value));
    return join(quoted);
}

}  // namespace deep_jit::str
