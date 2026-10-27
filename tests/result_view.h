#pragma once

// Test-side view over a tool result. The contract has one execution channel: the tool writes every
// declared port, the two status keys included. These helpers read that channel instead of a second
// copy, so a test cannot pass on data the host never receives.

#include <visionrt/contracts/Tool.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace testview {

inline std::int64_t code(const visionrt::ToolResult &result)
{
    return std::get<std::int64_t>(result.outputs.at(std::string(visionrt::kStatusCodePort)));
}

inline std::string message(const visionrt::ToolResult &result)
{
    return std::get<std::string>(result.outputs.at(std::string(visionrt::kStatusMessagePort)));
}

inline visionrt::ToolStatus status(const visionrt::ToolResult &result)
{
    return {code(result), message(result)};
}

inline const visionrt::ToolStatus &status(const visionrt::ToolStatus &value)
{
    return value;
}

template<class T>
auto status(const T &value) -> std::decay_t<decltype(value.status)>
{
    return value.status;
}

// Business ports only, i.e. the declared ports minus the two status keys.
inline std::size_t businessCount(const visionrt::ToolResult &result)
{
    return result.outputs.size() - 2;
}

} // namespace testview
