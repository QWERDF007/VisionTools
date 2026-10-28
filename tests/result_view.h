#pragma once

// Test-side view over what a tool or a node actually produced. A tool answers with business ports
// plus one status; the host is what turns that status into the two published status ports. These
// helpers read whichever of the two a test is holding, so a test cannot pass on data the host never
// receives.

#include <visionrt/contracts/Tool.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace testview {

inline std::int64_t code(const visionrt::ToolResult &result)
{
    return result.status.code;
}

inline std::string message(const visionrt::ToolResult &result)
{
    return result.status.message;
}

inline visionrt::ToolStatus status(const visionrt::ToolResult &result)
{
    return result.status;
}

// A lifecycle status already is the status itself.
inline const visionrt::ToolStatus &status(const visionrt::ToolStatus &value)
{
    return value;
}

// Anything that carries its own lifecycle status member (a node snapshot, registry availability, a
// registry failure, a load result) is read through that member instead of duplicated.
template<class T>
auto status(const T &value) -> std::decay_t<decltype(value.status)>
{
    return value.status;
}

// A tool result already holds business ports only.
inline std::size_t businessCount(const visionrt::ToolResult &result)
{
    return result.outputs.size();
}

} // namespace testview
