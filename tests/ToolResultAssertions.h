#pragma once

// Test-side view over what a tool or a node actually produced. Tool output packs the two status
// ports into outputs and carries detail separately. These helpers read whichever of the two a
// test is holding, so a test cannot pass on data the host never receives.

#include <visionrt/contracts/Tool.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace testview {

inline std::int64_t code(const visionrt::ToolResult &result)
{
    return std::get<std::int64_t>(result.outputs.at(visionrt::kStatusCodePort));
}

inline const std::string &message(const visionrt::ToolResult &result)
{
    return std::get<std::string>(result.outputs.at(visionrt::kStatusMessagePort));
}

inline visionrt::ToolStatus status(const visionrt::ToolResult &result)
{
    return {code(result), message(result), result.detail};
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

// Filters out the two system status ports so tests can count business ports.
inline std::size_t businessCount(const visionrt::ToolResult &result)
{
    std::size_t count = 0;
    for (const auto &[key, _] : result.outputs)
        if (!visionrt::isStatusPort(key))
            ++count;
    return count;
}

// Runs an execution the contract cannot satisfy. Such a run has no tool answer, so the failure
// surfaces as the exception the host turns into a diagnostic; this returns that message.
template<class Instance>
std::string executionFailure(Instance &instance, visionrt::Values inputs = {})
{
    try
    {
        instance.execute(inputs);
    }
    catch (const std::exception &error)
    {
        return error.what();
    }
    return {};
}

} // namespace testview

// Reading several outputs of one run is the common case in these tests, so the readable nested form
// of an Array stays visible unqualified next to the fixture code instead of repeated on every line.
using visionrt::nestedOf;
