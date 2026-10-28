#pragma once

#include <visionrt/contracts/Tool.h>

#include <cstdint>
#include <string>

namespace visiontools {

// The upstream status every contract carries. The framework never blocks a node on a business
// status, so each tool decides for itself what a non-zero status means. Reading it is a type
// check, not a policy: the policy belongs to the tool. The host normalises this port from the
// declaration, so anything but an integer means "no upstream status".
inline std::int64_t upstreamStatus(const visionrt::Values &inputs) noexcept
{
    const auto found = inputs.find(std::string(visionrt::kUpstreamStatusPort));
    if (found == inputs.end())
        return 0;
    const auto *code = std::get_if<std::int64_t>(&found->second);
    return code ? *code : 0;
}

// The answer of an image or measurement tool whose upstream branch was not selected: it does no
// algorithm work, touches no file, and keeps the upstream status as its own so the branch stays
// visibly unprocessed instead of looking like a zero-millisecond success.
inline visionrt::ToolResult skippedByUpstream(std::int64_t status, visionrt::Values missingOutputs)
{
    return visionrt::toolResult(std::move(missingOutputs), {status, "上游未选中本分支，工具未执行"});
}

} // namespace visiontools
