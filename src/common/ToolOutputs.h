#pragma once

#include <visionrt/contracts/Tool.h>

#include <initializer_list>

namespace visiontools {

// A failed result still has to carry every declared output port as `Missing`; a port that simply
// disappears looks to a downstream consumer like a wiring bug rather than an explicit "no value".
// Tools with a long output list use this instead of hand-rolling the same loop, so adding a port
// has one place to change. Tools with one or two ports keep the brace list inline.
inline visionrt::Values missingPorts(std::initializer_list<const char *> ports)
{
    visionrt::Values outputs;
    for (const auto *id : ports)
        outputs.emplace(id, visionrt::Missing{});
    return outputs;
}

} // namespace visiontools
