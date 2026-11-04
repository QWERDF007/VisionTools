#pragma once

#include <visionrt/contracts/Tool.h>

namespace visionrt::infer {

inline ToolResult missingResult(ToolStatus status)
{
    return toolResult({{2, Missing{}},
                       {3, Missing{}},
                       {4, Missing{}},
                       {5, Missing{}},
                       {6, Missing{}}},
                      std::move(status));
}

} // namespace visionrt::infer
