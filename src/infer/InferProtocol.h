#pragma once

#include <visionrt/contracts/Tool.h>

namespace visionrt::infer {

inline ToolResult missingResult(ToolStatus status)
{
    return toolResult({{"images", Missing{}},
                       {"rois", Missing{}},
                       {"scores", Missing{}},
                       {"classes", Missing{}},
                       {"class_names", Missing{}}},
                      std::move(status));
}

} // namespace visionrt::infer
