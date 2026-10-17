#pragma once

#include <visionrt/sdk/Tool.h>

namespace visionrt::infer {

inline ToolResult missingResult(ToolStatus status) {
    return {std::move(status), {{"images", Missing{}}, {"rois", Missing{}},
        {"scores", Missing{}}, {"classes", Missing{}}, {"class_names", Missing{}}}};
}

} // namespace visionrt::infer
