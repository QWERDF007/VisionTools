#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_brightness {
inline constexpr char TOOL_ID[] = "visiontools.image.brightness";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId GAIN = 2;
inline constexpr visionrt::PortId OFFSET = 3;
inline constexpr visionrt::PortId ROIS = 4;
inline constexpr visionrt::PortId MASK = 5;
inline constexpr visionrt::PortId ALIGNMENT = 6;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId APPLIED_MASK = 3;
}
}

// A stateless tool: gain, offset, ROI list, mask and alignment all arrive with this call, so there
// is no instance state to reset between nodes. registerFunction adapts it to IToolInstance, which
// keeps scheduling, cancellation, readback and teardown on the existing path.
visionrt::ToolResult brightnessTool(const visionrt::Values &inputs, const visionrt::ToolServices &services);

} // namespace visiontools::preprocess
