#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_brightness {
inline constexpr char TOOL_ID[] = "visiontools.image.brightness";

namespace in {
inline constexpr char IMAGES[]     = "images";
inline constexpr char GAIN[]       = "gain";
inline constexpr char OFFSET[]     = "offset";
inline constexpr char ROIS[]       = "rois";
inline constexpr char MASK[]       = "mask";
inline constexpr char ALIGNMENT[]  = "alignment";
inline constexpr char EMIT_MASK[]  = "emit_mask";
}

namespace out {
inline constexpr char IMAGES[]       = "images";
inline constexpr char APPLIED_MASK[] = "applied_mask";
}
}

// A stateless tool: gain, offset, ROI list, mask and alignment all arrive with this call, so there
// is no instance state to reset between nodes. registerFunction adapts it to IToolInstance, which
// keeps scheduling, cancellation, readback and teardown on the existing path.
visionrt::ToolResult brightnessTool(const visionrt::Values &inputs, const visionrt::ToolServices &services);

} // namespace visiontools::preprocess
