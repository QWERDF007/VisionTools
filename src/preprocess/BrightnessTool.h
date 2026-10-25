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

class BrightnessTool final : public visionrt::IToolInstance
{
public:
    explicit BrightnessTool(const visionrt::ToolServices &) {}
    ~BrightnessTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::preprocess
