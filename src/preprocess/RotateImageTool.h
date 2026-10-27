#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_rotate {
inline constexpr char TOOL_ID[] = "visiontools.image.rotate";

namespace in {
inline constexpr char IMAGES[]        = "images";
inline constexpr char ANGLE_DEG[]     = "angle_deg";
inline constexpr char CENTER[]        = "center";
inline constexpr char CANVAS[]        = "canvas";
inline constexpr char INTERPOLATION[] = "interpolation";
inline constexpr char BORDER_VALUE[]  = "border_value";
inline constexpr char ALIGNMENT[]     = "alignment";
}

namespace out {
inline constexpr char IMAGES[]    = "images";
inline constexpr char ALIGNMENT[] = "alignment";
}
}

class RotateImageTool final : public visionrt::IToolInstance
{
public:
    explicit RotateImageTool(const visionrt::ToolServices &) {}
    ~RotateImageTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::preprocess
