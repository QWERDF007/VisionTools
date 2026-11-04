#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::image_rotate {
inline constexpr char TOOL_ID[] = "visiontools.image.rotate";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId ANGLE_DEG = 2;
inline constexpr visionrt::PortId CENTER = 3;
inline constexpr visionrt::PortId CANVAS = 4;
inline constexpr visionrt::PortId INTERPOLATION = 5;
inline constexpr visionrt::PortId BORDER_VALUE = 6;
inline constexpr visionrt::PortId ALIGNMENT = 7;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId ALIGNMENT = 3;
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
