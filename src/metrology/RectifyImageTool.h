#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::rectify_image {
inline constexpr char TOOL_ID[] = "RectifyImage";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId CALIBRATION = 2;
inline constexpr visionrt::PortId UNIT = 3;
} // namespace in

namespace out {
inline constexpr visionrt::PortId RECTIFIED_IMAGES = 2;
inline constexpr visionrt::PortId ADAPTED_CALIBRATION = 3;
inline constexpr visionrt::PortId PIXEL_SIZE = 4;
inline constexpr visionrt::PortId COORDINATE_FRAME = 5;
} // namespace out
} // namespace ports::rectify_image

class RectifyImageTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &configuration) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
