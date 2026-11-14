#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::roi_projection {
inline constexpr char TOOL_ID[] = "ProjectRoi";

namespace in {
inline constexpr visionrt::PortId AXIS = 2;
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId MERGE_GAP = 7;
inline constexpr visionrt::PortId MIN_EXTENT = 6;
inline constexpr visionrt::PortId MIN_PIXELS = 5;
inline constexpr visionrt::PortId PADDING = 8;
inline constexpr visionrt::PortId POLARITY = 3;
inline constexpr visionrt::PortId THRESHOLD = 4;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId ROIS = 3;
}
}

class RoiProjectionTool final : public visionrt::IToolInstance
{
public:
    ~RoiProjectionTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
};

} // namespace visiontools::preprocess
