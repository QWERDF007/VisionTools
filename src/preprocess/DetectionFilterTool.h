#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::detection_filter {
inline constexpr char TOOL_ID[] = "FilterDetections";

namespace in {
inline constexpr visionrt::PortId ALLOWED_CLASSES = 8;
inline constexpr visionrt::PortId CLASS_NAMES = 5;
inline constexpr visionrt::PortId CLASSES = 4;
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId MIN_AREA = 7;
inline constexpr visionrt::PortId MIN_SCORE = 6;
inline constexpr visionrt::PortId ROIS  = 2;
inline constexpr visionrt::PortId SCORES = 3;
}

namespace out {
inline constexpr visionrt::PortId CLASS_NAMES = 6;
inline constexpr visionrt::PortId CLASSES = 5;
inline constexpr visionrt::PortId COUNTS = 7;
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId ROIS = 3;
inline constexpr visionrt::PortId SCORES = 4;
}
}

class DetectionFilterTool final : public visionrt::IToolInstance
{
public:
    ~DetectionFilterTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
};

} // namespace visiontools::preprocess
