#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::measure_circle {
inline constexpr char TOOL_ID[] = "MeasureCircle";

namespace in {
inline constexpr visionrt::PortId IMAGES  = 1;
inline constexpr visionrt::PortId ROIS    = 2;
inline constexpr visionrt::PortId ALIGNMENT = 3;
inline constexpr visionrt::PortId NUM_CALIPERS = 4;
inline constexpr visionrt::PortId SEARCH_HALF_WIDTH = 5;
inline constexpr visionrt::PortId AVERAGE_WIDTH = 6;
inline constexpr visionrt::PortId SIGMA   = 7;
inline constexpr visionrt::PortId MIN_GRADIENT = 8;
inline constexpr visionrt::PortId POLARITY = 9;
inline constexpr visionrt::PortId MIN_POINTS = 10;
inline constexpr visionrt::PortId MIN_COMPLETENESS = 11;
inline constexpr visionrt::PortId MIN_COVERAGE_DEG = 12;
inline constexpr visionrt::PortId RADIUS_MIN = 13;
inline constexpr visionrt::PortId RADIUS_MAX = 14;
inline constexpr visionrt::PortId MAX_RMS = 15;
inline constexpr visionrt::PortId UNITS_PER_PIXEL = 16;
inline constexpr visionrt::PortId UNIT    = 17;
}

namespace out {
inline constexpr visionrt::PortId IMAGES  = 2;
inline constexpr visionrt::PortId VALID   = 3;
inline constexpr visionrt::PortId REASONS = 4;
inline constexpr visionrt::PortId VALID_COUNT = 5;
inline constexpr visionrt::PortId SOURCE_ROI_INDICES = 6;
inline constexpr visionrt::PortId CIRCLES = 7;
inline constexpr visionrt::PortId CENTERS = 8;
inline constexpr visionrt::PortId RADIUS_PX = 9;
inline constexpr visionrt::PortId RADIUS  = 10;
inline constexpr visionrt::PortId RMS     = 11;
inline constexpr visionrt::PortId COMPLETENESS = 12;
inline constexpr visionrt::PortId USED_EDGES = 13;
inline constexpr visionrt::PortId UNIT    = 14;
}
}

class CircleTool final : public visionrt::IToolInstance
{
public:
    ~CircleTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
