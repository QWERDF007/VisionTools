#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::measure_row_spacing {
inline constexpr char TOOL_ID[] = "MeasureRowSpacing";

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId ROIS  = 2;
inline constexpr visionrt::PortId ALIGNMENT = 3;
inline constexpr visionrt::PortId SIGMA = 4;
inline constexpr visionrt::PortId MIN_GRADIENT = 5;
inline constexpr visionrt::PortId OBJECT_POLARITY = 6;
inline constexpr visionrt::PortId WIDTH_MIN = 7;
inline constexpr visionrt::PortId WIDTH_MAX = 8;
inline constexpr visionrt::PortId UNITS_PER_PIXEL = 9;
inline constexpr visionrt::PortId UNIT  = 10;
}

namespace out {
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId VALID_ROI = 3;
inline constexpr visionrt::PortId REASONS = 4;
inline constexpr visionrt::PortId PAIR_COUNT = 5;
inline constexpr visionrt::PortId ACCEPTED_COUNT = 6;
inline constexpr visionrt::PortId UNPAIRED_COUNT = 7;
inline constexpr visionrt::PortId ACCEPTED = 8;
inline constexpr visionrt::PortId WIDTH_PX = 9;
inline constexpr visionrt::PortId GAP_PX = 10;
inline constexpr visionrt::PortId PITCH_PX = 11;
inline constexpr visionrt::PortId WIDTH = 12;
inline constexpr visionrt::PortId GAP   = 13;
inline constexpr visionrt::PortId PITCH = 14;
inline constexpr visionrt::PortId PAIR_POINTS = 15;
inline constexpr visionrt::PortId OVERLAY_SEGMENTS = 16;
inline constexpr visionrt::PortId UNIT  = 17;
}
}

class RowSpacingTool final : public visionrt::IToolInstance
{
public:
    ~RowSpacingTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
