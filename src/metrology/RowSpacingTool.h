#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::measure_row_spacing {
inline constexpr char TOOL_ID[] = "visiontools.measure.row_spacing";

namespace in {
inline constexpr char IMAGES[]          = "images";
inline constexpr char ROIS[]            = "rois";
inline constexpr char ALIGNMENT[]       = "alignment";
inline constexpr char SIGMA[]           = "sigma";
inline constexpr char MIN_GRADIENT[]    = "min_gradient";
inline constexpr char OBJECT_POLARITY[] = "object_polarity";
inline constexpr char WIDTH_MIN[]       = "width_min";
inline constexpr char WIDTH_MAX[]       = "width_max";
inline constexpr char UNITS_PER_PIXEL[] = "units_per_pixel";
inline constexpr char UNIT[]            = "unit";
}

namespace out {
inline constexpr char IMAGES[]          = "images";
inline constexpr char VALID_ROI[]       = "valid_roi";
inline constexpr char REASONS[]         = "reasons";
inline constexpr char PAIR_COUNT[]      = "pair_count";
inline constexpr char ACCEPTED_COUNT[]  = "accepted_count";
inline constexpr char UNPAIRED_COUNT[]  = "unpaired_count";
inline constexpr char ACCEPTED[]        = "accepted";
inline constexpr char WIDTH_PX[]        = "width_px";
inline constexpr char GAP_PX[]          = "gap_px";
inline constexpr char PITCH_PX[]        = "pitch_px";
inline constexpr char WIDTH[]           = "width";
inline constexpr char GAP[]             = "gap";
inline constexpr char PITCH[]           = "pitch";
inline constexpr char PAIR_POINTS[]     = "pair_points";
inline constexpr char OVERLAY_SEGMENTS[] = "overlay_segments";
inline constexpr char UNIT[]            = "unit";
}
}

class RowSpacingTool final : public visionrt::IToolInstance
{
public:
    explicit RowSpacingTool(const visionrt::ToolServices &) {}
    ~RowSpacingTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
