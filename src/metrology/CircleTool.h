#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::measure_circle {
inline constexpr char TOOL_ID[] = "visiontools.measure.circle";

namespace in {
inline constexpr char IMAGES[]            = "images";
inline constexpr char ROIS[]              = "rois";
inline constexpr char ALIGNMENT[]         = "alignment";
inline constexpr char NUM_CALIPERS[]      = "num_calipers";
inline constexpr char SEARCH_HALF_WIDTH[] = "search_half_width";
inline constexpr char AVERAGE_WIDTH[]     = "average_width";
inline constexpr char SIGMA[]             = "sigma";
inline constexpr char MIN_GRADIENT[]      = "min_gradient";
inline constexpr char POLARITY[]          = "polarity";
inline constexpr char MIN_POINTS[]        = "min_points";
inline constexpr char MIN_COMPLETENESS[]  = "min_completeness";
inline constexpr char MIN_COVERAGE_DEG[]  = "min_coverage_deg";
inline constexpr char RADIUS_MIN[]        = "radius_min";
inline constexpr char RADIUS_MAX[]        = "radius_max";
inline constexpr char MAX_RMS[]           = "max_rms";
inline constexpr char UNITS_PER_PIXEL[]   = "units_per_pixel";
inline constexpr char UNIT[]              = "unit";
}

namespace out {
inline constexpr char IMAGES[]            = "images";
inline constexpr char VALID[]             = "valid";
inline constexpr char REASONS[]           = "reasons";
inline constexpr char VALID_COUNT[]       = "valid_count";
inline constexpr char SOURCE_ROI_INDICES[] = "source_roi_indices";
inline constexpr char CIRCLES[]           = "circles";
inline constexpr char CENTERS[]           = "centers";
inline constexpr char RADIUS_PX[]         = "radius_px";
inline constexpr char RADIUS[]            = "radius";
inline constexpr char RMS[]               = "rms";
inline constexpr char COMPLETENESS[]      = "completeness";
inline constexpr char USED_EDGES[]        = "used_edges";
inline constexpr char UNIT[]              = "unit";
}
}

class CircleTool final : public visionrt::IToolInstance
{
public:
    explicit CircleTool(const visionrt::ToolServices &) {}
    ~CircleTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
