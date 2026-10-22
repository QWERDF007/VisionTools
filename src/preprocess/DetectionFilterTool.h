#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::preprocess {

namespace ports::detection_filter {
inline constexpr char TOOL_ID[] = "visionrt.detection.filter";

namespace in {
inline constexpr char ALLOWED_CLASSES[] = "allowed_classes";
inline constexpr char CLASS_NAMES[]     = "class_names";
inline constexpr char CLASSES[]         = "classes";
inline constexpr char IMAGES[]          = "images";
inline constexpr char MIN_AREA[]        = "min_area";
inline constexpr char MIN_SCORE[]       = "min_score";
inline constexpr char ROIS[]            = "rois";
inline constexpr char SCORES[]          = "scores";
}

namespace out {
inline constexpr char CLASS_NAMES[] = "class_names";
inline constexpr char CLASSES[]     = "classes";
inline constexpr char COUNTS[]      = "counts";
inline constexpr char IMAGES[]      = "images";
inline constexpr char ROIS[]        = "rois";
inline constexpr char SCORES[]      = "scores";
}
}

class DetectionFilterTool final : public visionrt::IToolInstance
{
public:
    explicit DetectionFilterTool(const visionrt::ToolServices &services);
    ~DetectionFilterTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    visionrt::ToolServices services_;
};

} // namespace visiontools::preprocess
