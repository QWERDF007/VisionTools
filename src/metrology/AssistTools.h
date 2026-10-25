#pragma once

#include <visionrt/contracts/Tool.h>

namespace visiontools::metrology {

namespace ports::assist {
inline constexpr char CIRCLE_ID[] = "visiontools.assist.circle";
inline constexpr char ROW_ID[]    = "visiontools.assist.row_spacing";

namespace in {
inline constexpr char IMAGES[]        = "images";
inline constexpr char CURSOR[]        = "cursor";
inline constexpr char PHASE[]         = "phase";
inline constexpr char SEED_ROIS[]     = "seed_rois";
inline constexpr char ALIGNMENT[]     = "alignment";
inline constexpr char SEARCH_EXTENT[] = "search_extent";
}

namespace out {
inline constexpr char PREVIEW_ROIS[]        = "preview_rois";
inline constexpr char PROPOSAL_ROIS[]       = "proposal_rois";
inline constexpr char CANDIDATE_VALID[]     = "candidate_valid";
inline constexpr char QUALITY[]             = "quality";
inline constexpr char SUGGESTED_POLARITY[]  = "suggested_polarity";
inline constexpr char SUGGESTED_SEARCH[]    = "suggested_search_half_width";
inline constexpr char SUGGESTED_WIDTH_MIN[] = "suggested_width_min";
inline constexpr char SUGGESTED_WIDTH_MAX[] = "suggested_width_max";
}
}

class AssistCircleTool final : public visionrt::IToolInstance
{
public:
    explicit AssistCircleTool(const visionrt::ToolServices &) {}
    ~AssistCircleTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

class AssistRowSpacingTool final : public visionrt::IToolInstance
{
public:
    explicit AssistRowSpacingTool(const visionrt::ToolServices &) {}
    ~AssistRowSpacingTool() override = default;

    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
};

} // namespace visiontools::metrology
