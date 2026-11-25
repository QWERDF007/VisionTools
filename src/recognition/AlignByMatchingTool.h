#pragma once

#include "TemplateMatcher.h"

#include <visionrt/contracts/Tool.h>
#include <string>

namespace visiontools::recognition {

namespace ports::align_by_matching {
inline constexpr char TOOL_ID[] = "DetermineAlignmentWithMatching";
inline constexpr char MODEL_ARTIFACT[] = "alignment-model";

namespace train {
inline constexpr visionrt::PortId MODEL_PATH = 1;
inline constexpr visionrt::PortId METHOD = 2;
inline constexpr visionrt::PortId MODEL_CONTRAST = 3;
inline constexpr visionrt::PortId SEARCH_CONTRAST = 4;
inline constexpr visionrt::PortId ANGLE_TOLERANCE = 5;
inline constexpr visionrt::PortId ANGLE_STEP = 6;
inline constexpr visionrt::PortId POLARITY = 7;
inline constexpr visionrt::PortId NUM_FEATURES = 8;
inline constexpr visionrt::PortId DATASET = 9;
} // namespace train

namespace field {
inline constexpr visionrt::TrainingFieldId TEMPLATE_ROI = 2;
} // namespace field

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId SEARCH_ROI = 2;
inline constexpr visionrt::PortId MIN_SCORE = 3;
inline constexpr visionrt::PortId ANGLE_TOLERANCE = 4;
} // namespace in

namespace out {
inline constexpr visionrt::PortId ALIGNMENTS = 2;
inline constexpr visionrt::PortId SCORES = 3;
inline constexpr visionrt::PortId MATCH_REGIONS = 4;
inline constexpr visionrt::PortId MODEL_PATH = 5;
} // namespace out
} // namespace ports::align_by_matching

class AlignByMatchingTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &configuration) override;
    visionrt::ToolResult train(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    static visionrt::Values missingOutputs();
    std::string modelPath_;
    TemplateModel model_;
};

} // namespace visiontools::recognition
