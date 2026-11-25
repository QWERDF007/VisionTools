#pragma once

#include "TemplateMatcher.h"

#include <visionrt/contracts/Tool.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace visiontools::recognition {

namespace ports::check_presence {
inline constexpr char TOOL_ID[] = "CheckPresenceWithMatching";
inline constexpr char MODEL_ARTIFACT[] = "presence-model";

namespace train {
inline constexpr visionrt::PortId MODEL_PATH = 1;
inline constexpr visionrt::PortId METHOD = 2;
inline constexpr visionrt::PortId MIN_SCORE = 3;
inline constexpr visionrt::PortId ANGLE_TOLERANCE = 4;
inline constexpr visionrt::PortId MAX_OVERLAP = 5;
inline constexpr visionrt::PortId MODEL_CONTRAST = 6;
inline constexpr visionrt::PortId OBJECT_DATASET = 7;
inline constexpr visionrt::PortId POSITION_DATASET = 8;
} // namespace train

namespace field {
inline constexpr visionrt::TrainingFieldId OBJECT_ROI = 2;
inline constexpr visionrt::TrainingFieldId REFERENCE_POINTS = 3;
} // namespace field

namespace in {
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId ALIGNMENTS = 2;
inline constexpr visionrt::PortId MAX_DISTANCE = 3;
inline constexpr visionrt::PortId MIN_SCORE = 4;
inline constexpr visionrt::PortId MAX_OVERLAP = 5;
inline constexpr visionrt::PortId FAILED_PROCESSING = 6;
} // namespace in

namespace out {
inline constexpr visionrt::PortId MODEL_PATH = 2;
inline constexpr visionrt::PortId OBJECT_ACCEPTED = 3;
inline constexpr visionrt::PortId ALL_ACCEPTED = 4;
inline constexpr visionrt::PortId ABSOLUTE_DISTANCE = 5;
inline constexpr visionrt::PortId RELATIVE_DISTANCE = 6;
inline constexpr visionrt::PortId ACCEPTED_OBJECTS = 7;
inline constexpr visionrt::PortId REJECTED_POSITIONS = 8;
inline constexpr visionrt::PortId REFERENCE_POSITIONS = 9;
} // namespace out
} // namespace ports::check_presence

class CheckPresenceTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &configuration) override;
    visionrt::ToolResult train(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    static visionrt::Values missingOutputs();
    std::string modelPath_;
    TemplateModel objectModel_;
    std::vector<cv::Point2d> referencePositions_;
};

} // namespace visiontools::recognition
