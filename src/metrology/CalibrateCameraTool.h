#pragma once

#include "CalibrationModel.h"
#include <visionrt/contracts/Tool.h>
#include <optional>

namespace visiontools::metrology {
namespace ports::calibrate_camera {
inline constexpr char TOOL_ID[] = "CalibrateCamera";
inline constexpr char MODEL_ARTIFACT[] = "calibration-model";
namespace train {
inline constexpr visionrt::PortId DATASET = 0, PLATE_DESCRIPTION = 1, MODEL_PATH = 2, MAX_REPROJECTION_ERROR = 3;
}
namespace in {
inline constexpr visionrt::PortId IMAGES = 1, REFERENCE_SEGMENTS = 2, REFERENCE_LENGTH = 3, UNIT = 4;
}
namespace out {
inline constexpr visionrt::PortId MODEL_PATH = 2, CALIBRATION = 3, MARKS = 4, REPROJECTION_ERROR = 5, COORDINATE_FRAME = 6;
}
}

class CalibrateCameraTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &configuration) override;
    visionrt::ToolResult train(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;
private:
    std::optional<CalibrationModel> model_;
    std::string modelPath_;
};
}
