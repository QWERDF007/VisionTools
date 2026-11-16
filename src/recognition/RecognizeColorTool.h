#pragma once

#include "ColorModel.h"
#include "TrainingAssets.h"

#include <visionrt/contracts/Tool.h>

namespace visiontools::recognition {

namespace ports::recognize_color {
inline constexpr char TOOL_ID[] = "RecognizeColor";

namespace init {
inline constexpr visionrt::PortId TRAINING = 0;
}

namespace in {
inline constexpr visionrt::PortId IMAGES      = 1;
inline constexpr visionrt::PortId ALIGNMENT   = 2;
inline constexpr visionrt::PortId REGIONS     = 3;
inline constexpr visionrt::PortId MIN_SCORE   = 4;
inline constexpr visionrt::PortId MIN_RATIO   = 5;
}

namespace out {
inline constexpr visionrt::PortId IMAGES       = 2;
inline constexpr visionrt::PortId REGIONS      = 3;
inline constexpr visionrt::PortId COLORS       = 4;
inline constexpr visionrt::PortId SCORES       = 5;
inline constexpr visionrt::PortId RATIOS       = 6;
inline constexpr visionrt::PortId CENTER_X     = 7;
inline constexpr visionrt::PortId CENTER_Y     = 8;
}
} // namespace ports::recognize_color

// Recognizes the trained colour in every detection circle. Circles are stored in reference
// coordinates and mapped to the current frame through the upstream alignment, so the same trained
// geometry follows the part without rotating the source image.
class RecognizeColorTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::Values &parameters) override;
    visionrt::ToolResult execute(const visionrt::Values &inputs) override;
    visionrt::ToolStatus release() override;

private:
    ColorModel model_;
};

} // namespace visiontools::recognition
