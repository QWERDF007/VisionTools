#pragma once

#include "YoloPostprocess.h"

#include <visionrt/contracts/Tool.h>

#include <array>

namespace visionrt::infer {

struct TensorDescription
{
    std::string               name;
    std::string               dtype;
    std::string               layout;
    std::vector<std::int64_t> shape;
};

struct ImageTrace
{
    LetterboxGeometry                 geometry;
    std::array<std::vector<float>, 3> rawOutputs;
    std::vector<Detection>            candidates;
    std::vector<Detection>            detections;
};

struct GeometryProbe
{
    LetterboxGeometry geometry;
    double            maximumPixelError = 0;
    double            paddingPixelError = 0;
};

GeometryProbe probePreprocessing();

// One model/session/stream and reusable staging buffers per node. Not reentrant;
// the runtime serializes calls on each node, not across different nodes.
namespace ports::yolo {
inline constexpr char TOOL_ID[] = "DetectObjectsYolo";

namespace init {
inline constexpr visionrt::PortId ENGINE_OUTPUT_PATH = 3;
inline constexpr visionrt::PortId GPU_INDEX = 6;
inline constexpr visionrt::PortId INPUT_HEIGHT = 8;
inline constexpr visionrt::PortId INPUT_WIDTH = 7;
inline constexpr visionrt::PortId LABELS_PATH = 4;
inline constexpr visionrt::PortId LOAD_MODE = 1;
inline constexpr visionrt::PortId MODEL_NAME = 0;
inline constexpr visionrt::PortId MODEL_PATH = 2;
inline constexpr visionrt::PortId NUM_CLASSES = 9;
inline constexpr visionrt::PortId PRECISION = 5;
}

namespace in {
inline constexpr visionrt::PortId CONF_THRESHOLD = 2;
inline constexpr visionrt::PortId IMAGES = 1;
inline constexpr visionrt::PortId MAX_DETECTIONS = 4;
inline constexpr visionrt::PortId NMS_THRESHOLD = 3;
}

namespace out {
inline constexpr visionrt::PortId CLASS_NAMES = 6;
inline constexpr visionrt::PortId CLASSES = 5;
inline constexpr visionrt::PortId IMAGES = 2;
inline constexpr visionrt::PortId ROIS = 3;
inline constexpr visionrt::PortId SCORES = 4;
}
}

class YoloTool final : public IToolInstance
{
public:
    YoloTool();
    ~YoloTool() override;
    ToolStatus                     initialize(const Values &parameters) override;
    ToolResult                     execute(const Values &inputs) override;
    ToolStatus                     release() override;
    // The standalone probe crosses exactly the same execution path. Capture is
    // opt-in and never retains staging-buffer aliases after the call returns.
    ToolResult                     executeWithTrace(const Values &inputs, std::vector<ImageTrace> &traces);
    std::vector<TensorDescription> tensorDescriptions() const;

private:
    struct State;
    ToolResult             run(const Values &inputs, std::vector<ImageTrace> *traces);
    std::unique_ptr<State> state_;
};

} // namespace visionrt::infer
