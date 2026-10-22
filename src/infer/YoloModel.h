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
inline constexpr char TOOL_ID[] = "visionrt.dl.yolo";

namespace init {
inline constexpr char ENGINE_OUTPUT_PATH[] = "engine_output_path";
inline constexpr char GPU_INDEX[]          = "gpu_index";
inline constexpr char INPUT_HEIGHT[]       = "input_height";
inline constexpr char INPUT_WIDTH[]        = "input_width";
inline constexpr char LABELS_PATH[]        = "labels_path";
inline constexpr char LOAD_MODE[]          = "load_mode";
inline constexpr char MODEL_NAME[]         = "model_name";
inline constexpr char MODEL_PATH[]         = "model_path";
inline constexpr char NUM_CLASSES[]        = "num_classes";
inline constexpr char PRECISION[]          = "precision";
}

namespace in {
inline constexpr char CONF_THRESHOLD[] = "conf_threshold";
inline constexpr char IMAGES[]         = "images";
inline constexpr char MAX_DETECTIONS[] = "max_detections";
inline constexpr char NMS_THRESHOLD[]  = "nms_threshold";
}

namespace out {
inline constexpr char CLASS_NAMES[] = "class_names";
inline constexpr char CLASSES[]     = "classes";
inline constexpr char IMAGES[]      = "images";
inline constexpr char ROIS[]        = "rois";
inline constexpr char SCORES[]      = "scores";
}
}

class YoloTool final : public IToolInstance
{
public:
    explicit YoloTool(const ToolServices &services);
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
    PathResolver           paths_;
    std::unique_ptr<State> state_;
};

} // namespace visionrt::infer
