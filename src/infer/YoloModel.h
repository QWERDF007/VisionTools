#pragma once

#include "YoloPostprocess.h"

#include <visionrt/sdk/Tool.h>

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
