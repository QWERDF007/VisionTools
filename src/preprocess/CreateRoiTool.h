#pragma once
#include "common/InputValues.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"

#include <visionrt/contracts/FunctionTool.h>

namespace visiontools::preprocess {
template<class Shape>
class CreateRoiTool final : public visionrt::IToolInstance
{
public:
    visionrt::ToolStatus initialize(const visionrt::ToolConfiguration &configuration) override
    {
        const auto &parameters = configuration.parameters;
        return parameters.empty()
                 ? visionrt::ToolStatus{}
                 : visionrt::ToolStatus{visionrt::kContractViolationStatus, "创建工具不接受初始化参数"};
    }

    visionrt::ToolStatus release() override
    {
        return {};
    }

    visionrt::ToolResult execute(const visionrt::Values &inputs) override
    {
        if (const auto upstream = upstreamStatus(inputs); upstream != 0)
            return skippedByUpstream(upstream, missingPorts({2}));
        Readings   readings(inputs, "CreateRoi");
        const auto regions = readings.shape<Shape>(1);
        if (readings.failed())
            return visionrt::toolResult(
                missingPorts({2}), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
        return visionrt::toolResult({{2, regions}}, visionrt::executionSuccess());
    }
};

inline visionrt::ToolStatus registerRoiCreators(visionrt::IToolRegistry &registry)
{
    if (auto result = visionrt::registerClass<CreateRoiTool<visionrt::Point>>(registry, "CreatePoint"); !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<visionrt::Line>>(registry, "CreateLine"); !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<visionrt::Rectangle>>(registry, "CreateRotatedRectangle");
        !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<visionrt::Circle>>(registry, "CreateCircle"); !result)
        return result;
    return visionrt::registerClass<CreateRoiTool<visionrt::Polygon>>(registry, "CreatePolygon");
}
} // namespace visiontools::preprocess
