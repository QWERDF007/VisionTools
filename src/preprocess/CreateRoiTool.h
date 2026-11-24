#pragma once
#include "common/InputValues.h"
#include "common/ToolOutputs.h"
#include "common/UpstreamStatus.h"
#include "common/errors/ToolError.h"

#include <visionrt/contracts/FunctionTool.h>

namespace visiontools::preprocess {
template<visionrt::RoiKind Kind>
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
        const auto regions = readings.roi(1);
        if (readings.failed())
            return visionrt::toolResult(
                missingPorts({2}), readings.status(CommonToolError::InputMissing, CommonToolError::InputTypeMismatch));
        try
        {
            for (const auto &group : regions.get())
                for (const auto &encoded : group)
                    if (visionrt::RegionView(encoded).kind() != Kind)
                        return failure(missingPorts({2}), CommonToolError::InputTypeMismatch, "几何类型与创建工具不符");
            return visionrt::toolResult(
                {
                    {2, regions}
            },
                visionrt::executionSuccess());
        }
        catch (const visionrt::ContractError &error)
        {
            return failure(missingPorts({2}), CommonToolError::InputOutOfRange, "几何数据无效", error.what());
        }
    }
};

inline visionrt::ToolStatus registerRoiCreators(visionrt::IToolRegistry &registry)
{
    using visionrt::RoiKind;
    if (auto result = visionrt::registerClass<CreateRoiTool<RoiKind::Point>>(registry, "CreatePoint"); !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<RoiKind::Segment>>(registry, "CreateLine"); !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<RoiKind::Rectangle>>(registry, "CreateRotatedRectangle");
        !result)
        return result;
    if (auto result = visionrt::registerClass<CreateRoiTool<RoiKind::Circle>>(registry, "CreateCircle"); !result)
        return result;
    return visionrt::registerClass<CreateRoiTool<RoiKind::Polygon>>(registry, "CreatePolygon");
}
} // namespace visiontools::preprocess
