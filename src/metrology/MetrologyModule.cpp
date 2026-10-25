#include "../ToolFactory.h"
#include "AssistTools.h"
#include "CircleTool.h"
#include "RowSpacingTool.h"

#include <visionrt/contracts/Module.h>

namespace visiontools::metrology { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    const visionrt::ToolFactoryDescriptor factories[]{
        {   ports::measure_circle::TOOL_ID,       visiontools::createTool<CircleTool>, visiontools::destroyTool},
        {ports::measure_row_spacing::TOOL_ID, visiontools::createTool<RowSpacingTool>, visiontools::destroyTool},
        {              ports::assist::CIRCLE_ID, visiontools::createTool<AssistCircleTool>, visiontools::destroyTool},
        {                  ports::assist::ROW_ID, visiontools::createTool<AssistRowSpacingTool>, visiontools::destroyTool},
    };
    for (const auto &factory : factories)
        if (auto status = registry.add(factory); !status)
            return status;
    return {};
}
}} // namespace visiontools::metrology

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visiontools::metrology::registerTools};
    return &api;
}
