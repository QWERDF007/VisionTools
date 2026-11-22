#include "CircleTool.h"
#include "LocatorTools.h"
#include "RowSpacingTool.h"

#include <visionrt/contracts/FunctionTool.h>
#include <visionrt/contracts/Module.h>

namespace visiontools::metrology { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    if (auto s = visionrt::registerClass<CircleTool>(registry, ports::measure_circle::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<RowSpacingTool>(registry, ports::measure_row_spacing::TOOL_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<LocateCircleTool>(registry, ports::locate::CIRCLE_ID); !s)
        return s;
    if (auto s = visionrt::registerClass<LocateRowSpacingTool>(registry, ports::locate::ROW_ID); !s)
        return s;
    return {};
}
}} // namespace visiontools::metrology

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visiontools::metrology::registerTools};
    return &api;
}
