#include "AlignStraightBordersTool.h"
#include "LocateStraightEdgeTool.h"
#include "RecognizeColorTool.h"
#include "TrainColorModel.h"
#include "TrainStraightBorders.h"

#include <visionrt/contracts/FunctionTool.h>
#include <visionrt/contracts/Module.h>

namespace visiontools::recognition { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    if (auto status = visionrt::registerClass<AlignStraightBordersTool>(
            registry, ports::align_straight_borders::TOOL_ID);
        !status)
        return status;
    if (auto status = visionrt::registerClass<RecognizeColorTool>(registry, ports::recognize_color::TOOL_ID);
        !status)
        return status;
    if (auto status = visionrt::registerClass<LocateStraightEdgeTool>(registry, ports::locate_straight_edge::TOOL_ID);
        !status)
        return status;
    if (auto status = visionrt::registerClass<TrainColorModelTool>(registry, ports::train_color_model::TOOL_ID);
        !status)
        return status;
    return visionrt::registerClass<TrainStraightBordersTool>(registry, ports::train_straight_borders::TOOL_ID);
}
}} // namespace visiontools::recognition

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visiontools::recognition::registerTools};
    return &api;
}
