#include "YoloModel.h"

#include <visionrt/contracts/FunctionTool.h>
#include <visionrt/contracts/Module.h>

namespace visionrt::infer { namespace {
visionrt::ToolStatus VT_CALL registerTools(visionrt::IToolRegistry &registry) noexcept
{
    return visionrt::registerClass<YoloTool>(registry, ports::yolo::TOOL_ID);
}
}}

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    static const visionrt::ModuleApi api{visionrt::MODULE_ABI_MAJOR, sizeof(visionrt::ModuleApi),
                                         visionrt::infer::registerTools};
    return &api;
}
