#include "visiontools_infer_factory.inc"

#include <visionrt/contracts/Module.h>

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    return &visionrt::infer::s_generatedModuleApi;
}
