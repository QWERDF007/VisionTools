#include "visiontools_preprocess_factory.inc"

#include <visionrt/contracts/Module.h>

extern "C" VT_EXPORT const visionrt::ModuleApi *VT_CALL visiontoolsModuleApi() noexcept
{
    return &visiontools::preprocess::s_generatedModuleApi;
}
