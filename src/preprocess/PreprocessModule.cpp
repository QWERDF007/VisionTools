#include <visionrt/sdk/Module.h>
#include "visiontools_preprocess_factory.inc"

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visiontools::preprocess::s_generatedModuleApi;
}
