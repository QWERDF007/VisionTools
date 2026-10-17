#include <visionrt/sdk/Module.h>

#if __has_include("visiontools_preprocess_factory.inc")
#include "visiontools_preprocess_factory.inc"
#elif __has_include("generated/visiontools_preprocess_factory.inc")
#include "generated/visiontools_preprocess_factory.inc"
#else
#include "../../src/generated/visiontools_preprocess_factory.inc"
#endif

extern "C" VT_EXPORT const visionrt::ModuleApi* VT_CALL visiontoolsModuleApi() noexcept {
    return &visiontools::preprocess::s_generatedModuleApi;
}
