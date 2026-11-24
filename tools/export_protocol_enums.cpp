#include "EnumExport.h"

#define VRT_ENUM_FILE "visionrt/contracts/ProtocolEnums.def"
#define VRT_ENUM_HEADER "visionrt/contracts/ProtocolEnums.h"
#define VRT_ENUM_CATALOG protocol
#include "EnumCatalog.h"
#define VRT_ENUM_FILE "visionrt/contracts/TypeSpec.def"
#define VRT_ENUM_HEADER "visionrt/contracts/TypeSpec.h"
#define VRT_ENUM_CATALOG types
#include "EnumCatalog.h"
#define VRT_ENUM_FILE "visionrt/contracts/ROI.def"
#define VRT_ENUM_HEADER "visionrt/contracts/ROI.h"
#define VRT_ENUM_CATALOG roi
#include "EnumCatalog.h"
#define VRT_ENUM_FILE "common/errors/CommonToolErrors.def"
#define VRT_ENUM_HEADER "common/errors/ToolError.h"
#define VRT_ENUM_CATALOG errors
#include "EnumCatalog.h"

constexpr visionrt::codegen::WireName wireNames[]{
#define VRT_DECLARATION_NAME(type, member, wire) {#type, #member, wire},
#include <visionrt/contracts/DeclarationNames.def>
#undef VRT_DECLARATION_NAME
};

#ifdef _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
    return visionrt::codegen::run(argc, argv,
        {protocol::enums, types::enums, roi::enums, errors::enums}, wireNames);
}
