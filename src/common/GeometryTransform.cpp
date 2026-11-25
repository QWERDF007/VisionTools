#include "GeometryTransform.h"

#include "common/errors/GeometryErrors.h"
#include "common/errors/ToolError.h"

namespace visiontools::geometry {

std::vector<Alignment> alignmentsFromValue(const visionrt::PortValue &value, std::size_t imageCount)
{
    if (visionrt::isMissing(value)) fail(CommonToolError::InputMissing, "对齐数据");
    const auto *array = std::get_if<visionrt::Array>(&value);
    if (!array || array->type() != visionrt::kAlignmentType)
        fail(CommonToolError::InputTypeMismatch, "对齐数据必须是二维浮点数数组");
    try { return visionrt::alignmentsFromArray(*array, imageCount); }
    catch (const visionrt::ContractError &error) { fail(GeometryError::AlignmentInvalid, error.what()); }
}

}
