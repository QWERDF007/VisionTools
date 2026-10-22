#pragma once

#include <visionrt/contracts/Tool.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace visiontools::preprocess {

// Small shared value boundary for direct consumers. The host fills declaration
// defaults before execute; operators require the values they actually use.
inline const visionrt::PortValue &requireInput(const visionrt::Values &inputs, std::string_view id)
{
    const auto found = inputs.find(id);
    if (found == inputs.end() || visionrt::isMissing(found->second))
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 缺失");
    return found->second;
}

template<class T>
const T &requireTypedInput(const visionrt::Values &inputs, std::string_view id, const visionrt::TypeSpec &type)
{
    const auto &value = requireInput(inputs, id);
    try
    {
        visionrt::validateValue(type, value);
    }
    catch (const visionrt::ContractError &error)
    {
        std::string expected = visionrt::typeName(type.type);
        if (type.dtype)
            expected += "(dtype=" + std::string(visionrt::dtypeName(*type.dtype))
                      + ", rank=" + std::to_string(type.rank) + ")";
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须是 " + expected + " 的有效值："
                                      + error.what());
    }
    return std::get<T>(value);
}

inline std::int64_t requireInt(const visionrt::Values &inputs, std::string_view id)
{
    const auto &value = requireInput(inputs, id);
    if (const auto *number = std::get_if<std::int64_t>(&value))
        return *number;
    throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须是整数");
}

inline double requireFloat(const visionrt::Values &inputs, std::string_view id)
{
    const auto &value = requireInput(inputs, id);
    if (const auto *number = std::get_if<double>(&value))
        return *number;
    throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须是浮点数（不接受整数）");
}

inline std::string requireString(const visionrt::Values &inputs, std::string_view id)
{
    const auto &value = requireInput(inputs, id);
    if (const auto *text = std::get_if<std::string>(&value))
    {
        visionrt::validateUtf8(*text);
        return *text;
    }
    throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须是字符串");
}

inline std::int64_t requireIntInRange(const visionrt::Values &inputs, std::string_view id,
                                      std::int64_t minimum, std::int64_t maximum)
{
    const auto value = requireInt(inputs, id);
    if (value < minimum || value > maximum)
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 的值 " + std::to_string(value)
                                      + " 超出允许范围 [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    return value;
}

inline int requireInt32(const visionrt::Values &inputs, std::string_view id, std::int64_t minimum, std::int64_t maximum)
{
    return static_cast<int>(requireIntInRange(inputs, id, minimum, maximum));
}

inline double requireFloatInRange(const visionrt::Values &inputs, std::string_view id, double minimum, double maximum)
{
    const auto value = requireFloat(inputs, id);
    if (!std::isfinite(value))
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 必须为有限数值");
    if (value < minimum || value > maximum)
        throw visionrt::ContractError("输入端口 '" + std::string(id) + "' 的值 " + std::to_string(value)
                                      + " 超出允许范围 [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    return value;
}

} // namespace visiontools::preprocess
