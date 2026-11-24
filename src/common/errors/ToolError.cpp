#include "common/errors/ToolError.h"

namespace visiontools {

std::string_view defaultMessage(std::int64_t code) noexcept
{
    switch (static_cast<CommonToolError>(code))
    {
#define VRT_ENUM_BEGIN(name, underlying, ...)
#define VRT_ENUM_VALUE(name, value, python, message) case CommonToolError::name: return message;
#define VRT_ENUM_END(name)
#include "common/errors/CommonToolErrors.def"
#undef VRT_ENUM_BEGIN
#undef VRT_ENUM_VALUE
#undef VRT_ENUM_END
    }
    return "工具执行失败";
}

std::string sentence(std::string_view text, std::string_view context)
{
    std::string message(text);
    if (!context.empty())
    {
        message += "：";
        message += context;
    }
    return message;
}

visionrt::ToolStatus status(const ToolFailure &failure)
{
    return {failure.code,
            failure.message.empty() ? std::string(defaultMessage(failure.code)) : failure.message,
            failure.detail};
}

void raise(std::int64_t code, std::string message, std::string detail)
{
    throw ToolFailure{code, message.empty() ? std::string(defaultMessage(code)) : std::move(message),
                      std::move(detail)};
}

visionrt::ToolStatus inputIssueStatus(const visionrt::PortIssue &issue, std::int64_t missingCode,
                                      std::int64_t typeCode)
{
    const auto code = issue.kind == visionrt::PortIssueKind::Missing ? missingCode : typeCode;
    // The SDK issue already names the port and the reason; repeating it in detail adds nothing.
    return {code, issue.field + (issue.kind == visionrt::PortIssueKind::Missing ? "缺失" : "类型不匹配")
                      + (issue.text.empty() ? "" : "：" + issue.text), {}};
}

} // namespace visiontools
