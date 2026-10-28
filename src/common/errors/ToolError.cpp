#include "common/errors/ToolError.h"

namespace visiontools {

std::string_view defaultMessage(std::int64_t code) noexcept
{
    switch (static_cast<CommonToolError>(code))
    {
    case CommonToolError::InputMissing:
        return "输入缺失";
    case CommonToolError::InputTypeMismatch:
        return "输入类型不匹配";
    case CommonToolError::InputOutOfRange:
        return "输入超出允许范围";
    case CommonToolError::ResourceUnavailable:
        return "所需资源不可用";
    case CommonToolError::OperatorFailed:
        return "算子执行失败";
    case CommonToolError::OutputContractViolation:
        return "输出端口不符合契约";
    case CommonToolError::UnsupportedImage:
        return "图像格式不受支持";
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
    return {code, issue.describe(), {}};
}

} // namespace visiontools
