#pragma once

#include "common/errors/ToolErrorDomains.h"

#include <visionrt/contracts/Issue.h>
#include <visionrt/contracts/Tool.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace visiontools {

// Tool business codes are positive and fixed: domain * 1000 + local. A domain is owned by one tool
// group, so a code keeps its meaning when a tool moves between modules and two tools can never
// report the same number for different failures. Host-side negative codes never appear here.

// Codes for failures that are not specific to one algorithm. A tool with its own failure classes
// declares its own enum in its own header and only falls back to these for generic ports.
enum class CommonToolError : std::int64_t
{
    InputMissing            = errorCode(domain::kCommon, 1),
    InputTypeMismatch       = errorCode(domain::kCommon, 2),
    InputOutOfRange         = errorCode(domain::kCommon, 3),
    ResourceUnavailable     = errorCode(domain::kCommon, 4),
    OperatorFailed          = errorCode(domain::kCommon, 5),
    OutputContractViolation = errorCode(domain::kCommon, 6),
    UnsupportedImage        = errorCode(domain::kCommon, 7)
};

// A business failure as the tool decided it: a precise code, a message a user can act on, and an
// optional technical detail that stays out of the message.
struct ToolFailure
{
    std::int64_t code = 0;
    std::string  message;
    std::string  detail;
};

// The catalog text for any registered code; an unregistered number still gets a usable sentence.
std::string_view defaultMessage(std::int64_t code) noexcept;

inline std::string_view message(CommonToolError code) noexcept
{
    return defaultMessage(static_cast<std::int64_t>(code));
}

// A message that reads as one sentence: the catalog text, then the concrete context.
std::string sentence(std::string_view text, std::string_view context);

visionrt::ToolStatus status(const ToolFailure &failure);

template<class ErrorCode>
    requires std::is_enum_v<ErrorCode>
inline visionrt::ToolStatus status(ErrorCode code, std::string_view context = {}, std::string detail = {})
{
    return {static_cast<std::int64_t>(code), sentence(message(code), context), std::move(detail)};
}

// Reports an expected failure from inside an algorithm. It is thrown only to leave nested helpers
// and is caught by that tool's own execute(), which turns it into a ToolResult status; it never
// crosses the IToolInstance boundary.
[[noreturn]] void raise(std::int64_t code, std::string message, std::string detail = {});

// The same for one tool's own catalog: the message comes from that tool's enum, the context says
// which image, ROI or parameter failed.
template<class ErrorCode>
    requires std::is_enum_v<ErrorCode>
[[noreturn]] inline void fail(ErrorCode code, std::string_view context = {}, std::string detail = {})
{
    raise(static_cast<std::int64_t>(code), sentence(message(code), context), std::move(detail));
}

// The mechanical port-read issues every tool can meet, mapped onto that tool's own codes.
visionrt::ToolStatus inputIssueStatus(const visionrt::PortIssue &issue, std::int64_t missingCode,
                                      std::int64_t typeCode);

// The one shape every tool uses to answer a failure: the declared ports as Missing plus the status.
template<class ErrorCode>
    requires std::is_enum_v<ErrorCode>
inline visionrt::ToolResult failure(visionrt::Values outputs, ErrorCode code, std::string_view context = {},
                                    std::string detail = {})
{
    return visionrt::toolResult(std::move(outputs),
                                {static_cast<std::int64_t>(code), sentence(message(code), context), std::move(detail)});
}

inline visionrt::ToolResult failureResult(visionrt::Values outputs, const ToolFailure &error)
{
    return visionrt::toolResult(std::move(outputs), status(error));
}

} // namespace visiontools
