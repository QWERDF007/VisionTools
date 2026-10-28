#pragma once

#include <cstdint>

namespace visiontools {

// The one table of business error domains. A domain is reserved once and never reused: the codes a
// tool publishes stay meaningful even if the tool moves between modules. Local numbering inside a
// domain belongs to that tool's own error header.
namespace domain {
inline constexpr std::int64_t kCommon           = 1;   // 1001–1999
inline constexpr std::int64_t kReadImage        = 101; // 101001–101999
inline constexpr std::int64_t kImageCrop        = 102; // 102001–102999
inline constexpr std::int64_t kImageGeometry    = 103; // 103001–103999
inline constexpr std::int64_t kMetrology        = 104; // 104001–104999
inline constexpr std::int64_t kInfer            = 301; // 301001–301999
inline constexpr std::int64_t kPythonExpression = 401; // 401001–401999
inline constexpr std::int64_t kPythonBranch     = 402; // 402001–402999
} // namespace domain

constexpr std::int64_t errorCode(std::int64_t domainId, int local) noexcept
{
    return domainId * 1000 + local;
}

} // namespace visiontools
