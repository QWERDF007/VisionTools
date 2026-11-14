#include "ProductPaths.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#endif

#include <vector>

namespace visiontools::common {
namespace {

std::filesystem::path executableFile()
{
#if defined(_WIN32)
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;)
    {
        const auto written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (written == 0)
            throw ProductDirectoryError("无法确定程序所在目录（Win32 错误："
                                        + std::to_string(GetLastError()) + "）");
        if (written < buffer.size())
        {
            buffer.resize(written);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
        throw ProductDirectoryError("无法确定程序所在目录");
    std::error_code error;
    const auto      resolved = std::filesystem::weakly_canonical(buffer.data(), error);
    if (error)
        throw ProductDirectoryError("无法确定程序所在目录：" + error.message());
    return resolved;
#else
    std::error_code error;
    const auto      self = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error || self.empty())
        throw ProductDirectoryError("无法确定程序所在目录：" + error.message());
    return self;
#endif
}

} // namespace

std::filesystem::path productDirectory()
{
    return executableFile().parent_path();
}

std::filesystem::path resolveResourcePath(std::string_view utf8)
{
    const auto path = std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
    if (path.is_absolute())
        return path.lexically_normal();
    return (productDirectory() / path).lexically_normal();
}

} // namespace visiontools::common
