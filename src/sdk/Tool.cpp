#include <visionrt/sdk/Tool.h>
#include <stdexcept>

namespace visionrt {

std::filesystem::path PathResolver::resolve(const std::string& utf8) const {
    const auto path = std::filesystem::path(std::u8string_view(
        reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
    if (path.is_absolute()) return path.lexically_normal();
    if (directory_.empty())
        throw std::runtime_error("相对资源路径需要先设置工程目录");
    return (directory_ / path).lexically_normal();
}

} // namespace visionrt
