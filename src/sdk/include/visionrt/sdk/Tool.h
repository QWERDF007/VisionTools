#pragma once

#include <visionrt/sdk/Value.h>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace visionrt {

class PathResolver {
public:
    PathResolver() = default;
    explicit PathResolver(std::filesystem::path projectDirectory)
        : directory_(std::move(projectDirectory)) {}
    std::filesystem::path resolve(const std::string& utf8) const;
private:
    std::filesystem::path directory_;
};

using Values = std::map<std::string, PortValue, std::less<>>;

struct ToolStatus {
    std::int64_t code = 0;
    std::string message;
    explicit operator bool() const noexcept { return code == 0; }
};

struct ToolResult {
    ToolStatus status;
    Values outputs;
};

struct ToolServices {
    PathResolver paths;
};

class IToolInstance {
public:
    virtual ~IToolInstance() = default;
    virtual ToolStatus initialize(const Values& initParams) = 0;
    virtual ToolResult execute(const Values& inputs) = 0;
    virtual ToolStatus release() = 0;
};

using ToolFactory = std::function<std::unique_ptr<IToolInstance>(const ToolServices&)>;

} // namespace visionrt
