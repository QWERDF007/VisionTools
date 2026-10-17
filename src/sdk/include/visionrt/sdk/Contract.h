#pragma once

#include <visionrt/sdk/Value.h>
#include <string>
#include <vector>

namespace visionrt {

struct PortContract {
    std::string id;
    TypeSpec type;
    bool required = true;
};

struct ToolContract {
    std::string toolId;
    std::string registryName;
    std::vector<PortContract> initParams;
    std::vector<PortContract> inputs;
    std::vector<PortContract> outputs;
};

} // namespace visionrt
