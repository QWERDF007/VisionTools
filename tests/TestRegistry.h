#pragma once

#include <visionrt/contracts/Registry.h>

#include <map>
#include <string>

class TestRegistry final : public visionrt::IToolRegistry
{
public:
    visionrt::ToolStatus add(const visionrt::ToolFactoryDescriptor &descriptor) noexcept override
    {
        try
        {
            if (!descriptor.tool_id || !*descriptor.tool_id || !descriptor.create || !descriptor.destroy)
                return {-3, "无效工具工厂"};
            auto [position, inserted] = factories.emplace(descriptor.tool_id, descriptor);
            if (!inserted)
                return {-3, "工具工厂 ID 重复"};
            position->second.tool_id = position->first.c_str();
            return {};
        }
        catch (const std::exception &error)
        {
            return {-3, "工具工厂注册失败", error.what()};
        }
    }

    std::map<std::string, visionrt::ToolFactoryDescriptor, std::less<>> factories;
};
