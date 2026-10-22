#pragma once

#include <visionrt/contracts/Registry.h>

#include <memory>

namespace visiontools {

template<class Tool>
visionrt::IToolInstance *VT_CALL createTool(const visionrt::ToolServices *services,
                                            visionrt::ToolStatus         *status) noexcept
{
    if (!services)
    {
        if (status)
            *status = {-3, "工具服务不能为空"};
        return nullptr;
    }
    try
    {
        auto instance = std::make_unique<Tool>(*services);
        if (status)
            *status = {};
        return instance.release();
    }
    catch (const std::exception &error)
    {
        if (status)
            *status = {-3, "工具实例创建失败", error.what()};
    }
    catch (...)
    {
        if (status)
            *status = {-3, "工具实例创建发生未知异常"};
    }
    return nullptr;
}

inline void VT_CALL destroyTool(visionrt::IToolInstance *instance) noexcept
{
    delete instance;
}

}
