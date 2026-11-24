#include "common/errors/ToolError.h"

#include <iostream>
#include <stdexcept>

int main()
try
{
    const std::pair<std::int64_t, std::string_view> expected[]{
        {1001, "输入缺失"}, {1002, "输入类型不匹配"}, {1003, "输入超出允许范围"},
        {1004, "所需资源不可用"}, {1005, "算子执行失败"}, {1006, "工具输出不符合声明"},
        {1007, "图像格式不受支持"},
        {1008, "路径为空"}, {1009, "文件不存在"}, {1010, "文件读取失败"}};
    for (const auto &[code, message] : expected)
    {
        const auto result = visiontools::status(visiontools::ToolFailure{code, {}, {}});
        if (result.code != code || result.message != message)
            throw std::runtime_error("common error message mismatch: " + std::to_string(code));
    }
    if (visiontools::defaultMessage(999999) != "工具执行失败")
        throw std::runtime_error("unknown common error fallback changed");
    std::cout << "Common errors contract passed\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
