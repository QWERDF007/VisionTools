include_guard(GLOBAL)

# 编译器与运行库选项配置
if(MSVC)
    # /utf-8: 将源文件和执行字符集设置为 UTF-8
    # /W4: 开启详细警告
    # /bigobj: 增加目标文件中的节数
    # /permissive-: 启用严格标准一致性模式
    # /Zc:__cplusplus: 报告正确的 C++ 标准版本宏
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus /bigobj)
    # MSVC 统一使用动态运行库，避免跨库链接冲突
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
