include_guard(GLOBAL)

if(TARGET yaml-cpp::yaml-cpp OR TARGET yaml-cpp)
    if(NOT TARGET yaml-cpp AND TARGET yaml-cpp::yaml-cpp)
        add_library(yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif()
    return()
endif()

# yaml-cpp 唯一来源：3rdparty/yaml-cpp 子模块（静态 /MD 构建，与 VisionRTToolSDK 内嵌版本同源）
set(_vt_yaml_cpp_dir "${CMAKE_SOURCE_DIR}/3rdparty/yaml-cpp")
if(NOT EXISTS "${_vt_yaml_cpp_dir}/CMakeLists.txt")
    message(FATAL_ERROR "yaml-cpp submodule missing at '${_vt_yaml_cpp_dir}'. Initialize it with: git submodule update --init --recursive")
endif()

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static yaml-cpp" FORCE)
set(YAML_MSVC_SHARED_RT ON CACHE BOOL "MSVC shared runtime (/MD)" FORCE)

add_subdirectory("${_vt_yaml_cpp_dir}" "${CMAKE_BINARY_DIR}/3rdparty/yaml-cpp")
