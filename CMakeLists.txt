cmake_minimum_required(VERSION 3.25)
project(VisionTools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 引入构建系统与编译器配置模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConfigCompiler)
include(ConfigDependencyDefaults)
include(ConfigYamlCpp)
include(ConfigOpenCV)
include(ConfigVisionRTToolSDK)
include(AddTest)
include(PrintConfig)

# 功能与测试构建开关
option(VISIONTOOLS_BUILD_TESTS "Build contract and unit tests" ON)
option(VISIONTOOLS_BUILD_INFER "Build InferRT GPU tools module" ON)

if(VISIONTOOLS_BUILD_INFER)
    include(ConfigCUDA)
    include(ConfigInferRT)
endif()

# 查找 Python 解释器用于代码与合同生成
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# 工具契约、常量与模块工厂代码自动生成 (YS-01 / YS-02)
file(GLOB_RECURSE VISIONTOOLS_TOOL_YAMLS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools/*.yaml")
set(VISIONTOOLS_MODULE_BINDINGS "${CMAKE_CURRENT_SOURCE_DIR}/tools/module_bindings.yaml")
set(VISIONTOOLS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VISIONTOOLS_GENERATED_DIR}")

set(VISIONTOOLS_GENERATOR_OUTPUTS
    "${VISIONTOOLS_GENERATED_DIR}/PortConstants.h"
    "${VISIONTOOLS_GENERATED_DIR}/visiontools_preprocess_factory.inc"
    "${VISIONTOOLS_GENERATED_DIR}/module_index.yaml"
)
if(VISIONTOOLS_BUILD_INFER)
    list(APPEND VISIONTOOLS_GENERATOR_OUTPUTS "${VISIONTOOLS_GENERATED_DIR}/visiontools_infer_factory.inc")
    set(VISIONTOOLS_GENERATOR_VARIANT "gpu")
else()
    set(VISIONTOOLS_GENERATOR_VARIANT "cpu")
endif()

get_target_property(_sdk_include_dirs VisionRT::ToolSDK INTERFACE_INCLUDE_DIRECTORIES)
list(GET _sdk_include_dirs 0 _primary_sdk_inc)

add_custom_command(
    OUTPUT ${VISIONTOOLS_GENERATOR_OUTPUTS}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            --tools-dir "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools"
            --bindings "${VISIONTOOLS_MODULE_BINDINGS}"
            --output-dir "${VISIONTOOLS_GENERATED_DIR}"
            --index-output "${VISIONTOOLS_GENERATED_DIR}/module_index.yaml"
            --variant "${VISIONTOOLS_GENERATOR_VARIANT}"
            --sdk-include-dir "${_primary_sdk_inc}"
    DEPENDS ${VISIONTOOLS_TOOL_YAMLS} ${VISIONTOOLS_MODULE_BINDINGS} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            "${_primary_sdk_inc}/visionrt/sdk/Value.h"
            "${_primary_sdk_inc}/visionrt/sdk/Schema.h"
    COMMENT "Generating tool contracts, constants, and module factories from YAML declarations..."
    VERBATIM
)

add_custom_target(generate_contracts DEPENDS ${VISIONTOOLS_GENERATOR_OUTPUTS})

# 导出 CMake 解析的运行依赖路径供打包脚本单源消费 (YS-01 / YS-03)
set(_clean_opencv_bin "${VISIONTOOLS_OPENCV_BIN}")
string(REPLACE "\\" "/" _clean_opencv_bin "${_clean_opencv_bin}")
set(_clean_inferrt_root "${INFERRT_ROOT}")
string(REPLACE "\\" "/" _clean_inferrt_root "${_clean_inferrt_root}")
set(_clean_cuda_root "${CUDAToolkit_ROOT}")
string(REPLACE "\\" "/" _clean_cuda_root "${_clean_cuda_root}")
set(_clean_trt_root "${TRT_ROOT}")
string(REPLACE "\\" "/" _clean_trt_root "${_clean_trt_root}")

set(_clean_yaml_cpp_bin "${YAML_CPP_DIR}/bin")
string(REPLACE "\\" "/" _clean_yaml_cpp_bin "${_clean_yaml_cpp_bin}")

set(_resolved_deps_content "# Auto-generated build dependencies manifest\n")
string(APPEND _resolved_deps_content "variant: \"${VISIONTOOLS_GENERATOR_VARIANT}\"\n")
string(APPEND _resolved_deps_content "config: \"$<CONFIG>\"\n")
string(APPEND _resolved_deps_content "modules:\n")
string(APPEND _resolved_deps_content "  visiontools_preprocess:\n")
string(APPEND _resolved_deps_content "    source: \"$<TARGET_FILE:visiontools_preprocess>\"\n")
if(VISIONTOOLS_BUILD_INFER)
    string(APPEND _resolved_deps_content "  visiontools_infer:\n")
    string(APPEND _resolved_deps_content "    source: \"$<TARGET_FILE:visiontools_infer>\"\n")
endif()

string(APPEND _resolved_deps_content "runtime_files:\n")
if(EXISTS "${_clean_opencv_bin}/opencv_world480.dll")
    string(APPEND _resolved_deps_content "  - source: \"${_clean_opencv_bin}/opencv_world480.dll\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/opencv_world480.dll\"\n")
endif()
if(EXISTS "${_clean_yaml_cpp_bin}/yaml-cpp.dll")
    string(APPEND _resolved_deps_content "  - source: \"${_clean_yaml_cpp_bin}/yaml-cpp.dll\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/yaml-cpp.dll\"\n")
elseif(EXISTS "D:/Software/anaconda3/Library/bin/yaml-cpp.dll")
    string(APPEND _resolved_deps_content "  - source: \"D:/Software/anaconda3/Library/bin/yaml-cpp.dll\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/yaml-cpp.dll\"\n")
endif()
if(VISIONTOOLS_BUILD_INFER)
    if(EXISTS "${_clean_inferrt_root}/bin/inferrt_core.dll")
        string(APPEND _resolved_deps_content "  - source: \"${_clean_inferrt_root}/bin/inferrt_core.dll\"\n")
        string(APPEND _resolved_deps_content "    destination: \"bin/inferrt_core.dll\"\n")
    endif()
    if(EXISTS "${_clean_inferrt_root}/bin/inferrt_model.dll")
        string(APPEND _resolved_deps_content "  - source: \"${_clean_inferrt_root}/bin/inferrt_model.dll\"\n")
        string(APPEND _resolved_deps_content "    destination: \"bin/inferrt_model.dll\"\n")
    endif()
    if(EXISTS "${_clean_cuda_root}/bin/cudart64_110.dll")
        string(APPEND _resolved_deps_content "  - source: \"${_clean_cuda_root}/bin/cudart64_110.dll\"\n")
        string(APPEND _resolved_deps_content "    destination: \"bin/cudart64_110.dll\"\n")
    endif()
    if(EXISTS "${_clean_trt_root}/bin/nvinfer_10.dll")
        string(APPEND _resolved_deps_content "  - source: \"${_clean_trt_root}/bin/nvinfer_10.dll\"\n")
        string(APPEND _resolved_deps_content "    destination: \"bin/nvinfer_10.dll\"\n")
    endif()
endif()

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/resolved_dependencies.yaml"
    CONTENT "${_resolved_deps_content}"
)

# 包含预处理领域模块
add_subdirectory(src/preprocess)
add_dependencies(visiontools_preprocess generate_contracts)

# 包含推理领域模块 (可选)
if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/infer")
    add_subdirectory(src/infer)
    add_dependencies(visiontools_infer generate_contracts)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/probes/infer")
        add_subdirectory(probes/infer)
    endif()
endif()

# 包含契约与单元测试
if(VISIONTOOLS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 打印构建配置摘要
visiontools_print_summary()

