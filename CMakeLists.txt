cmake_minimum_required(VERSION 3.25)
project(VisionTools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 引入构建系统与编译器配置模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConfigCompiler)
include(ConfigDependencyDefaults)
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

# 工具契约、常量与模块工厂代码自动生成 (VS-02)
file(GLOB_RECURSE VISIONTOOLS_TOOL_YAMLS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools/*.yaml")
set(VISIONTOOLS_MODULE_BINDINGS "${CMAKE_CURRENT_SOURCE_DIR}/tools/module_bindings.yaml")
set(VISIONTOOLS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VISIONTOOLS_GENERATED_DIR}")

set(VISIONTOOLS_GENERATOR_OUTPUTS
    "${VISIONTOOLS_GENERATED_DIR}/PortConstants.h"
    "${VISIONTOOLS_GENERATED_DIR}/visiontools_preprocess_factory.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/module_index.json"
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
            --index-output "${CMAKE_CURRENT_BINARY_DIR}/module_index.json"
            --variant "${VISIONTOOLS_GENERATOR_VARIANT}"
            --sdk-include-dir "${_primary_sdk_inc}"
    DEPENDS ${VISIONTOOLS_TOOL_YAMLS} ${VISIONTOOLS_MODULE_BINDINGS} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            "${_primary_sdk_inc}/visionrt/sdk/Value.h"
            "${_primary_sdk_inc}/visionrt/sdk/Schema.h"
    COMMENT "Generating tool contracts, constants, and module factories from YAML declarations..."
    VERBATIM
)

add_custom_target(generate_contracts DEPENDS ${VISIONTOOLS_GENERATOR_OUTPUTS})

# 导出 CMake 解析的运行依赖路径供打包脚本单源消费 (VS-06 / F05)
set(_build_infer_json "false")
if(VISIONTOOLS_BUILD_INFER)
    set(_build_infer_json "true")
endif()
set(_clean_opencv_dir "${OpenCV_DIR}")
string(REPLACE "\\" "/" _clean_opencv_dir "${_clean_opencv_dir}")
set(_clean_sdk_dir "${VisionRTToolSDK_DIR}")
string(REPLACE "\\" "/" _clean_sdk_dir "${_clean_sdk_dir}")
set(_clean_inferrt_root "${INFERRT_ROOT}")
string(REPLACE "\\" "/" _clean_inferrt_root "${_clean_inferrt_root}")
set(_clean_cuda_root "${CUDAToolkit_ROOT}")
string(REPLACE "\\" "/" _clean_cuda_root "${_clean_cuda_root}")
set(_clean_trt_root "${TRT_ROOT}")
string(REPLACE "\\" "/" _clean_trt_root "${_clean_trt_root}")

set(_resolved_deps_content "{\n")
string(APPEND _resolved_deps_content "  \"version\": \"0.1.0\",\n")
string(APPEND _resolved_deps_content "  \"build_infer\": ${_build_infer_json},\n")
string(APPEND _resolved_deps_content "  \"generator_variant\": \"${VISIONTOOLS_GENERATOR_VARIANT}\",\n")
string(APPEND _resolved_deps_content "  \"config\": \"$<CONFIG>\",\n")
string(APPEND _resolved_deps_content "  \"opencv_dir\": \"${_clean_opencv_dir}\",\n")
string(APPEND _resolved_deps_content "  \"visionrt_sdk_dir\": \"${_clean_sdk_dir}\"")
if(VISIONTOOLS_BUILD_INFER)
    string(APPEND _resolved_deps_content ",\n  \"inferrt_root\": \"${_clean_inferrt_root}\",\n")
    string(APPEND _resolved_deps_content "  \"cuda_root\": \"${_clean_cuda_root}\",\n")
    string(APPEND _resolved_deps_content "  \"tensorrt_root\": \"${_clean_trt_root}\"\n")
else()
    string(APPEND _resolved_deps_content "\n")
endif()
string(APPEND _resolved_deps_content "}\n")

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/resolved_dependencies.json"
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

