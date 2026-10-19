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
include(ConfigYamlCpp)
add_subdirectory(3rdparty/VisionRTSDK)
include(ConfigPython)
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
file(GLOB_RECURSE VISIONTOOLS_TOOL_YAMLS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools/*.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools/*.yml"
)
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

set(VISIONTOOLS_SDK_INCLUDE_DIR "${VISIONRT_TOOL_SDK_INCLUDE_DIR}")

add_custom_command(
    OUTPUT ${VISIONTOOLS_GENERATOR_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${VISIONTOOLS_OPENCV_BIN};$ENV{PATH}"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            --tools-dir "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools"
            --bindings "${VISIONTOOLS_MODULE_BINDINGS}"
            --output-dir "${VISIONTOOLS_GENERATED_DIR}"
            --index-output "${VISIONTOOLS_GENERATED_DIR}/module_index.yaml"
            --variant "${VISIONTOOLS_GENERATOR_VARIANT}"
            --sdk-include-dir "${VISIONTOOLS_SDK_INCLUDE_DIR}"
            --validator "$<TARGET_FILE:VisionRT::visionrt_validate_tool>"
    DEPENDS ${VISIONTOOLS_TOOL_YAMLS} ${VISIONTOOLS_MODULE_BINDINGS} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            "${VISIONTOOLS_SDK_INCLUDE_DIR}/visionrt/sdk/Value.h"
            "${VISIONTOOLS_SDK_INCLUDE_DIR}/visionrt/sdk/Schema.h"
            VisionRT::visionrt_validate_tool
    COMMENT "Generating tool contracts, constants, and module factories from YAML declarations..."
    VERBATIM
)

add_custom_target(generate_contracts DEPENDS ${VISIONTOOLS_GENERATOR_OUTPUTS})

# 导出 CMake 解析的运行依赖路径供打包脚本单源消费 (YS-01 / YS-03 / VT-S01)
set(_clean_inferrt_root "${INFERRT_ROOT}")
string(REPLACE "\\" "/" _clean_inferrt_root "${_clean_inferrt_root}")
set(_clean_cuda_root "${CUDAToolkit_ROOT}")
string(REPLACE "\\" "/" _clean_cuda_root "${_clean_cuda_root}")
set(_clean_trt_root "${TRT_ROOT}")
string(REPLACE "\\" "/" _clean_trt_root "${_clean_trt_root}")

set(_resolved_deps_content "# Auto-generated build dependencies manifest\n")
string(APPEND _resolved_deps_content "variant: \"${VISIONTOOLS_GENERATOR_VARIANT}\"\n")
string(APPEND _resolved_deps_content "config: \"$<CONFIG>\"\n")
string(APPEND _resolved_deps_content "validator: \"$<TARGET_FILE:VisionRT::visionrt_validate_tool>\"\n")
string(APPEND _resolved_deps_content "modules:\n")
string(APPEND _resolved_deps_content "  visiontools_preprocess:\n")
string(APPEND _resolved_deps_content "    variant: \"cpu\"\n")
string(APPEND _resolved_deps_content "    source: \"$<TARGET_FILE:visiontools_preprocess>\"\n")
if(VISIONTOOLS_BUILD_INFER)
    string(APPEND _resolved_deps_content "  visiontools_infer:\n")
    string(APPEND _resolved_deps_content "    variant: \"gpu\"\n")
    string(APPEND _resolved_deps_content "    source: \"$<TARGET_FILE:visiontools_infer>\"\n")
endif()

string(APPEND _resolved_deps_content "runtime_files:\n")

# OpenCV runtime dependency (Release / Debug sensitive)
get_filename_component(_opencv_rel_name "${VISIONTOOLS_OPENCV_RELEASE_DLL}" NAME)
if(VISIONTOOLS_OPENCV_DEBUG_DLL)
    get_filename_component(_opencv_dbg_name "${VISIONTOOLS_OPENCV_DEBUG_DLL}" NAME)
    string(APPEND _resolved_deps_content "  - source: \"$<IF:$<CONFIG:Debug>,${VISIONTOOLS_OPENCV_DEBUG_DLL},${VISIONTOOLS_OPENCV_RELEASE_DLL}>\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/$<IF:$<CONFIG:Debug>,${_opencv_dbg_name},${_opencv_rel_name}>\"\n")
else()
    string(APPEND _resolved_deps_content "  - source: \"${VISIONTOOLS_OPENCV_RELEASE_DLL}\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/${_opencv_rel_name}\"\n")
endif()

if(VISIONTOOLS_BUILD_INFER)
    if(NOT EXISTS "${_clean_inferrt_root}/bin/inferrt_core.dll")
        message(FATAL_ERROR "Required InferRT runtime DLL not found: ${_clean_inferrt_root}/bin/inferrt_core.dll")
    endif()
    if(NOT EXISTS "${_clean_inferrt_root}/bin/inferrt_model.dll")
        message(FATAL_ERROR "Required InferRT runtime DLL not found: ${_clean_inferrt_root}/bin/inferrt_model.dll")
    endif()
    string(APPEND _resolved_deps_content "  - source: \"${_clean_inferrt_root}/bin/inferrt_core.dll\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/inferrt_core.dll\"\n")
    string(APPEND _resolved_deps_content "  - source: \"${_clean_inferrt_root}/bin/inferrt_model.dll\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/inferrt_model.dll\"\n")

    file(GLOB _cuda_runtime_dlls "${_clean_cuda_root}/bin/cudart64_*.dll")
    if(NOT _cuda_runtime_dlls)
        message(FATAL_ERROR "Required CUDA runtime DLL (cudart64_*.dll) not found in ${_clean_cuda_root}/bin")
    endif()
    list(GET _cuda_runtime_dlls 0 _cuda_rt_dll)
    get_filename_component(_cuda_rt_name "${_cuda_rt_dll}" NAME)
    string(APPEND _resolved_deps_content "  - source: \"${_cuda_rt_dll}\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/${_cuda_rt_name}\"\n")

    file(GLOB _trt_candidates "${_clean_trt_root}/bin/nvinfer_*.dll" "${_clean_trt_root}/lib/nvinfer_*.dll")
    set(_trt_core_dll "")
    foreach(_tc IN LISTS _trt_candidates)
        get_filename_component(_tc_name "${_tc}" NAME)
        string(TOLOWER "${_tc_name}" _tc_lower)
        if(NOT _tc_lower MATCHES "plugin" AND NOT _tc_lower MATCHES "dispatch" AND NOT _tc_lower MATCHES "lean")
            set(_trt_core_dll "${_tc}")
            break()
        endif()
    endforeach()
    if(NOT _trt_core_dll)
        message(FATAL_ERROR "Required TensorRT core runtime DLL (nvinfer_*.dll) not found in ${_clean_trt_root}")
    endif()
    get_filename_component(_trt_core_name "${_trt_core_dll}" NAME)
    string(APPEND _resolved_deps_content "  - source: \"${_trt_core_dll}\"\n")
    string(APPEND _resolved_deps_content "    destination: \"bin/${_trt_core_name}\"\n")
endif()

file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/resolved_dependencies.yaml"
    CONTENT "${_resolved_deps_content}"
)

# 导出开发环境 PATH 前缀（全部来自依赖解析结果），供 tools\build.cmd 打包等阶段复用，避免机器硬编码
set(_devenv_dirs "${VISIONTOOLS_OPENCV_BIN}")
if(VISIONTOOLS_BUILD_INFER)
    list(APPEND _devenv_dirs "${INFERRT_BIN_DIR}" "${VISIONTOOLS_CUDA_BIN}" "${TRT_ROOT}/bin")
endif()
list(REMOVE_DUPLICATES _devenv_dirs)
list(REMOVE_ITEM _devenv_dirs "")
string(JOIN ";" _devenv_path_line ${_devenv_dirs})
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/devenv_paths.txt" CONTENT "${_devenv_path_line}\n")

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

