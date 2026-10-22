cmake_minimum_required(VERSION 3.25)
project(VisionTools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 引入构建系统与编译器配置模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConfigCompiler)
include(ConfigDependencyDefaults)
include(ConfigOpenCV)
include(ConfigYamlCpp)
add_subdirectory(3rdparty/VisionRTContracts)
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
set(VISIONTOOLS_DISTRIBUTION "${PROJECT_NAME}-${PROJECT_VERSION}" CACHE STRING "Distribution configuration name")
set(VISIONTOOLS_BUILT_MODULES visiontools_preprocess)
set(VISIONTOOLS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VISIONTOOLS_GENERATED_DIR}")

set(VISIONTOOLS_GENERATOR_OUTPUTS
    "${VISIONTOOLS_GENERATED_DIR}/PortConstants.h"
    "${VISIONTOOLS_GENERATED_DIR}/visiontools_preprocess_factory.inc"
    "${VISIONTOOLS_GENERATED_DIR}/module_index.yaml"
)
if(VISIONTOOLS_BUILD_INFER)
    list(APPEND VISIONTOOLS_GENERATOR_OUTPUTS "${VISIONTOOLS_GENERATED_DIR}/visiontools_infer_factory.inc")
    list(APPEND VISIONTOOLS_BUILT_MODULES visiontools_infer)
endif()

if(WIN32)
    set(_vt_path_separator ";")
else()
    set(_vt_path_separator ":")
endif()
if(VISIONTOOLS_OPENCV_BIN)
    set(_vt_path_prefix "${VISIONTOOLS_OPENCV_BIN}${_vt_path_separator}")
else()
    set(_vt_path_prefix "")
endif()

add_custom_command(
    OUTPUT ${VISIONTOOLS_GENERATOR_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${_vt_path_prefix}$ENV{PATH}"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            --tools-dir "${CMAKE_CURRENT_SOURCE_DIR}/assets/tools"
            --bindings "${VISIONTOOLS_MODULE_BINDINGS}"
            --output-dir "${VISIONTOOLS_GENERATED_DIR}"
            --index-output "${VISIONTOOLS_GENERATED_DIR}/module_index.yaml"
            --variant "${VISIONTOOLS_DISTRIBUTION}"
            --modules ${VISIONTOOLS_BUILT_MODULES}
            --validator "$<TARGET_FILE:VisionRT::visionrt_validate_tool>"
    DEPENDS ${VISIONTOOLS_TOOL_YAMLS} ${VISIONTOOLS_MODULE_BINDINGS} "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_tool_contracts.py"
            VisionRT::visionrt_validate_tool
    COMMENT "Generating tool contracts, constants, and module factories from normalized declarations..."
    VERBATIM
)

add_custom_target(generate_contracts DEPENDS ${VISIONTOOLS_GENERATOR_OUTPUTS})

set(_resolved_deps_content "variant: \"${VISIONTOOLS_DISTRIBUTION}\"\n")
string(APPEND _resolved_deps_content "config: \"$<CONFIG>\"\n")
string(APPEND _resolved_deps_content "validator: \"$<TARGET_FILE:VisionRT::visionrt_validate_tool>\"\n")
string(APPEND _resolved_deps_content "runtime_install:\n  cmake: \"${CMAKE_COMMAND}\"\n  component: VisionToolsRuntime\n")
string(APPEND _resolved_deps_content "runtime_files: []\nmodules:\n")
foreach(_module IN LISTS VISIONTOOLS_BUILT_MODULES)
    string(APPEND _resolved_deps_content "  ${_module}:\n    source: \"$<TARGET_FILE:${_module}>\"\n")
endforeach()
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

install(TARGETS ${VISIONTOOLS_BUILT_MODULES}
    RUNTIME_DEPENDENCIES
        DIRECTORIES ${_devenv_dirs}
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
        POST_EXCLUDE_REGEXES ".*[Ss][Yy][Ss][Tt][Ee][Mm]32/.*"
    RUNTIME DESTINATION bin COMPONENT VisionToolsRuntime
    LIBRARY DESTINATION bin COMPONENT VisionToolsRuntime
)
if(VISIONTOOLS_BUILD_INFER AND WIN32)
    install(DIRECTORY "${TRT_ROOT}/bin/"
        DESTINATION bin COMPONENT VisionToolsRuntime
        FILES_MATCHING PATTERN "nvinfer_builder_resource*.dll"
    )
endif()

# 包含契约与单元测试
if(VISIONTOOLS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 打印构建配置摘要
visiontools_print_summary()

