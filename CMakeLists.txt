cmake_minimum_required(VERSION 3.25)
project(VisionTools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConfigCompiler)
include(ConfigDependencyDefaults)
include(ConfigOpenCV)
add_subdirectory(3rdparty/VisionRTContracts)
include(AddTest)
include(PrintConfig)

option(VISIONTOOLS_BUILD_TESTS "Build contract and unit tests" ON)
option(VISIONTOOLS_BUILD_INFER "Build InferRT GPU tools module" ON)
option(VISIONTOOLS_BUILD_PYTHON "Enable Python tool and packaging tests" ON)
set(VISIONTOOLS_VALIDATOR "$ENV{VISIONTOOLS_VALIDATOR}" CACHE FILEPATH "Host tool declaration validator")

if(VISIONTOOLS_BUILD_INFER)
    include(ConfigCUDA)
    include(ConfigInferRT)
endif()
if(VISIONTOOLS_BUILD_PYTHON)
    include(ConfigPython)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(VISIONTOOLS_DISTRIBUTION "${PROJECT_NAME}-${PROJECT_VERSION}" CACHE STRING "Distribution configuration name")
set(VISIONTOOLS_BUILT_MODULES visiontools_preprocess visiontools_metrology)
if(VISIONTOOLS_BUILD_INFER)
    list(APPEND VISIONTOOLS_BUILT_MODULES visiontools_infer)
endif()
set(_resolved_deps_content "{\n  \"variant\": \"${VISIONTOOLS_DISTRIBUTION}\",\n  \"config\": \"$<CONFIG>\",\n  \"validator\": \"${VISIONTOOLS_VALIDATOR}\",\n")
string(APPEND _resolved_deps_content "  \"runtime_install\": {\"cmake\": \"${CMAKE_COMMAND}\", \"component\": \"VisionToolsRuntime\"},\n  \"runtime_files\": [],\n  \"modules\": {")
set(_module_separator "")
foreach(_module IN LISTS VISIONTOOLS_BUILT_MODULES)
    string(APPEND _resolved_deps_content "${_module_separator}\n    \"${_module}\": {\"source\": \"$<TARGET_FILE:${_module}>\"}")
    set(_module_separator ",")
endforeach()
string(APPEND _resolved_deps_content "\n  }\n}\n")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/resolved_dependencies.json" CONTENT "${_resolved_deps_content}")

# 生成工具包构建元数据 build_info.json (Ticket AT-01)
find_package(Git QUIET)
set(VISIONTOOLS_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _vt_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_vt_commit)
        set(VISIONTOOLS_COMMIT "${_vt_commit}")
    endif()
endif()
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _vt_branch
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_vt_branch)
        set(VISIONTOOLS_BRANCH "${_vt_branch}")
    endif()
endif()
set(_devenv_dirs "${VISIONTOOLS_OPENCV_BIN}")
if(VISIONTOOLS_BUILD_INFER)
    list(APPEND _devenv_dirs "${INFERRT_BIN_DIR}" "${VISIONTOOLS_CUDA_BIN}" "${TRT_ROOT}/bin")
endif()
list(REMOVE_DUPLICATES _devenv_dirs)
list(REMOVE_ITEM _devenv_dirs "")
string(JOIN ";" _devenv_path_line ${_devenv_dirs})
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/devenv_paths.txt" CONTENT "${_devenv_path_line}\n")

# 包含公共几何与预处理领域模块
add_subdirectory(src/common)
add_subdirectory(src/preprocess)
add_subdirectory(src/metrology)

# 包含推理领域模块 (可选)
if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/infer")
    add_subdirectory(src/infer)
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
    # TensorRT 10 splits builder resources per architecture. Install only the selected partitions:
    # copying every `nvinfer_builder_resource*` would ship tensors for GPUs the profile does not
    # support. An unrecognised layout stops the configure instead of silently falling back.
    set(VISIONTOOLS_TRT_TARGET_SMS "89" CACHE STRING
        "TensorRT builder 资源目标 SM（逗号分隔，例如 89 或 89,86）；空表示不安装构建资源")
    option(VISIONTOOLS_TRT_INCLUDE_PTX
        "安装 PTX 硬件前向兼容资源（仅在发行承诺前向兼容时开启）" OFF)
    set(_trt_sms "")
    if(VISIONTOOLS_TRT_TARGET_SMS)
        string(REPLACE "," ";" _trt_sms "${VISIONTOOLS_TRT_TARGET_SMS}")
    endif()
    file(GLOB _trt_builder_all "${TRT_ROOT}/bin/nvinfer_builder_resource*.dll")
    set(_trt_selected "")
    set(_trt_unknown "")
    foreach(_trt_file IN LISTS _trt_builder_all)
        get_filename_component(_trt_name "${_trt_file}" NAME)
        if(_trt_name MATCHES "_ptx")
            if(VISIONTOOLS_TRT_INCLUDE_PTX)
                list(APPEND _trt_selected "${_trt_file}")
            endif()
            continue()
        endif()
        set(_trt_matched FALSE)
        foreach(_trt_sm IN LISTS _trt_sms)
            if(_trt_name MATCHES "_sm${_trt_sm}([_.-]|$)")
                set(_trt_matched TRUE)
            endif()
        endforeach()
        if(_trt_matched)
            list(APPEND _trt_selected "${_trt_file}")
        elseif(NOT _trt_name MATCHES "_sm[0-9]+")
            # No architecture marker at all: either a monolithic SDK or an unknown partition scheme.
            list(APPEND _trt_unknown "${_trt_name}")
        endif()
    endforeach()
    if(_trt_unknown)
        message(FATAL_ERROR
            "无法识别的 TensorRT builder 资源布局，拒绝按通配复制：${_trt_unknown}\n"
            "请升级到按架构分区的 TensorRT 10 布局，或为该版本补充显式选择规则。")
    endif()
    if(_trt_selected)
        install(FILES ${_trt_selected} DESTINATION bin COMPONENT VisionToolsRuntime)
        list(LENGTH _trt_selected _trt_count)
        message(STATUS "TensorRT builder 资源：安装 ${_trt_count} 个文件，目标 SM=${VISIONTOOLS_TRT_TARGET_SMS}，PTX=${VISIONTOOLS_TRT_INCLUDE_PTX}")
    endif()
endif()

# 包含契约与单元测试
if(VISIONTOOLS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 工具包构建元数据：时刻取自发行模块产物，配置时刻不能冒充构建时刻（Ticket VT-01 / S45-03）。
set(_vt_build_info_deps visiontools_metrology visiontools_preprocess)
if(TARGET visiontools_infer)
    list(APPEND _vt_build_info_deps visiontools_infer)
endif()

add_custom_target(visiontools_build_info ALL
    COMMAND ${CMAKE_COMMAND}
        -DCOMPONENT=VisionTools
        -DVERSION=${PROJECT_VERSION}
        -DCOMMIT=${VISIONTOOLS_COMMIT}
        -DBRANCH=${VISIONTOOLS_BRANCH}
        -DREPO_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DSCAN_DIR=${CMAKE_CURRENT_BINARY_DIR}/src
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/build_info.json
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteBuildInfo.cmake
    COMMENT "记录 VisionTools 构建完成时刻"
    VERBATIM)
add_dependencies(visiontools_build_info ${_vt_build_info_deps})

# 打印构建配置摘要
visiontools_print_summary()

