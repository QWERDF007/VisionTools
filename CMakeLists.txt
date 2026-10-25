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

