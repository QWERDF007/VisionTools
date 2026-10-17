include_guard(GLOBAL)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_inferrt_root
    VARIABLES INFERRT_ROOT InferRT_ROOT INFERRT_DIR InferRT_DIR
    ENVIRONMENT_VARIABLES INFERRT_ROOT InferRT_ROOT INFERRT_DIR InferRT_DIR
    REQUIRED_FILE "lib/cmake/InferRT/InferRTConfig.cmake"
    DEFAULT "F:/Projects/InferRT/InferRT-0.0.3"
)

if(_vt_inferrt_root)
    set(INFERRT_ROOT "${_vt_inferrt_root}" CACHE PATH "InferRT installation root" FORCE)
    set(INFERRT_ROOT "${_vt_inferrt_root}")

    file(GLOB _installed_versions LIST_DIRECTORIES true "${_vt_inferrt_root}/InferRT-*")
    set(_candidates)
    foreach(_ver_dir IN LISTS _installed_versions)
        list(APPEND _candidates "${_ver_dir}/lib/cmake/InferRT")
    endforeach()
    list(APPEND _candidates
        "${_vt_inferrt_root}/lib/cmake/InferRT"
        "${_vt_inferrt_root}/lib/cmake"
        "${_vt_inferrt_root}/cmake"
        "${_vt_inferrt_root}/build"
        "${_vt_inferrt_root}"
    )
    foreach(_candidate IN LISTS _candidates)
        if(EXISTS "${_candidate}/InferRTConfig.cmake")
            set(InferRT_DIR "${_candidate}" CACHE PATH "InferRT CMake package directory" FORCE)
            set(InferRT_DIR "${_candidate}")
            break()
        endif()
    endforeach()
    list(PREPEND CMAKE_PREFIX_PATH "${_vt_inferrt_root}")
endif()
message(STATUS "InferRT Root resolved: '${_vt_inferrt_root}', InferRT_DIR: '${InferRT_DIR}'")
find_package(InferRT CONFIG REQUIRED COMPONENTS model)
if(NOT TARGET InferRT::model)
    message(FATAL_ERROR "InferRT package must export InferRT::model")
endif()

# Locate runtime DLLs
if(InferRT_DIR MATCHES "/lib/cmake/InferRT$")
    file(REAL_PATH "${InferRT_DIR}/../../.." VISIONTOOLS_INFERRT_ROOT)
elseif(DEFINED INFERRT_ROOT)
    set(VISIONTOOLS_INFERRT_ROOT "${INFERRT_ROOT}")
endif()

if(EXISTS "${VISIONTOOLS_INFERRT_ROOT}/bin")
    set(INFERRT_BIN_DIR "${VISIONTOOLS_INFERRT_ROOT}/bin")
endif()

if(INFERRT_BIN_DIR)
    file(GLOB INFERRT_DLLS "${INFERRT_BIN_DIR}/inferrt_*.dll")
endif()

visiontools_resolve_path(_vt_trt_root
    VARIABLES TRT_ROOT TensorRT_ROOT TENSORRT_DIR TRT_DIR
    ENVIRONMENT_VARIABLES TRT_ROOT TensorRT_ROOT TENSORRT_DIR TRT_DIR
    REQUIRED_FILE "lib/nvinfer_10.lib"
    DEFAULT "D:/Software/dev/TensorRT-10.16.1.11"
)
if(_vt_trt_root)
    set(TRT_ROOT "${_vt_trt_root}" CACHE PATH "TensorRT installation root" FORCE)
    set(TRT_ROOT "${_vt_trt_root}")
endif()
