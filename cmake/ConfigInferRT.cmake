include_guard(GLOBAL)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_inferrt_path
    VARIABLES INFERRT_ROOT InferRT_ROOT INFERRT_DIR InferRT_DIR
    ENVIRONMENT_VARIABLES INFERRT_ROOT InferRT_ROOT INFERRT_DIR InferRT_DIR
    REQUIRED_FILES "InferRTConfig.cmake" "lib/cmake/InferRT/InferRTConfig.cmake"
    DEPENDENCY inferrt
)

if(_vt_inferrt_path)
    if(EXISTS "${_vt_inferrt_path}/InferRTConfig.cmake")
        set(InferRT_DIR "${_vt_inferrt_path}" CACHE PATH "InferRT CMake package directory" FORCE)
        set(InferRT_DIR "${_vt_inferrt_path}")
        if(_vt_inferrt_path MATCHES "/lib/cmake/InferRT$")
            file(REAL_PATH "${_vt_inferrt_path}/../../.." _inferred_root)
            set(INFERRT_ROOT "${_inferred_root}" CACHE PATH "InferRT installation root" FORCE)
            set(INFERRT_ROOT "${_inferred_root}")
        else()
            set(INFERRT_ROOT "${_vt_inferrt_path}" CACHE PATH "InferRT installation root" FORCE)
            set(INFERRT_ROOT "${_vt_inferrt_path}")
        endif()
    elseif(EXISTS "${_vt_inferrt_path}/lib/cmake/InferRT/InferRTConfig.cmake")
        set(INFERRT_ROOT "${_vt_inferrt_path}" CACHE PATH "InferRT installation root" FORCE)
        set(INFERRT_ROOT "${_vt_inferrt_path}")
        set(InferRT_DIR "${_vt_inferrt_path}/lib/cmake/InferRT" CACHE PATH "InferRT CMake package directory" FORCE)
        set(InferRT_DIR "${_vt_inferrt_path}/lib/cmake/InferRT")
    else()
        set(INFERRT_ROOT "${_vt_inferrt_path}" CACHE PATH "InferRT installation root" FORCE)
        set(INFERRT_ROOT "${_vt_inferrt_path}")
        file(GLOB _installed_versions LIST_DIRECTORIES true "${_vt_inferrt_path}/InferRT-*")
        set(_candidates)
        foreach(_ver_dir IN LISTS _installed_versions)
            list(APPEND _candidates "${_ver_dir}/lib/cmake/InferRT")
        endforeach()
        list(APPEND _candidates
            "${_vt_inferrt_path}/lib/cmake/InferRT"
            "${_vt_inferrt_path}/lib/cmake"
            "${_vt_inferrt_path}/cmake"
            "${_vt_inferrt_path}/build"
            "${_vt_inferrt_path}"
        )
        foreach(_candidate IN LISTS _candidates)
            if(EXISTS "${_candidate}/InferRTConfig.cmake")
                set(InferRT_DIR "${_candidate}" CACHE PATH "InferRT CMake package directory" FORCE)
                set(InferRT_DIR "${_candidate}")
                break()
            endif()
        endforeach()
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${INFERRT_ROOT}")
endif()
message(STATUS "InferRT Root resolved: '${INFERRT_ROOT}', InferRT_DIR: '${InferRT_DIR}'")
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
    DEPENDENCY tensorrt
)
if(_vt_trt_root)
    set(TRT_ROOT "${_vt_trt_root}" CACHE PATH "TensorRT installation root" FORCE)
    set(TRT_ROOT "${_vt_trt_root}")
endif()
