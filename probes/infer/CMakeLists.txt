cmake_minimum_required(VERSION 3.25)
project(VisionToolsInferProbe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus)
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(VisionRTToolSDK CONFIG REQUIRED)
find_package(InferRT CONFIG REQUIRED COMPONENTS model)
find_package(CUDAToolkit REQUIRED)

get_filename_component(VISIONTOOLS_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

set(_gen_inc "${CMAKE_BINARY_DIR}/generated")
if(NOT EXISTS "${_gen_inc}/PortConstants.h" AND EXISTS "${VISIONTOOLS_SOURCE_ROOT}/build/generated/PortConstants.h")
    set(_gen_inc "${VISIONTOOLS_SOURCE_ROOT}/build/generated")
endif()

if(NOT TARGET visiontools_infer)
    find_library(VISIONTOOLS_INFER_IMPLIB NAMES visiontools_infer PATHS
        "${CMAKE_BINARY_DIR}/src/infer"
        "${CMAKE_BINARY_DIR}/lib"
        "${VISIONTOOLS_SOURCE_ROOT}/build/src/infer"
        "${VISIONTOOLS_SOURCE_ROOT}/build/lib"
        "${VISIONTOOLS_SOURCE_ROOT}/dist/VisionTools/lib"
    )
    find_file(VISIONTOOLS_INFER_DLL NAMES visiontools_infer.dll PATHS
        "${CMAKE_BINARY_DIR}/src/infer"
        "${CMAKE_BINARY_DIR}/bin"
        "${VISIONTOOLS_SOURCE_ROOT}/build/src/infer"
        "${VISIONTOOLS_SOURCE_ROOT}/build/bin"
        "${VISIONTOOLS_SOURCE_ROOT}/dist/VisionTools/bin"
    )
    if(VISIONTOOLS_INFER_IMPLIB)
        add_library(visiontools_infer SHARED IMPORTED)
        set_target_properties(visiontools_infer PROPERTIES
            IMPORTED_IMPLIB "${VISIONTOOLS_INFER_IMPLIB}"
        )
        if(VISIONTOOLS_INFER_DLL)
            set_target_properties(visiontools_infer PROPERTIES
                IMPORTED_LOCATION "${VISIONTOOLS_INFER_DLL}"
            )
        endif()
    endif()
endif()

add_executable(visiontools_infer_probe main.cpp)
target_include_directories(visiontools_infer_probe PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
if(EXISTS "${VISIONTOOLS_SOURCE_ROOT}/src/infer/YoloModel.h")
    target_include_directories(visiontools_infer_probe PRIVATE
        "${VISIONTOOLS_SOURCE_ROOT}/src"
        "${VISIONTOOLS_SOURCE_ROOT}/src/infer"
        "${_gen_inc}"
    )
endif()
target_link_libraries(visiontools_infer_probe PRIVATE
    visiontools_infer
    VisionRT::ToolSDK
    ${OpenCV_LIBS}
    InferRT::model
    CUDA::cudart
)
if(WIN32)
    target_link_libraries(visiontools_infer_probe PRIVATE version)
endif()
file(TO_CMAKE_PATH "${InferRT_DIR}" VISIONRT_PROBE_SDK_DIR)
target_compile_definitions(visiontools_infer_probe PRIVATE VISIONRT_PROBE_SDK_DIR="${VISIONRT_PROBE_SDK_DIR}")
