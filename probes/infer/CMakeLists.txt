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

add_executable(visiontools_infer_probe main.cpp)
target_include_directories(visiontools_infer_probe PRIVATE
    "${VISIONTOOLS_SOURCE_ROOT}/src"
    "${VISIONTOOLS_SOURCE_ROOT}/src/infer"
    "${VISIONTOOLS_SOURCE_ROOT}/src/generated"
    ${OpenCV_INCLUDE_DIRS}
)
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
