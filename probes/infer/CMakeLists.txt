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
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_executable(visiontools_infer_probe main.cpp)
target_include_directories(visiontools_infer_probe PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# 公开 probe 动态加载 visiontools_infer，仅依赖公开 SDK、OpenCV、CUDA 与 yaml-cpp (YS-04 / YVT2-04)
target_link_libraries(visiontools_infer_probe PRIVATE
    VisionRT::ToolSDK
    ${OpenCV_LIBS}
    CUDA::cudart
    yaml-cpp::yaml-cpp
)
if(WIN32)
    target_link_libraries(visiontools_infer_probe PRIVATE version)
endif()
target_compile_definitions(visiontools_infer_probe PRIVATE VISIONRT_SDK_VERSION="${VisionRTToolSDK_VERSION}")
