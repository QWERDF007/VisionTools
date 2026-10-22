cmake_minimum_required(VERSION 3.25)
project(VisionToolsInferProbe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus)
endif()

if(NOT TARGET opencv_core)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
endif()
if(NOT TARGET VisionRT::Contracts)
    find_package(VisionRTContracts CONFIG REQUIRED)
endif()
if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(visiontools_infer_probe main.cpp)
target_include_directories(visiontools_infer_probe PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# 公开 probe 动态加载 visiontools_infer，仅依赖公开 SDK、OpenCV 与 CUDA (YS-04 / YVT2-04)
target_link_libraries(visiontools_infer_probe PRIVATE
    VisionRT::Contracts
    ${OpenCV_LIBS}
    CUDA::cudart
)
if(WIN32)
    target_link_libraries(visiontools_infer_probe PRIVATE version)
endif()
