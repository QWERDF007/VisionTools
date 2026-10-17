cmake_minimum_required(VERSION 3.25)
project(VisionRTInferProbe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus)
endif()
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(InferRT CONFIG REQUIRED COMPONENTS model)
find_package(CUDAToolkit REQUIRED)
get_filename_component(VISIONRT_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Same production source, no Qt, SDK source checkout, private headers, or second
# preprocessor/decoder. Only this internal DLL links InferRT::model.
add_library(visionrt_probe_values STATIC
    "${VISIONRT_SOURCE_ROOT}/src/common/Value.cpp"
    "${VISIONRT_SOURCE_ROOT}/src/core/Tool.cpp")
target_include_directories(visionrt_probe_values PUBLIC "${VISIONRT_SOURCE_ROOT}/src" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visionrt_probe_values PUBLIC ${OpenCV_LIBS})
add_library(visionrt_infer_math STATIC "${VISIONRT_SOURCE_ROOT}/src/infer/YoloPostprocess.cpp")
target_include_directories(visionrt_infer_math PUBLIC "${VISIONRT_SOURCE_ROOT}/src")
add_library(VisionRTInfer SHARED
    "${VISIONRT_SOURCE_ROOT}/src/infer/InferModule.cpp"
    "${VISIONRT_SOURCE_ROOT}/src/infer/YoloModel.cpp")
target_compile_definitions(VisionRTInfer PRIVATE VISIONRT_INFER_EXPORTS)
if(DEFINED InferRT_VERSION)
    target_compile_definitions(VisionRTInfer PRIVATE VISIONRT_INFERRT_VERSION="${InferRT_VERSION}")
endif()
target_link_libraries(VisionRTInfer PRIVATE visionrt_probe_values visionrt_infer_math InferRT::model CUDA::cudart)
set_target_properties(VisionRTInfer PROPERTIES PREFIX "" OUTPUT_NAME VisionRTInfer)
add_executable(visionrt_infer_probe main.cpp)
target_link_libraries(visionrt_infer_probe PRIVATE VisionRTInfer visionrt_probe_values visionrt_infer_math)
if(WIN32)
    target_link_libraries(visionrt_infer_probe PRIVATE version)
endif()
file(TO_CMAKE_PATH "${InferRT_DIR}" VISIONRT_PROBE_SDK_DIR)
target_compile_definitions(visionrt_infer_probe PRIVATE VISIONRT_PROBE_SDK_DIR="${VISIONRT_PROBE_SDK_DIR}")
