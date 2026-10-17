cmake_minimum_required(VERSION 3.25)
project(VisionTools VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 引入构建系统与编译器配置模块
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConfigCompiler)
include(ConfigDependencyDefaults)
include(ConfigOpenCV)
include(ConfigVisionRTToolSDK)
include(AddTest)
include(PrintConfig)

# 功能与测试构建开关
option(VISIONTOOLS_BUILD_TESTS "Build contract and unit tests" ON)
option(VISIONTOOLS_BUILD_INFER "Build InferRT GPU tools module" ON)

if(VISIONTOOLS_BUILD_INFER)
    include(ConfigCUDA)
    include(ConfigInferRT)
endif()

# 包含预处理领域模块
add_subdirectory(src/preprocess)

# 包含推理领域模块 (可选)
if(VISIONTOOLS_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/infer")
    add_subdirectory(src/infer)
endif()

# 包含契约与单元测试
if(VISIONTOOLS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 打印构建配置摘要
visiontools_print_summary()
