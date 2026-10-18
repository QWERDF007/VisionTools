include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

# 统一 yaml-cpp 路径解析单一真相源 (SSOT)
# 优先级：CMake 显式变量 (-D) > 环境变量 > 平台默认基准路径
visiontools_resolve_path(YAML_CPP_DIR
    VARIABLES YAML_CPP_DIR yaml-cpp_DIR
    ENVIRONMENT_VARIABLES YAML_CPP_DIR yaml-cpp_DIR
    PATHS
        "D:/Software/anaconda3/Library"
        "D:/Software/anaconda3/Library/share/cmake/yaml-cpp"
        "C:/dev/yaml-cpp"
    DEFAULT "D:/Software/anaconda3/Library"
    REQUIRED_FILES "share/cmake/yaml-cpp/yaml-cpp-config.cmake" "include/yaml-cpp/yaml.h" "yaml-cpp-config.cmake"
)

if(YAML_CPP_DIR)
    if(EXISTS "${YAML_CPP_DIR}/yaml-cpp-config.cmake")
        find_package(yaml-cpp CONFIG REQUIRED PATHS "${YAML_CPP_DIR}" NO_DEFAULT_PATH)
    elseif(EXISTS "${YAML_CPP_DIR}/share/cmake/yaml-cpp/yaml-cpp-config.cmake")
        find_package(yaml-cpp CONFIG REQUIRED PATHS "${YAML_CPP_DIR}/share/cmake/yaml-cpp" NO_DEFAULT_PATH)
    else()
        find_package(yaml-cpp CONFIG REQUIRED PATHS "${YAML_CPP_DIR}" NO_DEFAULT_PATH)
    endif()
else()
    find_package(yaml-cpp CONFIG REQUIRED)
endif()

if(NOT TARGET yaml-cpp AND TARGET yaml-cpp::yaml-cpp)
    add_library(yaml-cpp ALIAS yaml-cpp::yaml-cpp)
endif()
