include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/ConfigDependencyDefaults.cmake")

visiontools_resolve_path(_vt_sdk_root
    VARIABLES VisionRTToolSDK_DIR VISIONRT_SDK_ROOT VisionRT_DIR
    ENVIRONMENT_VARIABLES VisionRTToolSDK_DIR VISIONRT_SDK_ROOT VisionRT_DIR
    PATHS ${CMAKE_PREFIX_PATH}
    DEFAULT "C:/dev/visionrt-sdk"
)

if(_vt_sdk_root)
    list(PREPEND CMAKE_PREFIX_PATH "${_vt_sdk_root}")
endif()

find_package(VisionRTToolSDK CONFIG QUIET)

if(TARGET VisionRT::ToolSDK)
    message(STATUS "Found installed VisionRTToolSDK: ${VisionRTToolSDK_DIR}")
else()
    message(STATUS "VisionRTToolSDK not found in CMAKE_PREFIX_PATH, providing contract SDK target from shared specification...")
    # Provide conforming standalone SDK target aligned with shared/SDK接口合同.md
    if(NOT TARGET VisionRT::ToolSDK)
        add_library(visionrt_toolsdk STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sdk/Value.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sdk/Tool.cpp"
        )
        target_include_directories(visionrt_toolsdk PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/sdk/include>"
            "$<INSTALL_INTERFACE:include>"
            ${OpenCV_INCLUDE_DIRS}
        )
        target_link_libraries(visionrt_toolsdk PUBLIC ${OpenCV_LIBS})
        add_library(VisionRT::ToolSDK ALIAS visionrt_toolsdk)
    endif()
endif()
