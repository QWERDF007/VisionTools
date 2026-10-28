add_library(visiontools_common STATIC
    GeometryTransform.cpp
    RoiSupport.cpp
    errors/ToolError.cpp
)

target_include_directories(visiontools_common PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(visiontools_common PUBLIC
    VisionRT::Contracts
    ${OpenCV_LIBS}
)
