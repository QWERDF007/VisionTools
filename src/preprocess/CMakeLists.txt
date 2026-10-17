add_library(visiontools_preprocess SHARED
    ReadImageTool.cpp
    RoiProjectionTool.cpp
    ImageCropTool.cpp
    ImagePreprocessTool.cpp
    DetectionFilterTool.cpp
    AdaptiveBinaryTool.cpp
    PreprocessModule.cpp
)

target_compile_definitions(visiontools_preprocess PRIVATE VISIONTOOLS_EXPORTS)
target_include_directories(visiontools_preprocess PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../generated"
)

target_link_libraries(visiontools_preprocess PRIVATE
    VisionRT::ToolSDK
    ${OpenCV_LIBS}
)

set_target_properties(visiontools_preprocess PROPERTIES
    PREFIX ""
    OUTPUT_NAME visiontools_preprocess
)
